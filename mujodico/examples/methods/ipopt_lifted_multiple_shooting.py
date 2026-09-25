import time
import jax
import jax.numpy as jp
from mujodico import mjx
from mujodico.examples.methods.lifted_multiple_shooting_transcription import (
    LiftedMultipleShootingTranscription
)
from mujodico.examples.tasks import Task
import numpy as np

try:
  import cyipopt
except ImportError as e:
  raise ImportError(
      'Optional solver dependency not installed. Install it with:\n\n'
      '    pip install cyipopt'
  ) from e


class IpoptLiftedMultipleShooting:
  # TODO(wei-chen): Handle terminal equality and inequality constraints.

  def __init__(
      self,
      task: Task,
      *,
      gauss_newton: bool = True,
  ):
    self.transcription = LiftedMultipleShootingTranscription(task)
    trans = self.transcription

    N = trans.N
    nx = trans.nx
    nw = trans.nw

    n_stage_eq = trans.n_stage_eq
    n_stage_ineq = trans.n_stage_ineq

    def stage_lagrangian(var, eq_scale, ineq_scale, cost_scale, params):
      assert len(var) == nx + nw + nx
      x = var[:nx]
      w = var[nx:nx+nw]
      xnext = var[nx+nw:]
      value = cost_scale * trans.stage_cost(x, w, params)
      if not gauss_newton:
        eq = trans.stage_equality_constraint(x, w, xnext, params)
        ineq = trans.stage_inequality_constraint(x, w, params)
        value += jp.vdot(eq_scale, eq)
        value += jp.vdot(ineq_scale, ineq)
      return value

    def objective(vars, params):
      xs, ws = self._unpack(vars)
      stage_costs = jax.vmap(trans.stage_cost, in_axes=(0,0,None))(xs[:-1], ws, params)
      terminal_cost = trans.terminal_cost(xs[-1], params)
      return jp.sum(stage_costs) + terminal_cost

    def constraints(vars, params):
      xs, ws = self._unpack(vars)
      stage_eqs = jax.vmap(trans.stage_equality_constraint, in_axes=(0,0,0,None))(xs[:-1], ws, xs[1:], params)
      stage_ineqs = jax.vmap(trans.stage_inequality_constraint, in_axes=(0,0,None))(xs[:-1], ws, params)
      return jp.concatenate([stage_eqs, stage_ineqs], axis=1).ravel()

    def constraints_jacobian_entries(vars, params):
      xs, ws = self._unpack(vars)
      def stage_jac(x, w, xnext):
        local = jp.concatenate([x, w, xnext])
        def local_constraints(var):
          x = var[:nx]
          w = var[nx:nx+nw]
          xnext = var[nx+nw:]
          return jp.concatenate([
              trans.stage_equality_constraint(x, w, xnext, params),
              trans.stage_inequality_constraint(x, w, params)
          ])
        return jax.jacfwd(local_constraints)(local).ravel()
      return jax.vmap(stage_jac)(xs[:-1], ws, xs[1:]).ravel()

    def lagrangian_hessian_entries(vars, lagrange, obj_factor, params):
      xs, ws = self._unpack(vars)
      stage_lagrange = lagrange[:N * (n_stage_eq + n_stage_ineq)].reshape(N, n_stage_eq + n_stage_ineq)
      eq_lagrange = stage_lagrange[:, :n_stage_eq]
      ineq_lagrange = stage_lagrange[:, n_stage_eq:]
      def stage_hess(x, w, xnext, eq_lam, ineq_lam, obj_scale, params):
        local = jp.concatenate([x, w, xnext])
        hess = jax.hessian(stage_lagrangian)(local, eq_lam, ineq_lam, obj_scale, params)
        return hess[jp.tril_indices(hess.shape[0])]
      stage_hessians = jax.vmap(stage_hess, in_axes=(0,0,0,0,0,None,None))(
          xs[:-1], ws, xs[1:], eq_lagrange, ineq_lagrange, obj_factor, params,
      )
      terminal_hess = params.Qf[jp.tril_indices(nx)] * 2 * obj_factor
      return jp.concatenate([stage_hessians.ravel(), terminal_hess])

    vars_trace = self._pack(
        xs=jp.zeros((N + 1, nx)),
        ws=jp.zeros((N, nw)),
    )
    params_trace = trans.params

    objective_grad = jax.jit(jax.grad(objective)).lower(vars_trace, params_trace).compile()
    objective = jax.jit(objective).lower(vars_trace, params_trace).compile()

    constraints = jax.jit(constraints).lower(vars_trace, params_trace).compile()
    constraints_jacobian_entries = jax.jit(constraints_jacobian_entries).lower(vars_trace, params_trace).compile()
    constraints_jacobian_structure = self._constraints_jacobian_structure()

    lagrangian_hessian_entries = jax.jit(lagrangian_hessian_entries).lower(
        vars_trace, jp.ones(N * (n_stage_eq + n_stage_ineq)), jp.array(1.0),
        self.transcription.params,
    ).compile()
    lagrangian_hessian_structure = self._lagrangian_hessian_structure()

    self._dynamics_time = 0

    def timed(fn):
      def wrapped(*args, **kwargs):
        tic = time.perf_counter()
        out = fn(*args, **kwargs)
        self._dynamics_time += time.perf_counter() - tic
        return out
      return wrapped

    class NLP:
      def __init__(self, params):
        self.params = params

      @timed
      def objective(self, vars):
        return float(objective(jp.asarray(vars), self.params))

      @timed
      def gradient(self, vars):
        return np.asarray(objective_grad(jp.asarray(vars), self.params))

      @timed
      def constraints(self, vars):
        return np.asarray(constraints(jp.asarray(vars), self.params))

      @timed
      def jacobian(self, vars):
        return np.asarray(constraints_jacobian_entries(jp.asarray(vars), self.params))

      @timed
      def jacobianstructure(self):
        return constraints_jacobian_structure

      @timed
      def hessian(self, vars, lagrange, objective_factor):
        return np.asarray(
            lagrangian_hessian_entries(
                jp.asarray(vars),
                jp.asarray(lagrange),
                jp.asarray(objective_factor),
                self.params,
            )
        )

      @timed
      def hessianstructure(self):
        return lagrangian_hessian_structure

    self._NLP = NLP
    self._refresh_problem()

  def update(self, task: Task):
    self.transcription.update(task)
    self._refresh_problem()

  def _refresh_problem(self):
    trans = self.transcription
    N = trans.N

    x_lb, x_ub = trans.x_bounds
    xs_lb = np.repeat(np.asarray(x_lb)[None, :], N + 1, axis=0)
    xs_ub = np.repeat(np.asarray(x_ub)[None, :], N + 1, axis=0)
    xs_lb[0] = np.asarray(trans.params.x_0)
    xs_ub[0] = np.asarray(trans.params.x_0)

    w_lb, w_ub = trans.w_bounds
    ws_lb = np.repeat(np.asarray(w_lb)[None, :], N, axis=0)
    ws_ub = np.repeat(np.asarray(w_ub)[None, :], N, axis=0)

    vars_lb = self._pack(xs_lb, ws_lb)
    vars_ub = self._pack(xs_ub, ws_ub)

    n_stage_eq = trans.n_stage_eq
    n_stage_ineq = trans.n_stage_ineq

    #  Each inequality row is represented g(z) <= 0.
    stage_lb = np.concatenate([np.zeros(n_stage_eq), np.full(n_stage_ineq, -np.inf)])
    stage_ub = np.zeros(n_stage_eq + n_stage_ineq)
    constraints_lb = np.tile(stage_lb, N)
    constraints_ub = np.tile(stage_ub, N)

    self.problem = cyipopt.Problem(
        n=vars_lb.size,
        m=constraints_lb.size,
        problem_obj=self._NLP(self.transcription.params),
        lb=vars_lb,
        ub=vars_ub,
        cl=constraints_lb,
        cu=constraints_ub,
    )

  def solve(
      self,
      traj_init: dict,
      *,
      iterations: int = 500,
      verbose: bool = True,
  ):
    xs_init = traj_init['xs']
    us_init = traj_init['us']

    N = self.transcription.N
    nw = self.transcription.nw
    nu = us_init.shape[-1]

    cfs_init = traj_init.get('cfs', np.zeros((N, nw - nu)))

    ws_init = np.hstack([us_init, cfs_init])
    vars_init = self._pack(xs_init, ws_init)

    self.problem.add_option('max_iter', iterations)
    if not verbose:
      self.problem.add_option('print_level', 0)

    self._dynamics_time = 0
    tic = time.perf_counter()

    vars_opt, info = self.problem.solve(vars_init)

    total_time = time.perf_counter() - tic

    print(f'''
Method: {self.__class__.__name__}
  total time: {total_time:.6f} sec
    dynamics time: {self._dynamics_time:.6f} sec / {self._dynamics_time/total_time*100:.2f}%
    ipopt time:    {total_time-self._dynamics_time:.6f} sec / {(1-self._dynamics_time/total_time)*100:.2f}%
''')

    xs_opt, ws_opt = self._unpack(vars_opt)
    us_opt = ws_opt[:, :nu]
    cfs_opt = ws_opt[:, nu:]

    return {
        'xs': xs_opt,
        'us': us_opt,
        'cfs': cfs_opt,
    }

  def _pack(self, xs: jax.Array | np.ndarray, ws: jax.Array | np.ndarray):
    N = self.transcription.N
    nx = self.transcription.nx
    nw = self.transcription.nw
    assert xs.shape == (N + 1, nx)
    assert ws.shape == (N, nw)
    cp = jp if isinstance(xs, jax.Array) else np
    return cp.concatenate([cp.hstack((xs[:-1], ws)).ravel(), xs[-1]])

  def _unpack(self, vars: jax.Array | np.ndarray):
    N = self.transcription.N
    nx = self.transcription.nx
    nw = self.transcription.nw
    assert vars.shape == ((N + 1) * nx + N * nw,)
    cp = jp if isinstance(vars, jax.Array) else np
    stage = vars[:-nx].reshape(N, nx + nw)
    xs = cp.vstack((stage[:, :nx], vars[-nx:]))
    ws = stage[:, nx:]
    return xs, ws

  def _constraints_jacobian_structure(self):
    trans = self.transcription

    N = trans.N
    nx = trans.nx
    nw = trans.nw

    n_stage_constraints = trans.n_stage_eq + trans.n_stage_ineq

    stage_i, stage_j = np.indices((n_stage_constraints, nx + nw + nx))
    stage_i, stage_j = stage_i.ravel(), stage_j.ravel()

    irow = []
    jcol = []
    for k in range(N):
      irow.append(stage_i + k * n_stage_constraints)
      jcol.append(stage_j + k * (nx + nw))

    return np.concatenate(irow), np.concatenate(jcol)

  def _lagrangian_hessian_structure(self):
    N = self.transcription.N
    nx = self.transcription.nx
    nw = self.transcription.nw

    stage_i, stage_j = np.tril_indices(nx + nw + nx)
    qf_i, qf_j = np.tril_indices(nx)

    irow = []
    jcol = []

    offset = 0
    for _ in range(N):
      irow.append(stage_i + offset)
      jcol.append(stage_j + offset)
      offset += nx + nw

    irow.append(qf_i + offset)
    jcol.append(qf_j + offset)

    return np.concatenate(irow), np.concatenate(jcol)
