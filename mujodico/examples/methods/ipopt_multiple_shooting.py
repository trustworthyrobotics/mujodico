import time
import jax
import jax.numpy as jp
from mujodico.examples import methods
from mujodico.examples.methods import support
from mujodico.examples.tasks import Task
import numpy as np

class IpoptMultipleShooting:

  def __init__(
      self,
      task: Task,
      *,
      gauss_newton: bool = True,
  ):
    self._cyipopt = support.load_cyipopt()

    self.transcription = methods.MultipleShootingTranscription(task)
    trans = self.transcription

    N = trans.N
    nx = trans.nx
    nu = trans.nu

    def stage_lagrangian(var, constr_scale, cost_scale, params):
      assert var.size == nx + nu + nx
      x = var[:nx]
      u = var[nx : nx + nu]
      xnext = var[nx + nu :]
      value = trans.stage_cost(x, u, params) * cost_scale
      if not gauss_newton:
        constr = trans.stage_equality_constraint(x, u, xnext, params)
        value += jp.vdot(constr, constr_scale)
      return value

    def objective(vars: jax.Array, params):
      xs, us = self._unpack(vars)
      stage_costs = jax.vmap(trans.stage_cost, in_axes=(0,0,None))(xs[:-1], us, params)
      terminal_cost = trans.terminal_cost(xs[-1], params)
      return jp.sum(stage_costs) + terminal_cost

    def constraints(vars: jax.Array, params):
      xs, us = self._unpack(vars)
      per_stage_constraint = jax.vmap(
          trans.stage_equality_constraint, in_axes=(0,0,0,None)
      )(xs[:-1], us, xs[1:], params)
      return per_stage_constraint.ravel()

    def constraints_jacobian_entries(vars: jax.Array):
      xs, us = self._unpack(vars)

      def stage_constraints_jac(x, u, xnext):
        var = jp.concatenate([x, u, xnext])

        def local_constraint(var):
          x = var[:nx]
          u = var[nx:nx+nu]
          xnext = var[nx+nu:]
          return trans.stage_equality_constraint(x, u, xnext, trans.params)

        jacobian = jax.jacfwd(local_constraint)(var)
        return jp.concatenate([
          jacobian[:, :nx].ravel(),
          jacobian[:, nx:nx+nu].ravel(),
          jp.diag(jacobian[:, nx+nu:]),
        ])

      return jax.vmap(stage_constraints_jac)(xs[:-1], us, xs[1:]).ravel()

    def lagrangian_hessian_entries(vars, lagrange, obj_factor, params):
      xs, us = self._unpack(vars)

      lagrange = lagrange.reshape(N, nx)

      def stage_lagrangian_hess(x, u, xnext, constr_scale, cost_scale, params):
        var = jp.concatenate([x.ravel(), u.ravel(), xnext.ravel()])
        hess = jax.hessian(stage_lagrangian)(var, constr_scale, cost_scale, params)
        return hess[jp.tril_indices(hess.shape[0])]

      per_stage_lagrangian_hess = jax.vmap(
          stage_lagrangian_hess, in_axes=(0,0,0,0,None,None)
      )(
          xs[:-1], us, xs[1:], lagrange, obj_factor, params
      )
      terminal_hess = jax.hessian(trans.terminal_cost)(xs[-1], params)

      return jp.concatenate([
          per_stage_lagrangian_hess.ravel(),
          terminal_hess[jp.tril_indices(nx)] * obj_factor,
      ])

    vars_trace = self._pack(xs=jp.zeros((N+1, nx)), us=jp.zeros((N, nu)))

    params_trace = trans.params
    objective_grad = jax.jit(jax.grad(objective)).lower(vars_trace, params_trace).compile()
    objective = jax.jit(objective).lower(vars_trace, params_trace).compile()

    constraints = jax.jit(constraints).lower(vars_trace, params_trace).compile()
    constraints_jacobian_entries = jax.jit(constraints_jacobian_entries).lower(vars_trace).compile()
    constraints_jacobian_structure = self._constraints_jacobian_structure()

    lagrangian_hessian_entries = jax.jit(lagrangian_hessian_entries).lower(
      vars_trace, jp.ones(N * nx), jp.array(1.0), params_trace,
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
        return np.asarray(constraints_jacobian_entries(jp.asarray(vars)))

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
    xs_lb[0] = np.asarray(trans.params.x_0)
    xs_ub = np.repeat(np.asarray(x_ub)[None, :], N + 1, axis=0)
    xs_ub[0] = np.asarray(trans.params.x_0)

    u_lb, u_ub = trans.u_bounds
    us_lb = np.repeat(np.asarray(u_lb)[None, :], N, axis=0)
    us_ub = np.repeat(np.asarray(u_ub)[None, :], N, axis=0)

    vars_lb = self._pack(xs_lb, us_lb)
    vars_ub = self._pack(xs_ub, us_ub)

    n_constraints = N * trans.nx
    constraints_lb = np.zeros(n_constraints)
    constraints_ub = np.zeros(n_constraints)

    self.problem = self._cyipopt.Problem(
        n=vars_lb.size,
        m=n_constraints,
        problem_obj=self._NLP(trans.params),
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
    vars_init = self._pack(
      xs=traj_init['xs'],
      us=traj_init['us'],
    )

    self.problem.add_option('max_iter', iterations)
    if not verbose:
      self.problem.add_option('print_level', 0)

    self._dynamics_time = 0
    tic = time.perf_counter()

    vars_opt, _info = self.problem.solve(vars_init)

    total_time = time.perf_counter() - tic

    print(f'''
Method: {self.__class__.__name__}
  total time: {total_time:.6f} sec
    dynamics time: {self._dynamics_time:.6f} sec / {self._dynamics_time/total_time*100:.2f}%
    ipopt time:    {total_time-self._dynamics_time:.6f} sec / {(1-self._dynamics_time/total_time)*100:.2f}%
''')

    xs_opt, us_opt = self._unpack(vars_opt)

    return {
        'xs': xs_opt,
        'us': us_opt,
    }

  def _pack(self, xs: jax.Array | np.ndarray, us: jax.Array | np.ndarray):
    N, nx, nu = self.transcription.N, self.transcription.nx, self.transcription.nu
    assert xs.shape == (N + 1, nx)
    assert us.shape == (N, nu)

    cp = jp if isinstance(xs, jax.Array) else np

    return cp.concatenate([
        cp.hstack((xs[:-1], us)).ravel(),
        xs[-1],
    ])

  def _unpack(self, vars: jax.Array | np.ndarray):
    N, nx, nu = self.transcription.N, self.transcription.nx, self.transcription.nu
    n_vars = (N + 1) * nx + N * nu
    assert vars.shape == (n_vars,)

    cp = jp if isinstance(vars, jax.Array) else np

    xus = vars[:-nx].reshape(N, nx + nu)
    xs = cp.vstack((xus[:, :nx], vars[-nx:]))
    us = xus[:, nx:]

    return xs, us

  def _constraints_jacobian_structure(self):
    N, nx, nu = self.transcription.N, self.transcription.nx, self.transcription.nu

    x_rows, x_cols = np.indices((nx, nx))
    u_rows, u_cols = np.indices((nx, nu))
    eye = np.arange(nx)

    x_rows, x_cols = x_rows.ravel(), x_cols.ravel()
    u_rows, u_cols = u_rows.ravel(), u_cols.ravel()

    stage_rows = np.concatenate([x_rows, u_rows, eye])
    stage_cols = np.concatenate([x_cols, nx + u_cols, nx + nu + eye])

    irow = []
    jcol = []

    for k in range(N):
      row_offset = k * nx
      col_offset = k * (nx + nu)

      irow.append(stage_rows + row_offset)
      jcol.append(stage_cols + col_offset)

    return (
        np.concatenate(irow).astype(np.int32),
        np.concatenate(jcol).astype(np.int32),
    )

  def _lagrangian_hessian_structure(self):
    N, nx, nu = self.transcription.N, self.transcription.nx, self.transcription.nu

    stage_i, stage_j = np.tril_indices(nx + nu + nx)
    qf_i, qf_j = np.tril_indices(nx)

    irow = []
    jcol = []

    offset = 0
    for _ in range(N):
      irow.append(stage_i + offset)
      jcol.append(stage_j + offset)
      offset += nx + nu

    irow.append(qf_i + offset)
    jcol.append(qf_j + offset)

    return (
        np.concatenate(irow).astype(np.int32),
        np.concatenate(jcol).astype(np.int32),
    )
