import time
import jax
import jax.numpy as jp
from mujodico import mjx
from mujodico.examples.methods import support
from mujodico.examples.tasks import Task
import numpy as np

try:
  import cyipopt
except ImportError as e:
  raise ImportError(
      "Optional solver dependency not installed. Install it with:\n\n"
      "    pip install cyipopt"
  ) from e


class IpoptMultipleShooting:

  def __init__(
      self,
      task: Task,
      *,
      gauss_newton: bool = True,
  ):
    self.task = task

    model = task.model

    mjx_model = mjx.put_model(
        model,
        impl='jax',
        contact_approximation=task.contact_approx,
        contact_smoothing=0.0,
    )

    mjx_smooth_model = mjx.put_model(
        model,
        impl=mjx_model.impl,
        contact_approximation=task.contact_approx,
        contact_smoothing=task.contact_smoothing,
    )

    mjx_data = mjx.make_data(mjx_model, impl=mjx_model.impl)

    N = task.horizon
    nx = task.nx
    nu = task.nu

    x_goal = jp.array(task.x_goal)
    Q = jp.array(task.Q)
    Qf = jp.array(task.Qf)
    u_ref = jp.array(task.u_ref)
    R = jp.array(task.R)

    def step(x, u):
      nq, nv = mjx_model.nq, mjx_model.nv
      d = mjx_data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)

      d0 = mjx.step(mjx_model, d)
      xnext = jp.concatenate((d0.qpos, d0.qvel, d0.act))

      d1 = mjx.step(mjx_smooth_model, d)
      xnext_smooth = jp.concatenate((d1.qpos, d1.qvel, d1.act))

      return (
          jax.lax.stop_gradient(xnext)
          + xnext_smooth
          - jax.lax.stop_gradient(xnext_smooth)
      )

    def stage_objective(x, u):
      return (
          support.state_cost(Q, x_goal, x, task.quaternion_w_idxs)
          + (u - u_ref).dot(R @ (u - u_ref))
      )

    def stage_constraint(x, u, xnext):
      return xnext - step(x, u)

    def stage_lagrangian(var, scale1, scale0):
      assert var.size == nx + nu + nx
      assert scale1.size == nx
      assert scale0.size == 1

      x = var[:nx]
      u = var[nx : nx + nu]
      xnext = var[nx + nu :]

      lagrangian = stage_objective(x, u) * scale0
      if not gauss_newton:
        lagrangian += jp.sum(stage_constraint(x, u, xnext) * scale1)

      return lagrangian

    def objective(vars: jax.Array):
      xs, us = self._unpack(vars)

      per_stage_cost = jax.vmap(stage_objective)(xs[:-1], us)

      cost = (
          jp.sum(per_stage_cost)
          + support.state_cost(Qf, x_goal, xs[-1], task.quaternion_w_idxs)
      )
      return cost

    def constraints(vars: jax.Array):
      xs, us = self._unpack(vars)

      per_stage_constraint = jax.vmap(stage_constraint)(xs[:-1], us, xs[1:])
      return per_stage_constraint.ravel()

    def constraints_jacobian_entries(vars: jax.Array):
      xs, us = self._unpack(vars)

      def stage_constraints_jac(x, u):
        A, B = jax.jacfwd(step, argnums=(0, 1))(x, u)
        return jp.concatenate([(-A).reshape(-1), (-B).reshape(-1), jp.ones(nx)])

      per_stage_constraints_jac = jax.vmap(stage_constraints_jac)(xs[:-1], us)
      return per_stage_constraints_jac.ravel()

    def lagrangian_hessian_entries(vars, lagrange, obj_factor):
      xs, us = self._unpack(vars)

      lagrange = lagrange.reshape(N, nx)

      def stage_lagrangian_hess(x, u, xnext, scale1, scale0):
        var = jp.concatenate([x.ravel(), u.ravel(), xnext.ravel()])
        hess = jax.hessian(stage_lagrangian)(var, scale1, scale0)
        return hess[jp.tril_indices(hess.shape[0])]

      per_stage_lagrangian_hess = jax.vmap(stage_lagrangian_hess, in_axes=(0,0,0,0,None))(
          xs[:-1], us, xs[1:], lagrange, obj_factor
      )

      return jp.concatenate([
          per_stage_lagrangian_hess.ravel(),
          Qf[jp.tril_indices(nx)] * 2,
      ])

    vars_trace = self._pack(xs=jp.zeros((N+1, nx)), us=jp.zeros((N, nu)))

    objective_grad = jax.jit(jax.grad(objective)).lower(vars_trace).compile()
    objective = jax.jit(objective).lower(vars_trace).compile()

    constraints = jax.jit(constraints).lower(vars_trace).compile()
    constraints_jacobian_entries = jax.jit(constraints_jacobian_entries).lower(vars_trace).compile()
    constraints_jacobian_structure = self._constraints_jacobian_structure()

    lagrangian_hessian_entries = jax.jit(lagrangian_hessian_entries).lower(
        vars_trace, jp.ones(N * nx), jp.array(1.0),
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

      @timed
      def objective(self, vars):
        return float(objective(jp.asarray(vars)))

      @timed
      def gradient(self, vars):
        return np.asarray(objective_grad(jp.asarray(vars)))

      @timed
      def constraints(self, vars):
        return np.asarray(constraints(jp.asarray(vars)))

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
            )
        )

      @timed
      def hessianstructure(self):
        return lagrangian_hessian_structure

    x_lb, x_ub = task.x_bounds
    xs_lb = np.repeat(x_lb[None, :], N + 1, axis=0)
    xs_lb[0] = task.x_0
    xs_ub = np.repeat(x_ub[None, :], N + 1, axis=0)
    xs_ub[0] = task.x_0

    u_lb, u_ub = task.u_bounds
    us_lb = np.repeat(u_lb[None, :], N, axis=0)
    us_ub = np.repeat(u_ub[None, :], N, axis=0)

    vars_lb = self._pack(xs_lb, us_lb)
    vars_ub = self._pack(xs_ub, us_ub)

    n_constraints = N * nx
    constraints_lb = np.zeros(n_constraints)
    constraints_ub = np.zeros(n_constraints)

    self.problem = cyipopt.Problem(
        n=vars_lb.size,
        m=n_constraints,
        problem_obj=NLP(),
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

    vars_opt, info = self.problem.solve(vars_init)

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
    N, nx, nu = self.task.horizon, self.task.nx, self.task.nu
    assert xs.shape == (N + 1, nx)
    assert us.shape == (N, nu)

    cp = jp if isinstance(xs, jax.Array) else np

    return cp.concatenate([
        cp.hstack((xs[:-1], us)).ravel(),
        xs[-1],
    ])

  def _unpack(self, vars: jax.Array | np.ndarray):
    N, nx, nu = self.task.horizon, self.task.nx, self.task.nu
    n_vars = (N + 1) * nx + N * nu
    assert vars.shape == (n_vars,)

    cp = jp if isinstance(vars, jax.Array) else np

    xus = vars[:-nx].reshape(N, nx + nu)
    xs = cp.vstack((xus[:, :nx], vars[-nx:]))
    us = xus[:, nx:]

    return xs, us

  def _constraints_jacobian_structure(self):
    N, nx, nu = self.task.horizon, self.task.nx, self.task.nu

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
    N, nx, nu = self.task.horizon, self.task.nx, self.task.nu

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
