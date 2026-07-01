import jax
import jax.numpy as jp
from mujodico import mjx
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

  def __init__(self, task: Task, contact_approx: mjx.ContactApproxType):
    self.task = task

    model = task.model

    mjx_smooth_model = mjx.put_model(
        model,
        contact_approximation=contact_approx,
        contact_smoothing=task.contact_smoothing,
    )

    mjx_model = mjx.put_model(
        model,
        contact_approximation=contact_approx,
        contact_smoothing=0.0,
    )

    mjx_data = mjx.make_data(mjx_model)

    N = task.horizon
    nq = model.nq
    nx = task.nx
    nu = task.nu

    x_goal = jp.array(task.x_goal)
    Q = jp.array(task.Q)
    Qf = jp.array(task.Qf)
    u_ref = jp.array(task.u_ref)
    R = jp.array(task.R)

    @jax.jit
    def step(x, u):
      d = mjx_data.replace(qpos=x[:nq], qvel=x[nq:], ctrl=u)
      d = mjx.step(mjx_model, d)
      return jp.concatenate((d.qpos, d.qvel))

    @jax.jit
    def step_smooth(x, u):
      d = mjx_data.replace(qpos=x[:nq], qvel=x[nq:], ctrl=u)
      d = mjx.step(mjx_smooth_model, d)
      return jp.concatenate((d.qpos, d.qvel))

    @jax.jit
    def objective(vars: jax.Array):
      xs, us = self._unpack(vars)

      dx = xs[:-1] - x_goal
      dxf = xs[-1] - x_goal
      du = us - u_ref

      cost = jp.sum((dx @ Q) * dx) + jp.sum((du @ R) * du) + dxf @ Qf @ dxf
      return cost

    @jax.jit
    def constraints(vars: jax.Array):
      xs, us = self._unpack(vars)
      xs_next = jax.vmap(step)(xs[:-1], us)
      return (xs[1:] - xs_next).ravel()

    @jax.jit
    def constraints_jac_entries(vars: jax.Array):
      xs, us = self._unpack(vars)

      def stage_jac(x, u):
        A, B = jax.jacrev(step_smooth, argnums=(0, 1))(x, u)
        return jp.concatenate([(-A).reshape(-1), (-B).reshape(-1), jp.ones(nx)])

      per_stage = jax.vmap(stage_jac)(xs[:-1], us)
      return per_stage.reshape(-1)

    objective_grad = jax.jit(jax.grad(objective))
    objective_hess_structure = self._objective_hessian_structure()
    objective_hess_entries = self._objective_hessian_entries()

    constraints_jac_structure = self._dynamics_constraint_jacobian_structure()

    class NLP:

      def objective(self, vars):
        return float(objective(jp.asarray(vars)))

      def gradient(self, vars):
        return np.asarray(objective_grad(jp.asarray(vars)))

      def constraints(self, vars):
        return np.asarray(constraints(jp.asarray(vars)))

      def jacobian(self, vars):
        return np.asarray(constraints_jac_entries(jp.asarray(vars)))

      def jacobianstructure(self):
        return constraints_jac_structure

      def hessian(self, vars, lmbda, objective_factor):
        return objective_hess_entries * objective_factor

      def hessianstructure(self):
        return objective_hess_structure

    x_lb, x_ub = task.x_bounds
    u_lb, u_ub = task.u_bounds

    xs_lb = np.repeat(x_lb[None, :], N + 1, axis=0)
    xs_lb[0] = task.x_0
    xs_ub = np.repeat(x_ub[None, :], N + 1, axis=0)
    xs_ub[0] = task.x_0

    us_lb = np.repeat(u_lb[None, :], N, axis=0)
    us_ub = np.repeat(u_ub[None, :], N, axis=0)

    vars_lb = self._pack(xs_lb, us_lb)
    vars_ub = self._pack(xs_ub, us_ub)

    constraints_lb = np.zeros(N * nx)
    constraints_ub = np.zeros(N * nx)

    self.problem = cyipopt.Problem(
        n=(N + 1) * nx + N * nu,
        m=N * nx,
        problem_obj=NLP(),
        lb=vars_lb,
        ub=vars_ub,
        cl=constraints_lb,
        cu=constraints_ub,
    )

  def solve(
      self,
      *,
      xs_init: np.ndarray,
      us_init: np.ndarray,
      iterations: int,
      **kwargs,
  ):
    vars_init = self._pack(xs_init, us_init)

    self.problem.add_option("max_iter", iterations)

    vars_opt, info = self.problem.solve(vars_init)

    xs_opt, us_opt = self._unpack(vars_opt)

    return np.asarray(xs_opt), np.asarray(us_opt)

  def _pack(self, xs: jax.Array | np.ndarray, us: jax.Array | np.ndarray):
    N, nx, nu = self.task.horizon, self.task.nx, self.task.nu
    assert xs.shape == (N + 1, nx)
    assert us.shape == (N, nu)

    cp = jp if isinstance(xs, jax.Array) else np

    return cp.concatenate([
        cp.hstack((xs[:-1], us)).ravel(),
        xs[-1],
    ])

  def _unpack(self, vars: jax.Array | jax.Array):
    N, nx, nu = self.task.horizon, self.task.nx, self.task.nu
    n_vars = (N + 1) * nx + N * nu
    assert vars.shape == (n_vars,)

    cp = jp if isinstance(vars, jax.Array) else np

    xus = vars[:-nx].reshape(N, nx + nu)
    xs = cp.vstack((xus[:, :nx], vars[-nx:]))
    us = xus[:, nx:]

    return xs, us

  def _dynamics_constraint_jacobian_structure(self):
    N, nx, nu = self.task.horizon, self.task.nx, self.task.nu

    x_rows, x_cols = np.indices((nx, nx))
    u_rows, u_cols = np.indices((nx, nu))
    eye = np.arange(nx)

    x_rows = x_rows.ravel()
    x_cols = x_cols.ravel()
    u_rows = u_rows.ravel()
    u_cols = u_cols.ravel()

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

  def _objective_hessian_structure(self):
    N, nx, nu = self.task.horizon, self.task.nx, self.task.nu

    q_i, q_j = np.tril_indices(nx)
    r_i, r_j = np.tril_indices(nu)

    irow = []
    jcol = []

    offset = 0
    for k in range(N + 1):
      # Q (or Qf) block
      irow.append(q_i + offset)
      jcol.append(q_j + offset)
      offset += nx

      if k == N:
        break

      # R block
      irow.append(r_i + offset)
      jcol.append(r_j + offset)
      offset += nu

    return (
        np.concatenate(irow).astype(np.int32),
        np.concatenate(jcol).astype(np.int32),
    )

  def _objective_hessian_entries(self):
    N = self.task.horizon

    Q = 2 * np.asarray(self.task.Q)
    Qf = 2 * np.asarray(self.task.Qf)
    R = 2 * np.asarray(self.task.R)

    q_entries = Q[np.tril_indices_from(Q)]
    r_entries = R[np.tril_indices_from(R)]
    qf_entries = Qf[np.tril_indices_from(Qf)]

    entries = np.concatenate((
        np.repeat(np.concatenate((q_entries, r_entries)), N),
        qf_entries,
    ))

    return entries
