import time
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

    mjx_model = mjx.put_model(
        model,
        impl='jax',
        contact_approximation=contact_approx,
        contact_smoothing=0.0,
    )

    mjx_smooth_model = mjx.put_model(
        model,
        impl=mjx_model.impl,
        contact_approximation=contact_approx,
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
      d = mjx.step(mjx_model, d)
      return jp.concatenate((d.qpos, d.qvel, d.act))

    def step_smooth(x, u):
      nq, nv = mjx_model.nq, mjx_model.nv
      d = mjx_data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)
      d = mjx.step(mjx_smooth_model, d)
      return jp.concatenate((d.qpos, d.qvel, d.act))

    def objective(vars: jax.Array):
      xs, us = self._unpack(vars)

      dx = xs[:-1] - x_goal
      dxf = xs[-1] - x_goal
      du = us - u_ref

      cost = jp.sum((dx @ Q) * dx) + jp.sum((du @ R) * du) + dxf @ Qf @ dxf
      return cost

    def constraints(vars: jax.Array):
      xs, us = self._unpack(vars)
      xs_next = jax.vmap(step)(xs[:-1], us)
      return (xs[1:] - xs_next).ravel()

    def constraints_jac_entries(vars: jax.Array):
      xs, us = self._unpack(vars)

      def stage_jac(x, u):
        A, B = jax.jacrev(step_smooth, argnums=(0, 1))(x, u)
        return jp.concatenate([(-A).reshape(-1), (-B).reshape(-1), jp.ones(nx)])

      per_stage = jax.vmap(stage_jac)(xs[:-1], us)
      return per_stage.ravel()

    vars_trace = self._pack(xs=jp.zeros((N+1, nx)), us=jp.zeros((N, nu)))

    objective_grad = jax.jit(jax.grad(objective)).lower(vars_trace).compile()
    objective = jax.jit(objective).lower(vars_trace).compile()
    objective_hess_structure, objective_hess_entries = self._objective_hessian()

    constraints = jax.jit(constraints).lower(vars_trace).compile()
    constraints_jac_entries = jax.jit(constraints_jac_entries).lower(vars_trace).compile()
    constraints_jac_structure = self._dynamics_constraint_jacobian_structure()

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
        return np.asarray(constraints_jac_entries(jp.asarray(vars)))

      @timed
      def jacobianstructure(self):
        return constraints_jac_structure

      @timed
      def hessian(self, vars, lmbda, objective_factor):
        return objective_hess_entries * objective_factor

      @timed
      def hessianstructure(self):
        return objective_hess_structure

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
      *,
      xs_init: np.ndarray,
      us_init: np.ndarray,
      iterations: int,
      **kwargs,
  ):
    vars_init = self._pack(xs_init, us_init)

    self.problem.add_option("max_iter", iterations)

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

  def _objective_hessian(self):
    N, nx, nu = self.task.horizon, self.task.nx, self.task.nu

    Q = 2 * np.asarray(self.task.Q)
    Qf = 2 * np.asarray(self.task.Qf)
    R = 2 * np.asarray(self.task.R)

    q_i, q_j = np.tril_indices(nx)
    qf_i, qf_j = np.tril_indices(nx)
    r_i, r_j = np.tril_indices(nu)

    q_entries = Q[q_i, q_j]
    qf_entries = Qf[qf_i, qf_j]
    r_entries = R[r_i, r_j]

    irow = []
    jcol = []
    entries = []

    offset = 0
    for _ in range(N):
      irow.append(q_i + offset)
      jcol.append(q_j + offset)
      entries.append(q_entries)
      offset += nx

      irow.append(r_i + offset)
      jcol.append(r_j + offset)
      entries.append(r_entries)
      offset += nu

    irow.append(qf_i + offset)
    jcol.append(qf_j + offset)
    entries.append(qf_entries)

    structure = (
        np.concatenate(irow).astype(np.int32),
        np.concatenate(jcol).astype(np.int32),
    )

    return (
        structure,
        np.concatenate(entries),
    )
