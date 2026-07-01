import jax
import jax.numpy as jp
from mujodico import mjx
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
    nx = task.nx
    nu = task.nu

    x_goal = jp.array(task.x_goal)
    Q = jp.array(task.Q)
    Qf = jp.array(task.Qf)
    u_ref = jp.array(task.u_ref)
    R = jp.array(task.R)

    efc_type = mjx_data._impl.efc_type
    if jp.any(efc_type == mjx.ConstraintType.CONTACT_ELLIPTIC):
      raise NotImplementedError(
          'IpoptLiftedMultipleShooting does not support ellictic cone yet'
      )

    efc_contact_mask = jp.bitwise_or(
        efc_type == mjx.ConstraintType.CONTACT_FRICTIONLESS,
        efc_type == mjx.ConstraintType.CONTACT_PYRAMIDAL,
    )

    nc = int(jp.count_nonzero(efc_contact_mask))
    self.nc = nc

    @jax.jit
    def forward(x, u):
      nq, nv = mjx_model.nq, mjx_model.nv
      d = mjx_data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)

      d0 = mjx.forward(mjx_model, d)
      cf = d0._impl.efc_force[efc_contact_mask]

      d1 = mjx.forward(mjx_smooth_model, d)
      cf_smooth = d1._impl.efc_force[efc_contact_mask]

      return jax.lax.stop_gradient(cf) + cf_smooth - jax.lax.stop_gradient(cf_smooth)

    @jax.jit
    def advance(x, u, cf):
      m = mjx_model
      nq, nv = m.nq, m.nv
      d = mjx_data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)

      d = mjx.fwd_position(m, d)
      d = mjx.fwd_velocity(m, d)
      d = mjx.fwd_actuation(m, d)
      d = mjx.fwd_acceleration(m, d)

      # TODO(wei-chen): consider using computed force with stop gradient instead of zero
      efc_force = jp.zeros(d._impl.nefc)
      efc_force = efc_force.at[efc_contact_mask].set(cf)

      d = d.tree_replace({
          'qfrc_constraint': d._impl.efc_J.T @ efc_force,
          '_impl.efc_force': efc_force,
          'qacc': None,
      })

      if m.opt.integrator == mjx.IntegratorType.EULER:
        d = mjx.euler(m, d)
      elif m.opt.integrator == mjx.IntegratorType.IMPLICITFAST:
        d = mjx.implicit(m, d)
      else:
        raise NotImplementedError(f'integrator {m.opt.integrator} not implemented.')

      return jp.concatenate((d.qpos, d.qvel, d.act))

    @jax.jit
    def objective(vars: jax.Array):
      xs, us, cfs = self._unpack(vars)

      dx = xs[:-1] - x_goal
      dxf = xs[-1] - x_goal
      du = us - u_ref

      cost = jp.sum((dx @ Q) * dx) + jp.sum((du @ R) * du) + dxf @ Qf @ dxf
      return cost

    @jax.jit
    def constraints(vars: jax.Array):
      xs, us, cfs = self._unpack(vars)

      def stage(x, u, cf, xnext):
        return jp.concatenate([
          xnext - advance(x, u, cf),
          cf - forward(x, u),
        ])

      per_stage = jax.vmap(stage)(xs[:-1], us, cfs, xs[1:])
      return per_stage.ravel()

    @jax.jit
    def constraints_jac_entries(vars: jax.Array):
      xs, us, cfs = self._unpack(vars)

      def stage_jac(x, u, cf):
        A, B, C = jax.jacfwd(advance, argnums=(0, 1, 2))(x, u, cf)
        Ac, Bc = jax.jacfwd(forward, argnums=(0, 1))(x, u)
        return jp.concatenate([
          (-A).reshape(-1),
          (-B).reshape(-1),
          (-C).reshape(-1),
          jp.ones(nx),
          (-Ac).reshape(-1),
          (-Bc).reshape(-1),
          jp.ones(nc),
        ])

      per_stage = jax.vmap(stage_jac)(xs[:-1], us, cfs)
      return per_stage.ravel()

    objective_grad = jax.jit(jax.grad(objective))
    objective_hess_structure, objective_hess_entries = self._objective_hessian()

    constraints_jac_structure = self._constraint_jacobian_structure()

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
    xs_lb = np.repeat(x_lb[None, :], N + 1, axis=0)
    xs_lb[0] = task.x_0
    xs_ub = np.repeat(x_ub[None, :], N + 1, axis=0)
    xs_ub[0] = task.x_0

    u_lb, u_ub = task.u_bounds
    us_lb = np.repeat(u_lb[None, :], N, axis=0)
    us_ub = np.repeat(u_ub[None, :], N, axis=0)

    cfs_lb = np.zeros((N, nc))
    cfs_ub = np.full((N, nc), np.inf)

    vars_lb = self._pack(xs_lb, us_lb, cfs_lb)
    vars_ub = self._pack(xs_ub, us_ub, cfs_ub)

    n_constraints = N * (nx + nc)
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
    cfs_init = np.zeros((self.task.horizon, self.nc))

    vars_init = self._pack(xs_init, us_init, cfs_init)

    self.problem.add_option('max_iter', iterations)

    vars_opt, info = self.problem.solve(vars_init)

    xs_opt, us_opt, cfs_opt = self._unpack(vars_opt)

    return np.asarray(xs_opt), np.asarray(us_opt)

  def _pack(
      self,
      xs: jax.Array | np.ndarray,
      us: jax.Array | np.ndarray,
      cfs: jax.Array | np.ndarray,
  ):
    N, nx, nu, nc = self.task.horizon, self.task.nx, self.task.nu, self.nc
    assert xs.shape == (N + 1, nx)
    assert us.shape == (N, nu)
    assert cfs.shape == (N, nc)

    cp = jp if isinstance(xs, jax.Array) else np

    return cp.concatenate([
        cp.hstack((xs[:-1], us, cfs)).ravel(),
        xs[-1],
    ])

  def _unpack(self, vars: jax.Array | jax.Array):
    N, nx, nu, nc = self.task.horizon, self.task.nx, self.task.nu, self.nc
    n_vars = (N + 1) * nx + N * nu + N * nc
    assert vars.shape == (n_vars,)

    cp = jp if isinstance(vars, jax.Array) else np

    xucs = vars[:-nx].reshape(N, nx + nu + nc)
    xs = cp.vstack((xucs[:, :nx], vars[-nx:]))
    us = xucs[:, nx : nx + nu]
    cfs = xucs[:, nx + nu :]

    return xs, us, cfs

  def _constraint_jacobian_structure(self):
    N, nx, nu, nc = self.task.horizon, self.task.nx, self.task.nu, self.nc

    xx_rows, xx_cols = np.indices((nx, nx))
    xu_rows, xu_cols = np.indices((nx, nu))
    xc_rows, xc_cols = np.indices((nx, nc))
    x_eye = np.arange(nx)
    cx_rows, cx_cols = np.indices((nc, nx))
    cu_rows, cu_cols = np.indices((nc, nu))
    c_eye = np.arange(nc)

    xx_rows, xx_cols = xx_rows.ravel(), xx_cols.ravel()
    xu_rows, xu_cols = xu_rows.ravel(), xu_cols.ravel()
    xc_rows, xc_cols = xc_rows.ravel(), xc_cols.ravel()
    cx_rows, cx_cols = cx_rows.ravel(), cx_cols.ravel()
    cu_rows, cu_cols = cu_rows.ravel(), cu_cols.ravel()

    stage_rows = np.concatenate([
      xx_rows,
      xu_rows,
      xc_rows,
      x_eye,
      nx + cx_rows,
      nx + cu_rows,
      nx + c_eye
    ])
    stage_cols = np.concatenate([
      xx_cols,
      nx + xu_cols,
      nx + nu + xc_cols,
      nx + nu + nc + x_eye,
      cx_cols,
      nx + cu_cols,
      nx + nu + c_eye,
    ])

    irow = []
    jcol = []

    for k in range(N):
      row_offset = k * (nx + nc)
      col_offset = k * (nx + nu + nc)

      irow.append(stage_rows + row_offset)
      jcol.append(stage_cols + col_offset)

    return (
        np.concatenate(irow).astype(np.int32),
        np.concatenate(jcol).astype(np.int32),
    )

  def _objective_hessian(self):
    N, nx, nu, nc = self.task.horizon, self.task.nx, self.task.nu, self.nc

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
      offset += nc

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
