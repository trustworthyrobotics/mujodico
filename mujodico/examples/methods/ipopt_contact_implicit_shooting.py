import time
import importlib
import logging
import jax
import jax.numpy as jp
from mujodico import mjx
from mujodico.examples.tasks import Task
from mujodico.examples.methods import support
from mujodico.mjx._src import constraint
from mujodico.mjx._src import solver_conic
from mujodico.trajopt._src import transcriptions
import numpy as np


_CONSTITUTIVE_RELATIONS_FUNC = {
    mjx.ContactApproxType.CONIC: solver_conic.constitutive_relations,
}


class IpoptContactImplicitShooting:

  def __init__(
      self,
      task: Task,
  ):
    try:
      self._cyipopt = importlib.import_module('cyipopt')
      logging.getLogger('cyipopt').setLevel(logging.WARNING)
    except ImportError as e:
      raise ImportError(
          'Optional solver dependency not installed. Install it with:\n\n'
          '    pip install cyipopt'
      ) from e

    self.task = task

    model = task.model

    contact_approx = mjx.ContactApproxType.CONIC

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

    efc_type = constraint.make_efc_type(mjx_smooth_model)
    if jp.any(efc_type == mjx.ConstraintType.CONTACT_ELLIPTIC):
      raise NotImplementedError(
          'IpoptKKTContactImplicit does not support ellictic cone yet'
      )

    efc_contact_mask = jp.bitwise_or(
        efc_type == mjx.ConstraintType.CONTACT_FRICTIONLESS,
        efc_type == mjx.ConstraintType.CONTACT_PYRAMIDAL,
    )

    nc = int(jp.count_nonzero(efc_contact_mask))
    self.nc = nc

    def advance(x, u, cf):
      # TODO(wei-chen): consider using computed force with stop gradient instead of zero
      efc_force = jp.zeros(mjx_data._impl.nefc)
      efc_force = efc_force.at[efc_contact_mask].set(cf)

      return transcriptions._step_given_efc_force(mjx_model, x, u, efc_force)

    contact_approx = mjx.ContactApproxType._resolve(contact_approx)
    if contact_approx not in _CONSTITUTIVE_RELATIONS_FUNC:
      raise NotImplementedError(
          f'constitutive relation for {contact_approx} contact approximation not implemented.'
      )
    constitutive_relations_func = _CONSTITUTIVE_RELATIONS_FUNC[contact_approx]

    constitutive_fn, constitutive_lb, constitutive_ub = constitutive_relations_func(mjx_smooth_model, efc_contact_mask)

    def objective(vars: jax.Array):
      xs, us, cfs = self._unpack(vars)

      du = us - u_ref

      cost = (
          jp.sum(jax.vmap(lambda x: support.state_cost(Q, x_goal, x, task.quaternion_w_idxs))(xs[:-1]))
          + jp.sum((du @ R) * du)
          + support.state_cost(Qf, x_goal, xs[-1], task.quaternion_w_idxs)
      )
      return cost

    def constraints(vars: jax.Array):
      xs, us, cfs = self._unpack(vars)

      def stage(x, u, cf, xnext):
        return jp.concatenate([
          xnext - advance(x, u, cf),
          constitutive_fn(x, u, cf, xnext),
        ])

      per_stage = jax.vmap(stage)(xs[:-1], us, cfs, xs[1:])
      return per_stage.ravel()

    def constraints_jac_entries(vars: jax.Array):
      xs, us, cfs = self._unpack(vars)

      def stage_jac(x, u, cf, xnext):
        A, B, C = jax.jacfwd(advance, argnums=(0, 1, 2))(x, u, cf)
        jacobians = jax.jacfwd(constitutive_fn, argnums=(0, 1, 2, 3))(x, u, cf, xnext)
        return jp.concatenate([
          (-A).ravel(),
          (-B).ravel(),
          (-C).ravel(),
          jp.ones(nx),
          jp.concatenate(jacobians, axis=1).ravel(),
        ])

      per_stage = jax.vmap(stage_jac)(xs[:-1], us, cfs, xs[1:])
      return per_stage.ravel()

    vars_trace = self._pack(
      xs=jp.zeros((N+1, nx)),
      us=jp.zeros((N, nu)),
      cfs=jp.zeros((N, nc)),
    )

    objective_grad = jax.jit(jax.grad(objective)).lower(vars_trace).compile()
    objective = jax.jit(objective).lower(vars_trace).compile()
    objective_hess_structure, objective_hess_entries = self._objective_hessian()

    constraints = jax.jit(constraints).lower(vars_trace).compile()
    constraints_jac_entries = jax.jit(constraints_jac_entries).lower(vars_trace).compile()
    constraints_jac_structure = self._constraint_jacobian_structure(constitutive_lb.size)

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

    cfs_lb = np.zeros((N, nc))
    cfs_ub = np.full((N, nc), np.inf)

    vars_lb = self._pack(xs_lb, us_lb, cfs_lb)
    vars_ub = self._pack(xs_ub, us_ub, cfs_ub)

    constraints_lb = np.tile(np.concatenate([
      np.zeros(nx),
      np.asarray(constitutive_lb)
    ]), N)
    constraints_ub = np.tile(np.concatenate([
      np.zeros(nx),
      np.asarray(constitutive_ub)
    ]), N)

    self.problem = self._cyipopt.Problem(
        n=vars_lb.size,
        m=constraints_lb.size,
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
    xs_init = traj_init['xs']
    us_init = traj_init['us']
    cfs_init = traj_init.get('cfs', np.full((self.task.horizon, self.nc), 1e-6))

    vars_init = self._pack(xs_init, us_init, cfs_init)

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

    xs_opt, us_opt, cfs_opt = self._unpack(vars_opt)

    return {
        'xs': xs_opt,
        'us': us_opt,
        'cfs': cfs_opt,
    }

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

  def _unpack(self, vars: jax.Array | np.ndarray):
    N, nx, nu, nc = self.task.horizon, self.task.nx, self.task.nu, self.nc
    n_vars = (N + 1) * nx + N * nu + N * nc
    assert vars.shape == (n_vars,)

    cp = jp if isinstance(vars, jax.Array) else np

    xucs = vars[:-nx].reshape(N, nx + nu + nc)
    xs = cp.vstack((xucs[:, :nx], vars[-nx:]))
    us = xucs[:, nx : nx + nu]
    cfs = xucs[:, nx + nu :]

    return xs, us, cfs

  def _constraint_jacobian_structure(self, n_constitutive_per_stage: int):
    N, nx, nu, nc = self.task.horizon, self.task.nx, self.task.nu, self.nc

    xx_rows, xx_cols = np.indices((nx, nx))
    xu_rows, xu_cols = np.indices((nx, nu))
    xc_rows, xc_cols = np.indices((nx, nc))
    x_eye = np.arange(nx)
    constitutive_rows, constitutive_cols = np.indices((n_constitutive_per_stage, nx + nu + nc + nx))

    xx_rows, xx_cols = xx_rows.ravel(), xx_cols.ravel()
    xu_rows, xu_cols = xu_rows.ravel(), xu_cols.ravel()
    xc_rows, xc_cols = xc_rows.ravel(), xc_cols.ravel()
    constitutive_rows = constitutive_rows.ravel()
    constitutive_cols = constitutive_cols.ravel()

    stage_rows = np.concatenate([
      xx_rows,
      xu_rows,
      xc_rows,
      x_eye,
      nx + constitutive_rows,
    ])
    stage_cols = np.concatenate([
      xx_cols,
      nx + xu_cols,
      nx + nu + xc_cols,
      nx + nu + nc + x_eye,
      constitutive_cols,
    ])

    irow = []
    jcol = []

    for k in range(N):
      row_offset = k * (nx + n_constitutive_per_stage)
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
