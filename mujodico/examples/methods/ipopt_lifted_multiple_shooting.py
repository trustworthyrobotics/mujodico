import time
import jax
import jax.numpy as jp
import mujoco
from mujodico import mjx
from mujodico.examples.methods import support
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

  def __init__(
      self,
      task: Task,
      contact_approx: mjx.ContactApproxType,
      hess_approx: support.HessianApproxType,
  ):
    self.task = task

    model = task.model

    mjx_model = mjx.put_model(
        model,
        impl='jax',
        contact_approximation=contact_approx,
        contact_smoothing=task.contact_smoothing * 0.001,
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

    efc_contact_addresses, socs = support.extract_contact_indices(mjx_model, mjx_data)

    ncf = int(efc_contact_addresses.size)
    self.ncf = ncf
    nsoc = len(socs)

    def step1(x, u):
      nq, nv = mjx_model.nq, mjx_model.nv
      d = mjx_data.replace(
          qpos=x[:nq], qvel=x[nq : nq + nv], act=x[nq + nv :], ctrl=u
      )

      d = mjx.forward(mjx_model, d)
      cf = d._impl.efc_force[efc_contact_addresses]

      return cf

    def step2(x, u, cf):
      # TODO(wei-chen): consider using computed force with stop gradient instead of zero
      efc_force = jp.zeros(mjx_data._impl.nefc)
      efc_force = efc_force.at[efc_contact_addresses].set(cf)

      return support.step_given_efc_force(mjx_model, x, u, efc_force)

    def soc_constraint(cf):
      constraints = jp.zeros(nsoc)
      for i, soc in enumerate(socs):
        cf_n = cf[soc.start_index]
        cf_t = cf[soc.start_index + 1 : soc.start_index + soc.dim] / soc.friction
        constraints = constraints.at[i].set(cf_n * cf_n - cf_t.dot(cf_t))
      return constraints

    def stage_objective(x, u, cf):
      return (x - x_goal).dot(Q @ (x - x_goal)) + (u - u_ref).dot(R @ (u - u_ref))

    def stage_constraint(x, u, cf, xnext):
      return jp.concatenate([
          xnext - step2(x, u, cf),
          cf - step1(x, u),
          soc_constraint(cf),
      ])

    def stage_lagrangian(var, scale1, scale0):
      assert var.size == nx + nu + ncf + nx
      assert scale1.size == nx + ncf + nsoc
      assert scale0.size == 1

      x = var[:nx]
      u = var[nx : nx + nu]
      cf = var[nx + nu : nx + nu + ncf]
      xnext = var[nx + nu + ncf :]

      lagrangian = stage_objective(x, u, cf) * scale0

      state_constraint_scale1 = stage_constraint(x, u, cf, xnext) * scale1
      if hess_approx == support.HessianApproxType.Full:
        lagrangian += jp.sum(state_constraint_scale1)
      else:
        lagrangian += jp.sum(state_constraint_scale1[nx + ncf:])

      return lagrangian

    def objective(vars: jax.Array):
      xs, us, cfs = self._unpack(vars)

      per_stage_cost = jax.vmap(stage_objective)(xs[:-1], us, cfs)

      cost = jp.sum(per_stage_cost) + (xs[-1] - x_goal).dot(Qf @ (xs[-1] - x_goal))
      return cost

    def constraints(vars: jax.Array):
      xs, us, cfs = self._unpack(vars)

      per_stage_constraint = jax.vmap(stage_constraint)(xs[:-1], us, cfs, xs[1:])
      return per_stage_constraint.ravel()

    def constraints_jacobian_entries(vars: jax.Array):
      xs, us, cfs = self._unpack(vars)

      def stage_constraints_jac(x, u, cf):
        A, B, C = jax.jacfwd(step2, argnums=(0, 1, 2))(x, u, cf)
        Acf, Bcf = jax.jacfwd(step1, argnums=(0, 1))(x, u)
        Jsoc = jax.jacfwd(soc_constraint)(cf)
        return jp.concatenate([
            (-A).ravel(),
            (-B).ravel(),
            (-C).ravel(),
            jp.ones(nx),
            (-Acf).ravel(),
            (-Bcf).ravel(),
            jp.ones(ncf),
            Jsoc.ravel(),
        ])

      per_stage_constraints_jac = jax.vmap(stage_constraints_jac)(xs[:-1], us, cfs)
      return per_stage_constraints_jac.ravel()

    def lagrangian_hessian_entries(vars, lagrange, obj_factor):
      xs, us, cfs = self._unpack(vars)

      lagrange = lagrange.reshape(N, nx + ncf + nsoc)
      obj_factor = jp.repeat(obj_factor, N)

      def stage_lagrangian_hess(x, u, cf, xnext, scale1, scale0):
        var = jp.concatenate([x.ravel(), u.ravel(), cf.ravel(), xnext.ravel()])
        hess = jax.hessian(stage_lagrangian)(var, scale1, scale0)
        return hess[jp.tril_indices(hess.shape[0])]

      per_stage_lagrangian_hess = jax.vmap(stage_lagrangian_hess)(
          xs[:-1], us, cfs, xs[1:], lagrange, obj_factor
      )

      return jp.concatenate([
          per_stage_lagrangian_hess.ravel(),
          Qf[jp.tril_indices(nx)] * 2,
      ])

    vars_trace = self._pack(
        xs=jp.zeros((N + 1, nx)),
        us=jp.zeros((N, nu)),
        cfs=jp.zeros((N, ncf)),
    )

    objective_grad = jax.jit(jax.grad(objective)).lower(vars_trace).compile()
    objective = jax.jit(objective).lower(vars_trace).compile()

    constraints = jax.jit(constraints).lower(vars_trace).compile()
    constraints_jacobian_entries = jax.jit(constraints_jacobian_entries).lower(vars_trace).compile()
    constraints_jacobian_structure = self._constraints_jacobian_structure(nsoc=nsoc)

    lagrangian_hessian_entries = jax.jit(lagrangian_hessian_entries).lower(
        vars_trace, jp.ones(N * (nx + ncf + nsoc)), jp.array(1.0),
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
        if hess_approx == support.HessianApproxType.LBFGS:
          raise AttributeError()  # cyipopt defaults to L-BFGS if hessian() throws

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

    cf_lb = np.zeros(ncf)
    cf_ub = np.full(ncf, np.inf)
    for soc in socs:
      cf_lb[soc.start_index + 1 : soc.start_index + soc.dim] = -np.inf
    cfs_lb = np.repeat(cf_lb[None, :], N, axis=0)
    cfs_ub = np.repeat(cf_ub[None, :], N, axis=0)

    vars_lb = self._pack(xs_lb, us_lb, cfs_lb)
    vars_ub = self._pack(xs_ub, us_ub, cfs_ub)

    constraint_lb = np.concatenate([np.zeros(nx + ncf), np.zeros(nsoc)])
    constraint_ub = np.concatenate([np.zeros(nx + ncf), np.full(nsoc, np.inf)])
    constraints_lb = np.tile(constraint_lb, N)
    constraints_ub = np.tile(constraint_ub, N)

    self.problem = cyipopt.Problem(
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
      *,
      xs_init: np.ndarray,
      us_init: np.ndarray,
      iterations: int,
      **kwargs,
  ):
    cfs_init = np.zeros((self.task.horizon, self.ncf))
    vars_init = self._pack(xs_init, us_init, cfs_init)

    self.problem.add_option('max_iter', iterations)

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

    return np.asarray(xs_opt), np.asarray(us_opt)

  def _pack(
      self,
      xs: jax.Array | np.ndarray,
      us: jax.Array | np.ndarray,
      cfs: jax.Array | np.ndarray,
  ):
    N, nx, nu, ncf = self.task.horizon, self.task.nx, self.task.nu, self.ncf
    assert xs.shape == (N + 1, nx)
    assert us.shape == (N, nu)
    assert cfs.shape == (N, ncf)

    cp = jp if isinstance(xs, jax.Array) else np

    return cp.concatenate([
        cp.hstack((xs[:-1], us, cfs)).ravel(),
        xs[-1],
    ])

  def _unpack(self, vars: jax.Array | np.ndarray):
    N, nx, nu, ncf = self.task.horizon, self.task.nx, self.task.nu, self.ncf
    n_vars = (N + 1) * nx + N * nu + N * ncf
    assert vars.shape == (n_vars,)

    cp = jp if isinstance(vars, jax.Array) else np

    xucs = vars[:-nx].reshape(N, nx + nu + ncf)
    xs = cp.vstack((xucs[:, :nx], vars[-nx:]))
    us = xucs[:, nx : nx + nu]
    cfs = xucs[:, nx + nu :]

    return xs, us, cfs

  def _constraints_jacobian_structure(self, nsoc: int):
    N, nx, nu, ncf = self.task.horizon, self.task.nx, self.task.nu, self.ncf

    xx_rows, xx_cols = np.indices((nx, nx))
    xu_rows, xu_cols = np.indices((nx, nu))
    xc_rows, xc_cols = np.indices((nx, ncf))
    x_eye = np.arange(nx)
    cx_rows, cx_cols = np.indices((ncf, nx))
    cu_rows, cu_cols = np.indices((ncf, nu))
    c_eye = np.arange(ncf)
    sc_rows, sc_cols = np.indices((nsoc, ncf))

    xx_rows, xx_cols = xx_rows.ravel(), xx_cols.ravel()
    xu_rows, xu_cols = xu_rows.ravel(), xu_cols.ravel()
    xc_rows, xc_cols = xc_rows.ravel(), xc_cols.ravel()
    cx_rows, cx_cols = cx_rows.ravel(), cx_cols.ravel()
    cu_rows, cu_cols = cu_rows.ravel(), cu_cols.ravel()
    sc_rows, sc_cols = sc_rows.ravel(), sc_cols.ravel()

    stage_rows = np.concatenate([
        xx_rows,
        xu_rows,
        xc_rows,
        x_eye,
        nx + cx_rows,
        nx + cu_rows,
        nx + c_eye,
        nx + ncf + sc_rows,
    ])
    stage_cols = np.concatenate([
        xx_cols,
        nx + xu_cols,
        nx + nu + xc_cols,
        nx + nu + ncf + x_eye,
        cx_cols,
        nx + cu_cols,
        nx + nu + c_eye,
        nx + nu + sc_cols,
    ])

    irow = []
    jcol = []

    for k in range(N):
      row_offset = k * (nx + ncf + nsoc)
      col_offset = k * (nx + nu + ncf)

      irow.append(stage_rows + row_offset)
      jcol.append(stage_cols + col_offset)

    return (
        np.concatenate(irow).astype(np.int32),
        np.concatenate(jcol).astype(np.int32),
    )

  def _lagrangian_hessian_structure(self):
    N, nx, nu, ncf = self.task.horizon, self.task.nx, self.task.nu, self.ncf

    stage_i, stage_j = np.tril_indices(nx + nu + ncf + nx)
    qf_i, qf_j = np.tril_indices(nx)

    irow = []
    jcol = []

    offset = 0
    for _ in range(N):
      irow.append(stage_i + offset)
      jcol.append(stage_j + offset)
      offset += nx + nu + ncf

    irow.append(qf_i + offset)
    jcol.append(qf_j + offset)

    return (
        np.concatenate(irow).astype(np.int32),
        np.concatenate(jcol).astype(np.int32),
    )
