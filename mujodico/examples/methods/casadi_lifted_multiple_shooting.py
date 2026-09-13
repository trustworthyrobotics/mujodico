import time

import jax
import jax.numpy as jp
import numpy as np

from mujodico import mjx
from mujodico.examples.methods import support
from mujodico.examples.tasks import Task

try:
  import casadi as ca
  from mujodico.examples.methods import casadi_support
except ImportError as e:
  raise ImportError(
      "Optional solver dependency not installed. Install it with:\n\n"
      "    pip install casadi"
  ) from e


class _CasadiLiftedMultipleShooting:

  def __init__(
      self,
      task: Task,
      contact_approx: mjx.ContactApproxType,
  ):
    self.task = task
    model = task.model

    N = task.horizon
    nx = task.nx
    nu = task.nu

    mjx_model = mjx.put_model(
        model,
        impl='jax',
        contact_approximation=contact_approx,
        contact_smoothing=task.contact_smoothing * 0.01,
    )

    mjx_data = mjx.make_data(mjx_model, impl=mjx_model.impl)

    x_goal = ca.DM(task.x_goal)
    Q = ca.DM(task.Q)
    Qf = ca.DM(task.Qf)
    u_ref = ca.DM(task.u_ref)
    R = ca.DM(task.R)

    efc_contact_addresses, socs = support.extract_contact_indices(mjx_model, mjx_data)

    ncf = int(efc_contact_addresses.size)
    self.ncf = ncf
    nsoc = len(socs)
    self.nsoc = nsoc

    def step1(x, u):
      nq, nv = mjx_model.nq, mjx_model.nv
      d = mjx_data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u,)
      d = mjx.forward(mjx_model, d)
      return d._impl.efc_force[efc_contact_addresses]

    def step2(x, u, cf):
      efc_force = jp.zeros(mjx_data._impl.nefc)
      efc_force = efc_force.at[efc_contact_addresses].set(cf)
      return support.step_given_efc_force(mjx_model, x, u, efc_force)

    self._step1_cb = casadi_support.JaxCallback('mjx_step1', step1, [nx, nu], ncf)
    self._step2_cb = casadi_support.JaxCallback('mjx_step2', step2, [nx, nu, ncf], nx)

    nvar = N * (nx + nu + ncf) + nx
    vars = ca.MX.sym('var', nvar, 1)

    xs = []
    us = []
    cfs = []

    offset = 0
    for k in range(N):
      xs.append(vars[offset:offset+nx])
      offset += nx
      us.append(vars[offset:offset+nu])
      offset += nu
      cfs.append(vars[offset:offset+ncf])
      offset += ncf
    xs.append(vars[offset:offset+nx])
    offset += nx
    assert offset == nvar

    def soc_constraint(cf):
      terms = []
      for soc in socs:
        cf_n = cf[soc.start_index]
        mu = ca.DM(np.asarray(soc.friction))
        cf_t = cf[soc.start_index + 1: soc.start_index + soc.dim] / mu
        terms.append(cf_n * cf_n - ca.dot(cf_t, cf_t))

      if terms:
        return ca.vertcat(*terms)
      return ca.DM.zeros(0, 1)

    obj = ca.MX(0)
    g_list = []

    for k in range(N):
      x = xs[k]
      u = us[k]
      cf = cfs[k]
      xnext = xs[k + 1]

      dx = x - x_goal
      du = u - u_ref

      obj += ca.mtimes([dx.T, Q, dx])
      obj += ca.mtimes([du.T, R, du])

      xnext_resid = xnext - self._step2_cb(x, u, cf)
      cf_resid = cf - self._step1_cb(x, u)

      g_list.append(xnext_resid)
      g_list.append(cf_resid)

      if nsoc > 0:
        g_list.append(soc_constraint(cf))

    dxf = xs[N] - x_goal
    obj += ca.mtimes([dxf.T, Qf, dxf])

    g = ca.vertcat(*g_list)

    self._nlp = {
        'x': vars,
        'f': obj,
        'g': g,
    }

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
      cf_lb[soc.start_index + 1: soc.start_index + soc.dim] = -np.inf
    cfs_lb = np.repeat(cf_lb[None, :], N, axis=0)
    cfs_ub = np.repeat(cf_ub[None, :], N, axis=0)

    self._vars_lb = self._pack(xs_lb, us_lb, cfs_lb)
    self._vars_ub = self._pack(xs_ub, us_ub, cfs_ub)

    constraint_lb = np.concatenate([np.zeros(nx), np.zeros(ncf), np.zeros(nsoc)])
    constraint_ub = np.concatenate([np.zeros(nx), np.zeros(ncf), np.full(nsoc, np.inf)])

    self._g_lb = np.tile(constraint_lb, N)
    self._g_ub = np.tile(constraint_ub, N)

    self._opts = {}

  def solve(
      self,
      *,
      xs_init,
      us_init,
      solver_name: str,
      options: dict = {},
  ):
    cfs_init = np.zeros((self.task.horizon, self.ncf))
    vars_init = self._pack(xs_init, us_init, cfs_init)

    solver = ca.nlpsol('solver', solver_name, self._nlp, options)

    tic = time.perf_counter()
    sol = solver(
        x0=vars_init,
        lbx=self._vars_lb,
        ubx=self._vars_ub,
        lbg=self._g_lb,
        ubg=self._g_ub,
    )
    total_time = time.perf_counter() - tic

    vars_opt = np.asarray(sol["x"]).flatten()
    xs_opt, us_opt, cfs_opt = self._unpack(vars_opt)

    return xs_opt, us_opt

  def _pack(
      self,
      xs: np.ndarray,
      us: np.ndarray,
      cfs: np.ndarray,
  ):
    N, nx, nu, ncf = self.task.horizon, self.task.nx, self.task.nu, self.ncf
    assert xs.shape == (N + 1, nx)
    assert us.shape == (N, nu)
    assert cfs.shape == (N, ncf)

    return np.concatenate([
        np.hstack((xs[:-1], us, cfs)).ravel(),
        xs[-1],
    ])

  def _unpack(self, vars: np.ndarray):
    N, nx, nu, ncf = self.task.horizon, self.task.nx, self.task.nu, self.ncf
    n_vars = (N + 1) * nx + N * nu + N * ncf
    assert vars.shape == (n_vars,)

    xucs = vars[:-nx].reshape(N, nx + nu + ncf)
    xs = np.vstack((xucs[:, :nx], vars[-nx:]))
    us = xucs[:, nx : nx + nu]
    cfs = xucs[:, nx + nu :]

    return xs, us, cfs


class CasadiIpoptLiftedMultipleShooting(_CasadiLiftedMultipleShooting):
  def __init__(
      self,
      task: Task,
      contact_approx: mjx.ContactApproxType,
      hess_approx,
  ):
    super().__init__(
        task=task,
        contact_approx=contact_approx,
    )

  def solve(
      self,
      *,
      xs_init,
      us_init,
      iterations: int,
      **kwargs,
  ):
    return super().solve(
        xs_init=xs_init,
        us_init=us_init,
        solver_name='ipopt',
        options={
            'ipopt.max_iter': iterations
        },
    )


class CasadiFatropLiftedMultipleShooting(_CasadiLiftedMultipleShooting):
  def __init__(
      self,
      task: Task,
      contact_approx: mjx.ContactApproxType,
      hess_approx,
  ):
    super().__init__(
        task=task,
        contact_approx=contact_approx,
    )

  def solve(
      self,
      *,
      xs_init,
      us_init,
      iterations: int,
      **kwargs,
  ):
    N, nx, ncf, nsoc = self.task.horizon, self.task.nx, self.ncf, self.nsoc
    equality = ([1] * (nx + ncf) + [0] * nsoc) * N

    return super().solve(
        xs_init=xs_init,
        us_init=us_init,
        solver_name='fatrop',
        options={
            'fatrop.max_iter': iterations,
            'structure_detection': 'auto',
            'equality': equality,
        },
    )
