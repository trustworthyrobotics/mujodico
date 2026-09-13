import time
import jax
import jax.numpy as jp
from mujodico import mjx
from mujodico.examples.methods import support
from mujodico.examples.tasks import Task
from mujodico.trajopt import ProxDDP
import numpy as np


class ProxddpLiftedMultipleShooting:

  def __init__(
      self,
      task: Task,
      contact_approx: mjx.ContactApproxType,
      hess_approx,
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
    self.nsoc = nsoc

    nu_aug = nu + ncf  # augmented control = [u, cf]
    self.nu = nu
    self.nu_aug = nu_aug

    def forward(x, u):
      nq, nv = mjx_model.nq, mjx_model.nv
      d = mjx_data.replace(
          qpos=x[:nq], qvel=x[nq : nq + nv], act=x[nq + nv :], ctrl=u
      )
      d = mjx.forward(mjx_model, d)
      return d._impl.efc_force[efc_contact_addresses]

    def advance(x, u, cf):
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

    x_lb, x_ub = task.x_bounds
    u_lb, u_ub = task.u_bounds

    cf_lb = np.zeros(ncf)
    cf_ub = np.full(ncf, np.inf)
    for soc in socs:
      cf_lb[soc.start_index + 1 : soc.start_index + soc.dim] = -np.inf

    u_aug_lb = np.concatenate([u_lb, cf_lb])
    u_aug_ub = np.concatenate([u_ub, cf_ub])

    def _make_box_constraint(lo, hi):
      lo_idx = np.nonzero(np.isfinite(lo))[0]
      hi_idx = np.nonzero(np.isfinite(hi))[0]

      lo_idx = jp.array(lo_idx)
      hi_idx = jp.array(hi_idx)
      lo_vals = jp.array(lo[lo_idx])
      hi_vals = jp.array(hi[hi_idx])

      def box_constraint(v):
        lo_res = lo_vals - v[lo_idx]   # lo - v <= 0
        hi_res = v[hi_idx] - hi_vals   # v - hi <= 0
        return jp.concatenate([lo_res, hi_res])

      return box_constraint

    x_box_constraint = _make_box_constraint(x_lb, x_ub)
    u_aug_box_constraint = _make_box_constraint(u_aug_lb, u_aug_ub)

    def stage_cost(x, u_aug, params):
      u = u_aug[:nu]
      return (x - x_goal).dot(Q @ (x - x_goal)) + (u - u_ref).dot(R @ (u - u_ref))

    def terminal_cost(x, params):
      return (x - x_goal).dot(Qf @ (x - x_goal))

    def implicit_dynamics(x, u_aug, xnext, params):
      u = u_aug[:nu]
      cf = u_aug[nu:]
      return jp.concatenate([
          xnext - advance(x, u, cf),
          cf - forward(x, u),
      ])

    def initial_state_constraint(x, params):
      return x - jp.array(self.task.x_0)

    def stage_constraint(x, u_aug, params):
      cf = u_aug[nu:]
      return jp.concatenate([
          -soc_constraint(cf),
          x_box_constraint(x),
          u_aug_box_constraint(u_aug),
      ])

    def terminal_constraint(x, params):
      return x_box_constraint(x)

    solver = ProxDDP(
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        implicit_dynamics=implicit_dynamics,
        initial_state_constraint=initial_state_constraint,
        stage_constraint=stage_constraint,
        terminal_constraint=terminal_constraint,
        horizon=N,
        num_states=nx,
        num_controls=nu_aug,
        params_example=None,
    )
    self.proxddp_solve = jax.jit(solver.solve, static_argnames=('options',))

  def solve(
      self,
      *,
      xs_init: np.ndarray,
      us_init: np.ndarray,
      iterations: int,
      **kwargs,
  ):
    N = self.task.horizon
    cfs_init = np.zeros((N, self.ncf))
    u_aug_init = jp.array(np.concatenate([us_init, cfs_init], axis=1))
    x_init = jp.array(xs_init)

    options = ProxDDP.Options(max_iters=iterations, **kwargs)

    tic = time.perf_counter()
    xs_opt, us_aug_opt = self.proxddp_solve(
        params=None,
        x_init=x_init,
        u_init=u_aug_init,
        options=options,
    )
    us_opt = us_aug_opt[:, :self.nu]
    total_time = time.perf_counter() - tic

    print(f'''
Method: {self.__class__.__name__}
  total time: {total_time:.6f} sec
''')

    return np.asarray(xs_opt), np.asarray(us_opt)
