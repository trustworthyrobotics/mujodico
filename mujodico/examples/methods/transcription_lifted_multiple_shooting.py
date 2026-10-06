import jax
import jax.numpy as jp
from mujodico import mjx
from mujodico.examples.methods import support
from mujodico.examples.tasks import Task
import numpy as np


class LiftedMultipleShootingTranscription:

  def __init__(self, task: Task):
    model = mjx.put_model(
        task.model,
        impl='jax',
        contact_approximation=task.contact_approx,
        contact_smoothing=task.contact_smoothing * 0.01,
    )
    data = mjx.make_data(model, impl=model.impl)

    addresses, socs = support.extract_contact_indices(model, data)

    nx = task.nx
    nu = task.nu
    ncf = int(addresses.size)

    def step1(x, u):
      nq, nv = model.nq, model.nv
      d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)
      d = mjx.forward(model, d)
      return d._impl.efc_force[addresses]

    def step2(x, u, cf):
      efc_force = jp.zeros(data._impl.nefc)
      efc_force = efc_force.at[addresses].set(cf)
      return support.step_given_efc_force(model, x, u, efc_force)

    def unpack_control(w):
      return w[:nu], w[nu:]

    def stage_cost(x, w, params):
      u, _ = unpack_control(w)
      return (
          support.state_cost(params.Q, params.x_goal, x, task.quaternion_w_idxs)
          + support.control_cost(params.R, params.u_ref, u)
      )

    def terminal_cost(x, params):
      return support.state_cost(params.Qf, params.x_goal, x, task.quaternion_w_idxs)

    def dynamics(x, w, params):
      u, cf = unpack_control(w)
      return step2(x, u, cf)

    def stage_equality(x, w, params):
      u, cf = unpack_control(w)
      return cf - step1(x, u)

    def stage_inequality(x, w, params):
      del x
      _, cf = unpack_control(w)
      constraints = jp.zeros(len(socs))
      for i, soc in enumerate(socs):
        normal = cf[soc.start_index]
        tangent = cf[soc.start_index+1:soc.start_index+soc.dim] / soc.friction
        constraints = constraints.at[i].set(tangent.dot(tangent) - normal * normal)
      return constraints

    u_lb, u_ub = task.u_bounds
    cf_lb = np.zeros(ncf)
    cf_ub = np.full(ncf, jp.inf)
    for soc in socs:
      cf_lb[soc.start_index+1:soc.start_index+soc.dim] = -np.inf

    x_bounds = task.x_bounds
    w_bounds = (np.concatenate([u_lb, cf_lb]), np.concatenate([u_ub, cf_ub]))

    self.stage_cost = stage_cost
    self.terminal_cost = terminal_cost
    self.dynamics = dynamics
    self.stage_equality_constraint = stage_equality
    self.stage_inequality_constraint = stage_inequality

    self.x_bounds = x_bounds
    self.w_bounds = w_bounds

    self.N = task.horizon
    self.nx = task.nx
    self.nw = nu + ncf

    self.n_stage_eq = nx + ncf
    self.n_stage_ineq = len(socs)

    self._task = task
    self.update(task)

  def update(self, task: Task):
    if type(task) is not type(self._task):
      raise ValueError('Updated task must be of the same class.')
    self._task = task

    self.params = support.TranscriptionParams(
        x_0=jp.asarray(task.x_0),
        x_goal=jp.asarray(task.x_goal),
        Q=jp.asarray(task.Q),
        Qf=jp.asarray(task.Qf),
        u_ref=jp.asarray(task.u_ref),
        R=jp.asarray(task.R),
    )
