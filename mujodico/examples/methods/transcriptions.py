from dataclasses import dataclass
import warnings
import jax
import jax.numpy as jp
from mujodico import mjx
from mujodico import trajopt
from mujodico.mjx._src import support
from mujodico.examples.tasks import Task
from mujodico.examples.tasks.task import control_cost, state_cost
import numpy as np


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TranscriptionParams:
  x_0: jax.Array
  x_goal: jax.Array
  Q: jax.Array
  Qf: jax.Array
  u_ref: jax.Array
  R: jax.Array


def _contact_frame_options(task: Task) -> dict:
  if task.planar_normal is not None:
    return {'planar_normal': task.planar_normal}
  return {'reference_state': task.x_0}


class SingleShootingTranscription:

  def __init__(
      self,
      task: Task,
  ):
    step = trajopt.shooting_transcription(
        model=mjx.put_model(task.model, impl='jax'),
        **_contact_frame_options(task),
    )

    def rollout(x_0, us):
      def scan_step(x, u):
        x_next = step(x, u)
        return x_next, x_next

      _, xs = jax.lax.scan(scan_step, x_0, us)
      return jp.concatenate([x_0[None], xs], axis=0)

    def objective(us, params):
      xs = rollout(params.x_0, us)

      stage_cost = jax.vmap(
          lambda x, u: (
              state_cost(params.Q, params.x_goal, x, task.quaternion_addrs)
              + control_cost(params.R, params.u_ref, u)
          )
      )(xs[:-1], us)

      terminal_cost = state_cost(
          params.Qf, params.x_goal, xs[-1], task.quaternion_addrs
      )

      return jp.sum(stage_cost) + terminal_cost

    self.rollout = rollout
    self.objective = objective
    self.u_bounds = task.u_bounds
    self.N = task.horizon
    self.nu = task.nu

    self._task = task
    self.update(task)

  def update(self, task: Task):
    if type(task) is not type(self._task):
      raise ValueError('Updated task must be of the same class.')
    self._task = task
    self.params = TranscriptionParams(
        x_0=jp.asarray(task.x_0),
        x_goal=jp.asarray(task.x_goal),
        Q=jp.asarray(task.Q),
        Qf=jp.asarray(task.Qf),
        u_ref=jp.asarray(task.u_ref),
        R=jp.asarray(task.R),
    )


class MultipleShootingTranscription:

  def __init__(
      self,
      task: Task,
      *,
      have_contact_force_inequality: bool = False,
      mjx_float32: bool = False,
  ):

    self.stage_inequality = None
    if not have_contact_force_inequality:
      dynamics = trajopt.shooting_transcription(
          model=mjx.put_model(task.model, impl='jax'),
          mjx_float32=mjx_float32,
          **_contact_frame_options(task),
      )

    else:
      model = mjx.put_model(task.model, impl='jax')
      model = support._with_contact_config(
          model, smoothing=support.contact_smoothing(task.model) * 0.001
      )
      lifted_dynamics, force_equality, _, bounds = (
          trajopt.lifted_shooting_transcription(
              model=model,
              u_bounds=task.u_bounds,
              mjx_float32=mjx_float32,
              **_contact_frame_options(task),
          )
      )
      ncf = len(bounds[0]) - task.nu
      positive = jp.asarray(np.flatnonzero(np.isfinite(bounds[0][task.nu :])))

      def predicted_force(x, u, x_next):
        return -force_equality(x, jp.concatenate([u, jp.zeros(ncf)]), x_next)

      def dynamics(x, u, x_next=None):
        force = predicted_force(x, u, x_next)
        return lifted_dynamics(x, jp.concatenate([u, force]))

      def stage_inequality(x, u, x_next=None):
        return predicted_force(x, u, x_next)[positive]

      self.stage_inequality = stage_inequality

    def stage_cost(x, u, params):
      return state_cost(
          params.Q, params.x_goal, x, task.quaternion_addrs
      ) + control_cost(params.R, params.u_ref, u)

    def terminal_cost(x, params):
      return state_cost(params.Qf, params.x_goal, x, task.quaternion_addrs)

    self.stage_cost = stage_cost
    self.terminal_cost = terminal_cost
    self.dynamics = dynamics
    self.x_bounds = task.x_bounds
    self.u_bounds = task.u_bounds
    self.N = task.horizon
    self.nx = task.nx
    self.nu = task.nu

    self._task = task
    self.update(task)

  def update(self, task: Task):
    if type(task) is not type(self._task):
      raise ValueError('Updated task must be of the same class.')
    self._task = task
    self.params = TranscriptionParams(
        x_0=jp.asarray(task.x_0),
        x_goal=jp.asarray(task.x_goal),
        Q=jp.asarray(task.Q),
        Qf=jp.asarray(task.Qf),
        u_ref=jp.asarray(task.u_ref),
        R=jp.asarray(task.R),
    )


class LiftedMultipleShootingTranscription:

  def __init__(self, task: Task, *, mjx_float32: bool = False):
    model = mjx.put_model(task.model, impl='jax')
    model = support._with_contact_config(
        model, smoothing=support.contact_smoothing(task.model) * 0.001
    )
    dynamics, stage_equality, stage_inequality, w_bounds = (
        trajopt.lifted_shooting_transcription(
            model=model,
            u_bounds=task.u_bounds,
            mjx_float32=mjx_float32,
            **_contact_frame_options(task),
        )
    )

    def stage_cost(x, w, params):
      u = w[: task.nu]
      return state_cost(
          params.Q, params.x_goal, x, task.quaternion_addrs
      ) + control_cost(params.R, params.u_ref, u)

    def terminal_cost(x, params):
      return state_cost(params.Qf, params.x_goal, x, task.quaternion_addrs)

    x_bounds = task.x_bounds

    self.stage_cost = stage_cost
    self.terminal_cost = terminal_cost
    self.dynamics = dynamics
    self.stage_equality = stage_equality
    self.stage_inequality = stage_inequality
    self.x_bounds = x_bounds
    self.w_bounds = w_bounds
    self.N = task.horizon
    self.nx = task.nx
    self.nw = w_bounds[0].size

    self._task = task
    self.update(task)

  def update(self, task: Task):
    if type(task) is not type(self._task):
      raise ValueError('Updated task must be of the same class.')
    self._task = task
    self.params = TranscriptionParams(
        x_0=jp.asarray(task.x_0),
        x_goal=jp.asarray(task.x_goal),
        Q=jp.asarray(task.Q),
        Qf=jp.asarray(task.Qf),
        u_ref=jp.asarray(task.u_ref),
        R=jp.asarray(task.R),
    )


class ContactImplicitTranscription:

  def __init__(self, task: Task, *, independent_next_state: bool = True):
    model = mjx.put_model(task.model, impl='jax')
    model = support._with_contact_config(model, smoothing=0.0)
    contact_approx = support.contact_approx(model)
    if contact_approx not in (
        support.ContactApproxType.CONIC,
        support.ContactApproxType.LINEAR_ELASTIC,
      ):
      warnings.warn(
          f'ContactImplicitTranscription does not support {contact_approx.value} '
          f'contact approximation; switching to conic.',
          UserWarning,
          stacklevel=2,
      )
      model = support._with_contact_config(model, approximation='conic')

    (
        dynamics,
        local_inequality,
        local_complementarity,
        w_bounds,
    ) = trajopt.contact_implicit_transcription(
        model=model,
        u_bounds=task.u_bounds,
        **_contact_frame_options(task),
    )

    def stage_cost(x, w, params):
      u = w[: task.nu]
      return state_cost(
          params.Q, params.x_goal, x, task.quaternion_addrs
      ) + control_cost(params.R, params.u_ref, u)

    def terminal_cost(x, params):
      return state_cost(params.Qf, params.x_goal, x, task.quaternion_addrs)

    def stage_inequality(x, w):
      xnext = dynamics(x, w)
      return local_inequality(x, w, xnext)

    self.dynamics = dynamics
    if independent_next_state:
      self.local_inequality = local_inequality
      self.local_complementarity = local_complementarity
    else:
      self.stage_inequality = stage_inequality
      self.stage_complementarity = local_complementarity
    self.stage_cost = stage_cost
    self.terminal_cost = terminal_cost
    self.x_bounds = task.x_bounds
    self.w_bounds = w_bounds
    self.N = task.horizon
    self.nx = task.nx
    self.nw = w_bounds[0].size

    self._task = task
    self.update(task)

  def update(self, task: Task):
    if type(task) is not type(self._task):
      raise ValueError('Updated task must be of the same class.')
    self._task = task
    self.params = TranscriptionParams(
        x_0=jp.asarray(task.x_0),
        x_goal=jp.asarray(task.x_goal),
        Q=jp.asarray(task.Q),
        Qf=jp.asarray(task.Qf),
        u_ref=jp.asarray(task.u_ref),
        R=jp.asarray(task.R),
    )
