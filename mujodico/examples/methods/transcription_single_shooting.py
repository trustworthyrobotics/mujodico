import jax
import jax.numpy as jp
from mujodico import trajopt
from mujodico.examples.methods import util
from mujodico.examples.tasks import Task


class SingleShootingTranscription:

  def __init__(
      self,
      task: Task,
  ):
    step = trajopt.shooting_transcription(
        mjmodel=task.model,
        contact_approximation=task.contact_approx,
        contact_smoothing=task.contact_smoothing,
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
            util.state_cost(params.Q, params.x_goal, x, task.quaternion_w_idxs)
            + util.control_cost(params.R, params.u_ref, u)
          )
      )(xs[:-1], us)

      terminal_cost = util.state_cost(params.Qf, params.x_goal, xs[-1], task.quaternion_w_idxs)

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
    self.params = util.TranscriptionParams(
        x_0=jp.asarray(task.x_0),
        x_goal=jp.asarray(task.x_goal),
        Q=jp.asarray(task.Q),
        Qf=jp.asarray(task.Qf),
        u_ref=jp.asarray(task.u_ref),
        R=jp.asarray(task.R),
    )
