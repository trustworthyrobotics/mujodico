import jax.numpy as jp
from mujodico import trajopt
from mujodico.examples.methods import util
from mujodico.examples.tasks import Task


class ContactImplicitTranscription:

  def __init__(self, task: Task):
    (
        dynamics,
        local_inequality,
        local_complementarity,
        w_bounds,
    ) = trajopt.contact_implicit_transcription(
        mjmodel=task.model,
        u_bounds=task.u_bounds,
    )

    def stage_cost(x, w, params):
      u = w[:task.nu]
      return (
          util.state_cost(params.Q, params.x_goal, x, task.quaternion_w_idxs)
          + util.control_cost(params.R, params.u_ref, u)
      )

    def terminal_cost(x, params):
      return util.state_cost(params.Qf, params.x_goal, x, task.quaternion_w_idxs)

    self.dynamics = dynamics
    self.local_inequality = local_inequality
    self.local_complementarity = local_complementarity
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
    self.params = util.TranscriptionParams(
        x_0=jp.asarray(task.x_0),
        x_goal=jp.asarray(task.x_goal),
        Q=jp.asarray(task.Q),
        Qf=jp.asarray(task.Qf),
        u_ref=jp.asarray(task.u_ref),
        R=jp.asarray(task.R),
    )
