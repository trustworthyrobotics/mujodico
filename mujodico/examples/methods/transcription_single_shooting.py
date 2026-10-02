import jax
import jax.numpy as jp
from mujodico import mjx
from mujodico.examples.methods import support
from mujodico.examples.tasks import Task


class SingleShootingTranscription:

  def __init__(
      self,
      task: Task,
  ):
    model = mjx.put_model(
        task.model,
        impl='jax',
        contact_approximation=task.contact_approx,
        contact_smoothing=0.0,
    )
    smooth_model = mjx.put_model(
        task.model,
        impl=model.impl,
        contact_approximation=task.contact_approx,
        contact_smoothing=task.contact_smoothing,
    )
    data = mjx.make_data(model, impl=model.impl)

    def step(x, u):
      nq, nv = model.nq, model.nv
      d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)

      next_data = mjx.step(model, d)
      xnext = jp.concatenate((next_data.qpos, next_data.qvel, next_data.act))

      smooth_data = mjx.step(smooth_model, d)
      xnext_smooth = jp.concatenate((smooth_data.qpos, smooth_data.qvel, smooth_data.act))

      return (
          jax.lax.stop_gradient(xnext)
          + xnext_smooth
          - jax.lax.stop_gradient(xnext_smooth)
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
            support.state_cost(params.Q, params.x_goal, x, task.quaternion_w_idxs)
            + support.control_cost(params.R, params.u_ref, u)
          )
      )(xs[:-1], us)

      terminal_cost = support.state_cost(params.Qf, params.x_goal, xs[-1], task.quaternion_w_idxs)

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
    self.params = support.TranscriptionParams(
        x_0=jp.asarray(task.x_0),
        x_goal=jp.asarray(task.x_goal),
        Q=jp.asarray(task.Q),
        Qf=jp.asarray(task.Qf),
        u_ref=jp.asarray(task.u_ref),
        R=jp.asarray(task.R),
    )