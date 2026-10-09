from collections.abc import Callable
import jax
import jax.numpy as jp
import mujoco
from mujodico import mjx


def shooting_transcription(
    mjmodel: mujoco.MjModel,
    contact_approximation: mjx.ContactApproxType,
    contact_smoothing: float,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
  model = mjx.put_model(
      mjmodel,
      impl='jax',
      contact_approximation=contact_approximation,
      contact_smoothing=0.0,
  )
  smooth_model = mjx.put_model(
      mjmodel,
      impl=model.impl,
      contact_approximation=contact_approximation,
      contact_smoothing=contact_smoothing,
  )
  data = mjx.make_data(model, impl=model.impl)

  def dynamics(x, u):
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

  return dynamics
