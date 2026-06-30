import os
import time
from typing import Sequence

from absl import app
from absl import flags
import jax
from jax import numpy as jp
import mujoco
from mujodico import mjx
from mujodico.examples.methods import get_method_names, parse_method_name
from mujodico.examples.tasks import get_task_names, parse_task_name
import numpy as np


_METHOD = flags.DEFINE_string(
    'method',
    None,
    '{' + '|'.join(get_method_names()) + '}',
)
_TASK = flags.DEFINE_string(
    'task',
    None,
    '{' + '|'.join(get_task_names()) + '}',
)


def _main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  method_fun = parse_method_name(_METHOD.value)
  task_cls = parse_task_name(_TASK.value)

  task = task_cls()

  xs_nominal, us_nominal = method_fun(task)

  model = task.model
  mjx_model = mjx.put_model(model)
  mjx_data = mjx.make_data(mjx_model)

  @jax.jit
  def step(x, u):
    d = mjx_data.replace(qpos=x[: model.nq], qvel=x[model.nq :], ctrl=u)
    d = mjx.step(mjx_model, d)
    return jp.concatenate((d.qpos, d.qvel))

  @jax.jit
  def rollout(x0, us):
    def scan_step(x, u):
      x_next = step(x, u)
      return x_next, x_next

    _, xs = jax.lax.scan(scan_step, x0, us)
    return jp.concatenate([x0[None], xs], axis=0)

  xs_rollout = rollout(task.x_0, us_nominal)
  xs_rollout = np.asarray(xs_rollout)

  task.visualize_trajectory(xs_rollout)


def main():
  app.run(_main)


if __name__ == '__main__':
  main()
