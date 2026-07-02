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
    f'<{"|".join(get_method_names())}>\n(required)',
    required=True,
)
_TASK = flags.DEFINE_string(
    'task',
    None,
    f'<{"|".join(get_task_names())}>\n(required)',
    required=True,
)
_PLAN_CONTACT_APPROX = flags.DEFINE_string(
    'plan_contact_approx',
    None,
    f'<{"|".join([t.value for t in mjx.ContactApproxType])}>:'
    ' Contact approximation for planning.\n(required)',
    required=True,
)
_EVAL_CONTACT_APPROX = flags.DEFINE_string(
    'eval_contact_approx',
    None,
    f'<{"|".join([t.value for t in mjx.ContactApproxType])}>:'
    ' Contact approximation for evaluation.\n'
    '(default: same as plan_contact_approx)',
)


def _main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  method_cls = parse_method_name(_METHOD.value)
  task_cls = parse_task_name(_TASK.value)

  plan_contact_approx = _PLAN_CONTACT_APPROX.value
  plan_contact_approx = mjx.ContactApproxType._resolve(plan_contact_approx)

  eval_contact_approx = _EVAL_CONTACT_APPROX.value or plan_contact_approx
  eval_contact_approx = mjx.ContactApproxType._resolve(eval_contact_approx)

  task = task_cls()
  method = method_cls(task, plan_contact_approx)

  xs_nominal, us_nominal = method.solve(
    xs_init=task.xs_init,
    us_init=task.us_init,
    iterations=100,
  )

  task.visualize_trajectory(xs_nominal)


def main():
  app.run(_main)


if __name__ == '__main__':
  main()
