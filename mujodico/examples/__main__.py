import os
import time

from absl import app
from absl import flags
import jax
from jax import numpy as jp
import mujoco
from mujodico import mjx
from mujodico.examples.methods import get_method_names, parse_method_name, support
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
_CONTACT_APPROX = flags.DEFINE_string(
    'contact_approx',
    None,
    f'<{"|".join([t.value for t in mjx.ContactApproxType])}>:'
    ' Contact approximation for planning.\n(required)',
    required=True,
)
_HESS_APPROX = flags.DEFINE_string(
    'hess_approx',
    'gaussnewton',
    f'<{"|".join([t.value for t in support.HessianApproxType])}>:'
    ' Trajectory optimization Hessian approximation.'
)
_ITERATIONS = flags.DEFINE_integer(
    'iterations', 500, 'Number of solver iterations', lower_bound=0,
)
_TESTS = flags.DEFINE_integer(
    'tests', 1, 'Number of test cases to run', lower_bound=1,
)


def _main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  method_cls = parse_method_name(_METHOD.value)
  task_cls = parse_task_name(_TASK.value)

  contact_approx = mjx.ContactApproxType._resolve(_CONTACT_APPROX.value)

  hess_approx = support.HessianApproxType._resolve(_HESS_APPROX.value)

  for seed in range(_TESTS.value):
    task = task_cls(seed)
    method = method_cls(task, contact_approx, hess_approx)

    tic = time.perf_counter()
    xs_nominal, us_nominal = method.solve(
      xs_init=task.xs_init,
      us_init=task.us_init,
      iterations=_ITERATIONS.value,
    )
    total_time = time.perf_counter() - tic

    x_error = xs_nominal[-1] - task.x_goal
    goal_error = x_error.dot(task.Q @ x_error)

    print(f'''
Task: {task_cls.__name__}
Method: {method_cls.__name__}
  seed: {seed}
  total time: {total_time:.6f} sec
  goal error: {goal_error:.6f}
''')

  task.visualize_trajectory(xs_nominal)


def main():
  app.run(_main)


if __name__ == '__main__':
  main()
