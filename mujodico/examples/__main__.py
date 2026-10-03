import os
import time
import csv
from pathlib import Path

from absl import app
from absl import flags
import jax
from jax import numpy as jp
import mujoco
from mujodico import mjx
from mujodico.examples.methods import get_method_names, parse_method_name, support
from mujodico.examples.tasks import get_task_names, parse_task_name
from mujodico.examples.visualizer import Visualizer
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
_ITERATIONS = flags.DEFINE_integer(
    'iterations', 500, 'Number of solver iterations', lower_bound=0,
)
_TESTS = flags.DEFINE_integer(
    'tests', 1, 'Number of test cases to run.', lower_bound=1,
)
_SEED = flags.DEFINE_integer(
    'seed', 0, 'Random seed for tests.',
)


def _main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  method_cls = parse_method_name(_METHOD.value)
  task_cls = parse_task_name(_TASK.value)

  result_dir = Path(__file__).resolve().parent / 'result'
  result_dir.mkdir(exist_ok=True)
  result_file = result_dir / f'{task_cls.__name__}_{method_cls.__name__}.csv'
  print(f'Writing results to "{result_file}"')

  with open(result_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
      writer.writerow(['seed', 'goal_error', 'solve_time_sec', 'solver_iters'])

    visualizer = Visualizer()
    visualizer.run()

    method = None
    for seed in range(_SEED.value, _SEED.value + _TESTS.value):
      task = task_cls(seed)
      if method is None:
        method = method_cls(task)
      elif hasattr(method, 'update'):
        method.update(task)
      else:
        method = method_cls(task)

      iters = _ITERATIONS.value

      tic = time.perf_counter()

      traj = method.solve(
        traj_init={
            'xs': task.xs_init,
            'us': task.us_init,
        },
        iterations=iters,
      )

      solve_time = time.perf_counter() - tic

      goal_error = support.state_cost(
          task.Q, task.x_goal, traj['xs'][-1], task.quaternion_w_idxs
      )

      writer.writerow([seed, goal_error, solve_time, iters])
      f.flush()

      traj['info'] = f'''
- Task: {task_cls.__name__}(seed={seed})
  - goal error: {goal_error:.8f}
- Method: {method_cls.__name__}
  - solve time: {solve_time:.4f} sec
  - iterations: {iters}
'''
      visualizer.update(task, traj)

  if _TESTS.value == 1:
    while True:
      time.sleep(3600)


if __name__ == '__main__':
  app.run(_main)
