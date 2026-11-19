import time
import csv
from pathlib import Path
from absl import app
from absl import flags
from mujodico.examples.methods import get_method_names, parse_method_name
from mujodico.examples.tasks import get_task_names, parse_task_name
from mujodico.examples.tasks.task import state_cost
from mujodico.examples.tasks.task_playback import TaskPlayback
from mujodico.examples.visualizer import Visualizer


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

  with open(result_file, 'a+', newline='') as f:
    f.seek(0)
    reader = csv.reader(f)
    header = next(reader, None)
    fields = ['seed', 'goal_error', 'solve_time_sec', 'solver_iters', 'status']
    writer = csv.writer(f)
    if header is None:
      writer.writerow(fields)
    elif header != fields:
      if header != fields[:-1]:
        raise ValueError(f'Unexpected CSV header in "{result_file}": {header}')
      # Preserve existing results; their solver statuses were not recorded.
      rows = list(reader)
      f.seek(0)
      f.truncate()
      writer.writerow(fields)
      writer.writerows([*row, ''] for row in rows)
    f.seek(0, 2)

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

      traj = method.solve(
        traj_init={
            'xs': task.xs_init,
            'us': task.us_init,
        },
        iterations=iters,
      )

      solve_time = traj['solve_time']
      status = traj['status']

      goal_error = state_cost(
          task.Q, task.x_goal, traj['xs'][-1], task.quaternion_addrs
      )

      writer.writerow([seed, goal_error, solve_time, iters, status])
      f.flush()

      traj['info'] = f'''
- Task: {task_cls.__name__}(seed={seed})
  - goal error: {goal_error:.8f}
- Method: {method_cls.__name__}
  - status: {status}
  - solve time: {solve_time:.4f} sec
  - iterations: {iters}
'''
      visualizer.add(TaskPlayback(task, traj))

  while True:
    time.sleep(3600)


if __name__ == '__main__':
  app.run(_main)
