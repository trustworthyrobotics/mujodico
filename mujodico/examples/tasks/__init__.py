import mujodico.examples.tasks.push_sphere_1d
import mujodico.examples.tasks.push_sphere_2d
from mujodico.examples.tasks.task import Task


_TASKS = [
    push_sphere_1d.PushSphere1d,
    push_sphere_2d.PushSphere2d,
]


def get_task_names():
  return [task.__name__ for task in _TASKS]


def parse_task_name(name: str | None):
  if name is None:
    raise ValueError('task not specified')

  name_clean = name.lower().strip().translate(str.maketrans('', '', ' -_'))

  for task in _TASKS:
    if name_clean == task.__name__.lower():
      return task

  raise ValueError(f'Unknown task {name}')
