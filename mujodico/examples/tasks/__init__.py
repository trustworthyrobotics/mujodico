from mujodico.examples.tasks import push_box_2d
from mujodico.examples.tasks import push_multisphere_2d
from mujodico.examples.tasks import push_sphere_1d
from mujodico.examples.tasks import push_sphere_2d
from mujodico.examples.tasks import push_circular_knob
from mujodico.examples.tasks import push_square_knob
from mujodico.examples.tasks import fingers_sphere_2d
from mujodico.examples.tasks.task import Task as Task


_TASKS = [
    push_sphere_1d.PushSphere1d,
    push_sphere_2d.PushSphere2d,
    push_box_2d.PushBox2d,
    push_multisphere_2d.PushMultiSphere2d,
    push_circular_knob.PushCircularKnob,
    push_square_knob.PushSquareKnob,
    fingers_sphere_2d.FingersSphere2d,
]


def get_task_names():
  return [task.__name__ for task in _TASKS]


def parse_task_name(name: str):
  name = name.lower().strip().translate(str.maketrans('', '', ' -_'))

  for task in _TASKS:
    if name == task.__name__.lower():
      return task

  raise ValueError(
      f'unknown task: value should be one of <{"|".join(get_task_names())}>'
  )
