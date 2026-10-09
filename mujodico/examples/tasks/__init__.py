from mujodico.examples.tasks.allegro_ball import AllegroBall
from mujodico.examples.tasks.allegro_cube import AllegroCube
from mujodico.examples.tasks.allegro_pen import AllegroPen
from mujodico.examples.tasks.finger_circular_knob import FingerCircularKnob
from mujodico.examples.tasks.fingers_sphere_2d import FingersSphere2d
from mujodico.examples.tasks.push_box_2d import PushBox2d
from mujodico.examples.tasks.push_circular_knob import PushCircularKnob
from mujodico.examples.tasks.push_multisphere_2d import PushMultiSphere2d
from mujodico.examples.tasks.push_sphere_1d import PushSphere1d
from mujodico.examples.tasks.push_sphere_2d import PushSphere2d
from mujodico.examples.tasks.push_square_knob import PushSquareKnob
from mujodico.examples.tasks.task import Task


__all__ = [
    'AllegroBall',
    'AllegroCube',
    'AllegroPen',
    'FingerCircularKnob',
    'FingersSphere2d',
    'PushBox2d',
    'PushCircularKnob',
    'PushMultiSphere2d',
    'PushSphere1d',
    'PushSphere2d',
    'PushSquareKnob',
    'Task',
]


_TASKS = [
    PushSphere1d,
    PushSphere2d,
    PushBox2d,
    PushMultiSphere2d,
    PushCircularKnob,
    PushSquareKnob,
    FingerCircularKnob,
    FingersSphere2d,
    AllegroBall,
    AllegroCube,
    AllegroPen,
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
