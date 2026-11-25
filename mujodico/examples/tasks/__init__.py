from mujodico.examples.tasks.allegro_ball import AllegroBall
from mujodico.examples.tasks.allegro_pen import AllegroPen
from mujodico.examples.tasks.double_cart_pole import DoubleCartPole
from mujodico.examples.tasks.finger_circular_knob import FingerCircularKnob
from mujodico.examples.tasks.fingers_sphere_2d import FingersSphere2d
from mujodico.examples.tasks.push_box_2d import PushBox2d
from mujodico.examples.tasks.push_knob_circular import PushKnobCircular
from mujodico.examples.tasks.push_knob_cube import PushKnobCube
from mujodico.examples.tasks.push_knob_sphere import PushKnobSphere
from mujodico.examples.tasks.push_knob_square import PushKnobSquare
from mujodico.examples.tasks.push_multisphere_2d import PushMultiSphere2d
from mujodico.examples.tasks.push_sphere_1d import PushSphere1d
from mujodico.examples.tasks.push_sphere_2d import PushSphere2d
from mujodico.examples.tasks.quadrotor import Quadrotor
from mujodico.examples.tasks.quadrotor_pole import QuadrotorPole
from mujodico.examples.tasks.rocket_landing import RocketLanding
from mujodico.examples.tasks.task import Task


__all__ = [
    'AllegroBall',
    'AllegroPen',
    'DoubleCartPole',
    'FingerCircularKnob',
    'FingersSphere2d',
    'PushBox2d',
    'PushKnobCircular',
    'PushKnobCube',
    'PushKnobSphere',
    'PushKnobSquare',
    'PushMultiSphere2d',
    'PushSphere1d',
    'PushSphere2d',
    'Quadrotor',
    'QuadrotorPole',
    'RocketLanding',
    'Task',
]


_TASKS = [
    PushSphere1d,
    PushSphere2d,
    PushBox2d,
    PushMultiSphere2d,
    PushKnobCircular,
    PushKnobSphere,
    PushKnobCube,
    PushKnobSquare,
    DoubleCartPole,
    Quadrotor,
    QuadrotorPole,
    RocketLanding,
    FingerCircularKnob,
    FingersSphere2d,
    AllegroBall,
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
