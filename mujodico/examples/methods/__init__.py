import mujodico.examples.methods.ipopt_multiple_shooting
import mujodico.examples.methods.ipopt_single_shooting


_METHODS = [
    ipopt_multiple_shooting.IpoptMultipleShooting,
    ipopt_single_shooting.IpoptSingleShooting,
]


def get_method_names():
  return [method.__name__ for method in _METHODS]


def parse_method_name(name: str):
  name = name.lower().strip().translate(str.maketrans('', '', ' -_'))

  for method in _METHODS:
    if name == method.__name__.lower():
      return method

  raise ValueError(
      f'unknown method: value should be one of <{"|".join(get_method_names())}>'
  )
