import mujodico.examples.methods.ipopt_multiple_shooting
import mujodico.examples.methods.ipopt_single_shooting



_METHODS = [
    ipopt_multiple_shooting.IpoptMultipleShooting,
    ipopt_single_shooting.IpoptSingleShooting,
]


def get_method_names():
  return [method.__name__ for method in _METHODS]


def parse_method_name(name: str | None):
  if name is None:
    raise ValueError('method not specified')

  name_clean = name.lower().strip().translate(str.maketrans('', '', ' -_'))

  for method in _METHODS:
    if name_clean == method.__name__.lower():
      return method

  raise ValueError(f'Unknown method {name}')
