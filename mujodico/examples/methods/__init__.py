from mujodico.examples.methods import ipopt_kkt_contact_implicit
from mujodico.examples.methods import ipopt_lifted_multiple_shooting
from mujodico.examples.methods import ipopt_multiple_shooting
from mujodico.examples.methods import ipopt_single_shooting


_METHODS = [
    ipopt_kkt_contact_implicit.IpoptKKTContactImplicit,
    ipopt_lifted_multiple_shooting.IpoptLiftedMultipleShooting,
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
