from mujodico.examples.methods.ipopt_kkt_contact_implicit import IpoptKKTContactImplicit
from mujodico.examples.methods.ipopt_lifted_multiple_shooting import IpoptLiftedMultipleShooting
from mujodico.examples.methods.ipopt_multiple_shooting import IpoptMultipleShooting
from mujodico.examples.methods.ipopt_single_shooting import IpoptSingleShooting
from mujodico.examples.methods.jax_pas_fatrop_lifted_multiple_shooting import JaxPasFatropLiftedMultipleShooting
from mujodico.examples.methods.transcription_lifted_multiple_shooting import LiftedMultipleShootingTranscription


_METHODS = [
    IpoptKKTContactImplicit,
    IpoptLiftedMultipleShooting,
    IpoptMultipleShooting,
    IpoptSingleShooting,
    JaxPasFatropLiftedMultipleShooting,
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
