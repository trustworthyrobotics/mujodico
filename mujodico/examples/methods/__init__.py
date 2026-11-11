from mujodico.examples.methods.ccopt_contact_implicit import CcoptCrossoverContactImplicit
from mujodico.examples.methods.ccopt_contact_implicit import CcoptPenaltyContactImplicit
from mujodico.examples.methods.ccopt_contact_implicit import CcoptRelaxationContactImplicit
from mujodico.examples.methods.ciopt_contact_implicit import CioptImpactContactImplicit
from mujodico.examples.methods.fatrop_lifted_multiple_shooting import FatropLiftedMultipleShooting
from mujodico.examples.methods.fatrop_multiple_shooting import FatropMultipleShooting
from mujodico.examples.methods.impact_contact_implicit import ImpactContactImplicit
from mujodico.examples.methods.ipopt_lifted_multiple_shooting import IpoptLiftedMultipleShooting
from mujodico.examples.methods.ipopt_multiple_shooting import IpoptMultipleShooting
from mujodico.examples.methods.ipopt_relaxation_contact_implicit import IpoptRelaxationContactImplicit
from mujodico.examples.methods.ipopt_single_shooting import IpoptSingleShooting
from mujodico.examples.methods.ippas_lifted_multiple_shooting import IppasLiftedMultipleShooting
from mujodico.examples.methods.knitro_lifted_multiple_shooting import KnitroLiftedMultipleShooting
from mujodico.examples.methods.transcription_contact_implicit import ContactImplicitTranscription
from mujodico.examples.methods.transcription_lifted_multiple_shooting import LiftedMultipleShootingTranscription
from mujodico.examples.methods.transcription_multiple_shooting import MultipleShootingTranscription
from mujodico.examples.methods.transcription_single_shooting import SingleShootingTranscription


__all__ = [
    'CcoptCrossoverContactImplicit',
    'CcoptPenaltyContactImplicit',
    'CcoptRelaxationContactImplicit',
    'CioptImpactContactImplicit',
    'FatropLiftedMultipleShooting',
    'FatropMultipleShooting',
    'ImpactContactImplicit',
    'IpoptLiftedMultipleShooting',
    'IpoptMultipleShooting',
    'IpoptRelaxationContactImplicit',
    'IpoptSingleShooting',
    'IppasLiftedMultipleShooting',
    'KnitroLiftedMultipleShooting',
    'ContactImplicitTranscription',
    'LiftedMultipleShootingTranscription',
    'MultipleShootingTranscription',
    'SingleShootingTranscription',
]


_METHODS = [
    CcoptCrossoverContactImplicit,
    CcoptPenaltyContactImplicit,
    CcoptRelaxationContactImplicit,
    CioptImpactContactImplicit,
    FatropLiftedMultipleShooting,
    FatropMultipleShooting,
    ImpactContactImplicit,
    IpoptRelaxationContactImplicit,
    IpoptLiftedMultipleShooting,
    IpoptMultipleShooting,
    IpoptSingleShooting,
    IppasLiftedMultipleShooting,
    KnitroLiftedMultipleShooting,
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
