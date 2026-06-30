import mujodico.examples.methods.ipopt_multiple_shooting
import mujodico.examples.methods.ipopt_single_shooting



_METHODS = {
    'IpoptMultipleShooting': ipopt_multiple_shooting.solve,
    'IpoptSingleShooting': ipopt_single_shooting.solve,
}


def get_method_names():
  return list(_METHODS.keys())


def parse_method_name(name: str | None):
  if name is None:
    raise ValueError('method not specified')

  name_clean = (
      name.lower().strip().replace(' ', '').replace('-', '').replace('_', '')
  )

  for k, v in _METHODS.items():
    if name_clean == k.lower():
      return v

  raise ValueError(f'Unknown method {name}')
