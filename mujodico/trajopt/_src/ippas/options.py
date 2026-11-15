"""Static configuration for the compiled IPPAS solver."""

from dataclasses import dataclass
from dataclasses import fields
import math
from numbers import Integral
from numbers import Real


@dataclass(frozen=True)
class Options:

  max_iter: int = 1000
  tol: float = 1e-8
  tol_acceptable: float = 1e-6
  acceptable_iter: int = 15
  constr_viol_tol: float = 1e-4
  mu_init: float = 0.1
  barrier_tol_factor: float = 10.0
  mu_linear_decrease_factor: float = 0.2
  mu_superlinear_decrease_power: float = 1.5
  compl_inf_tol: float = 1e-4
  tau_min: float = 0.99
  bound_push: float = 1e-2
  bound_frac: float = 1e-2
  bound_relax_factor: float = 1e-8
  initialize_duals: bool = True
  lam_max: float = 1e3
  kappa_d: float = 1e-5
  kappa_sigma: float = 1e10
  s_max: float = 100.0
  max_ls: int = 40
  max_soc: int = 4
  kappa_soc: float = 0.99
  alpha_red_factor: float = 0.5
  alpha_min_frac: float = 0.05
  theta_max_fact: float = 1e4
  theta_min_fact: float = 1e-4
  s_phi: float = 2.3
  s_theta: float = 1.1
  delta: float = 1.0
  gamma_theta: float = 1e-5
  gamma_phi: float = 1e-8
  eta_phi: float = 1e-8
  obj_max_incr: float = 5.0
  filter_reset_trigger: int = 5
  max_filter_resets: int = -1
  delta_c_stripe: float = 1e-6
  kappa_c: float = 0.25
  delta_w0: float = 1e-4
  delta_wmin: float = 1e-20
  delta_w_max: float = 1e40
  kappa_wmin: float = 1.0 / 3.0
  kappa_wplus: float = 8.0
  kappa_wplusem: float = 100.0
  max_regularization: int = 24
  max_refinement: int = 3
  linear_tol: float = 1e-10
  rank_tol: float = 1e-12
  print_level: int = 0

  def __post_init__(self) -> None:
    positive_counts = ('max_ls', 'max_regularization', 'filter_reset_trigger')
    nonnegative_reals = ('kappa_d', 'bound_relax_factor')
    # Iterate over the dataclass schema so every field receives validation,
    # including new fields added later. Reject booleans as numeric settings.
    for field in fields(self):
      name, value = field.name, getattr(self, field.name)
      if field.type is bool:
        if not isinstance(value, bool):
          raise ValueError(f'{name} must be a bool.')
      elif field.type is int:
        minimum = (
            -1
            if name == 'max_filter_resets'
            else 1
            if name in positive_counts
            else 0
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < minimum
        ):
          raise ValueError(f'{name} must be an integer >= {minimum}.')
      elif field.type is float:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
          raise ValueError(f'{name} must be a finite real number.')
        if name in nonnegative_reals:
          if value < 0:
            raise ValueError(f'{name} must be nonnegative.')
        elif value <= 0:
          raise ValueError(f'{name} must be positive.')
      else:
        raise ValueError(f'No validation defined for option {name}.')
    for name in (
        'tau_min',
        'bound_frac',
        'alpha_red_factor',
        'alpha_min_frac',
        'gamma_theta',
        'gamma_phi',
        'eta_phi',
        'kappa_soc',
        'mu_linear_decrease_factor',
        'kappa_wmin',
    ):
      if not 0 < getattr(self, name) < 1:
        raise ValueError(f'{name} must lie strictly between zero and one.')
    if self.bound_frac >= 0.5:
      raise ValueError('bound_frac must be less than 0.5.')
    if self.mu_superlinear_decrease_power <= 1:
      raise ValueError('mu_superlinear_decrease_power must exceed one.')
    if self.delta_wmin > self.delta_w0 or self.delta_w0 > self.delta_w_max:
      raise ValueError('Require delta_wmin <= delta_w0 <= delta_w_max.')
    if min(self.kappa_wplus, self.kappa_wplusem) <= 1:
      raise ValueError('Regularization growth factors must exceed one.')
    if self.kappa_sigma < 1:
      raise ValueError('kappa_sigma must be >= 1.')
    if self.theta_min_fact > self.theta_max_fact:
      raise ValueError('Require theta_min_fact <= theta_max_fact.')
