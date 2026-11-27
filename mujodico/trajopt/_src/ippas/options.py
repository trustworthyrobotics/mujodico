"""Static configuration for the compiled IPPAS solver."""

from dataclasses import dataclass
from dataclasses import fields
import math
from numbers import Integral
from numbers import Real


@dataclass(frozen=True)
class Options:
  """Static solver settings, supplied by keyword when constructing the solver.

  Defaults follow FATROP/IPOPT where their policies are implemented. Counts
  and tolerances below are validated at construction, before JAX tracing.
  """

  # Convergence: scaled KKT accuracy, original-bound feasibility, and limits.
  # Maximum outer iterations; zero returns the initialized result.
  max_iter: int = 1000
  # Required scaled original-NLP KKT error.
  tol: float = 1e-8
  # Looser scaled KKT error allowed for acceptable termination.
  tol_acceptable: float = 1e-6
  # Consecutive acceptable iterates required; zero disables this termination.
  acceptable_iter: int = 15
  # Original-bound feasibility tolerance and absolute bound-relaxation cap.
  constr_viol_tol: float = 1e-4
  # Complementarity tolerance used with tol to set the minimum barrier target.
  compl_inf_tol: float = 1e-4

  # Monotone barrier updates.
  # Initial barrier parameter μ.
  mu_init: float = 0.1
  # Permit a reduction when Eμ < barrier_tol_factor × μ.
  barrier_tol_factor: float = 10.0
  # Linear reduction factor κμ ∈ (0, 1).
  mu_linear_decrease_factor: float = 0.2
  # Superlinear exponent applied to μ, greater than one.
  mu_superlinear_decrease_power: float = 1.5
  # Allow several reductions per update; the first update always allows this.
  mu_allow_fast_monotone_decrease: bool = True
  # Minimum fraction-to-boundary parameter: τ = max(tau_min, 1 − μ).
  tau_min: float = 0.99
  # One-sided slack damping coefficient, multiplied by μ; zero disables it.
  kappa_d: float = 1e-5

  # Bounds and cold-start initialization.
  # Absolute/relative margin from a finite bound when pushing a slack inside.
  bound_push: float = 1e-2
  # Maximum push as a fraction of a two-sided interval; strictly below 1/2.
  bound_frac: float = 1e-2
  # Relative bound expansion, capped by constr_viol_tol; zero disables it.
  bound_relax_factor: float = 1e-8
  # Discard the initial equality-multiplier estimate if its ∞-norm exceeds this.
  lam_max: float = 1e3

  # Multiplier safeguards and residual scaling.
  # Clamp finite-bound multipliers to [μ/(κσ distance), κσ μ/distance]; ≥ 1.
  kappa_sigma: float = 1e10
  # Average multiplier magnitude above which KKT residuals are scaled.
  s_max: float = 100.0

  # Filter line search.
  # Multiply the primal step by this factor after an ordinary rejection.
  alpha_red_factor: float = 0.5
  # Safety factor on the minimum permitted backtracking step.
  alpha_min_frac: float = 0.05
  # θ_max = theta_max_fact × max(1, initial constraint violation).
  theta_max_fact: float = 1e4
  # θ_min = theta_min_fact × max(1, initial constraint violation).
  theta_min_fact: float = 1e-4
  # Barrier directional-derivative exponent in the f-type switching test.
  s_phi: float = 2.3
  # Constraint-violation exponent in the f-type switching test.
  s_theta: float = 1.1
  # Multiplier of θ**s_theta in the switching test.
  delta: float = 1.0
  # Required relative feasibility progress and horizontal filter margin.
  gamma_theta: float = 1e-5
  # Objective progress and vertical filter margin, scaled by current θ.
  gamma_phi: float = 1e-8
  # Armijo sufficient-decrease coefficient.
  eta_phi: float = 1e-8
  # Maximum barrier-objective rise in decimal orders of magnitude.
  obj_max_incr: float = 5.0
  # Consecutive searches containing filter-induced rejections before a reset.
  filter_reset_trigger: int = 5
  # Maximum filter resets; −1 is unlimited and zero disables resets.
  max_filter_resets: int = -1

  # Watchdog: zero trigger disables it; trial limit includes rejected trials.
  watchdog_shortened_iter_trigger: int = 10
  watchdog_trial_iter_max: int = 3

  # Soft restoration: zero reduction factor disables it. The iteration limit
  # counts continuations after the initial soft-restoration trial (FATROP).
  soft_rest_pd_error_reduction_factor: float = 0.9999
  max_soft_resto_iters: int = 10

  # Full elastic ℓ₁ restoration. All optimization phases share max_iter.
  restoration_enabled: bool = True
  resto_penalty_parameter: float = 1000.0
  # Required fraction of entry infeasibility; zero requires exact feasibility.
  required_infeasibility_reduction: float = 0.9
  # Reset finite-side bound duals to one if this magnitude is exceeded.
  bound_mult_reset_threshold: float = 1000.0

  # Second-order corrections (SOC).
  # Maximum SOC trials per search; zero disables SOC.
  max_soc: int = 4
  # Required feasibility contraction to continue correcting a rejected SOC.
  kappa_soc: float = 0.99

  # Tiny steps: IPOPT's numerical-stagnation heuristic.
  # Relative primal-step threshold; zero disables detection. Default: 10 ε64.
  tiny_step_tol: float = 10 * math.ulp(1.0)
  # Equality-dual step threshold for repeat history; zero disables that history.
  tiny_step_y_tol: float = 1e-2

  # Primal and stage-equality dual regularization.
  # Stage-equality dual shift coefficient: δc = delta_c_stripe × μ**kappa_c.
  delta_c_stripe: float = 1e-6
  # Barrier exponent in the dual shift.
  kappa_c: float = 0.25
  # First positive primal shift without a previous successful shift.
  delta_w0: float = 1e-4
  # Minimum first shift when reusing a previous successful shift.
  delta_w_min: float = 1e-20
  # Largest primal shift attempted before reporting step-computation failure.
  delta_w_max: float = 1e40
  # Reduce the previous successful shift by this factor on the first retry.
  kappa_wmin: float = 1.0 / 3.0
  # Retry growth factor when a previous successful shift is available.
  kappa_wplus: float = 8.0
  # Retry growth factor without a previous successful shift.
  kappa_wplusem: float = 100.0
  # Relative SVD rank threshold for each stage-control equality Jacobian.
  rank_tol: float = 1e-12

  # Newton-system iterative refinement.
  # Maximum correction solves after the initial solve; zero checks only.
  max_refinement: int = 3
  # Require ‖Newton defect‖∞ ≤ linear_ref_tol × (1 + ‖RHS‖∞).
  linear_ref_tol: float = 1e-10

  # Output: zero is silent, positive prints iterations, > 1 adds Newton details.
  print_level: int = 0
  # Save the initialized iterate and each subsequent iterate in the result.
  record_intermediates: bool = False

  def __post_init__(self) -> None:
    positive_counts = ('filter_reset_trigger', 'watchdog_trial_iter_max')
    nonnegative_reals = (
        'kappa_d',
        'bound_relax_factor',
        'tiny_step_tol',
        'tiny_step_y_tol',
        'soft_rest_pd_error_reduction_factor',
        'required_infeasibility_reduction',
        'bound_mult_reset_threshold',
    )
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
    if self.delta_w_min > self.delta_w0 or self.delta_w0 > self.delta_w_max:
      raise ValueError('Require delta_w_min <= delta_w0 <= delta_w_max.')
    if min(self.kappa_wplus, self.kappa_wplusem) <= 1:
      raise ValueError('Regularization growth factors must exceed one.')
    if self.kappa_sigma < 1:
      raise ValueError('kappa_sigma must be >= 1.')
    if self.theta_min_fact > self.theta_max_fact:
      raise ValueError('Require theta_min_fact <= theta_max_fact.')

    for name in (
        'soft_rest_pd_error_reduction_factor',
        'required_infeasibility_reduction',
    ):
      if getattr(self, name) >= 1:
        raise ValueError(f'{name} must lie in [0, 1).')
