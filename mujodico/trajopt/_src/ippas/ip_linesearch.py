"""Globalize a primal-dual Newton direction with filter line search and SOC.

Keep μ fixed throughout a search. Measure each point by

  θ = ‖r₀‖₁ + ‖rᶠ‖₁ + ‖rᵉ‖₁ + ‖rᵍ‖₁ + ‖rᵍᶠ‖₁,
  φ = objective + logarithmic slack barrier + one-sided linear slack damping.

Search sequence:

  1. Compute separate maximum step length for primal slacks and z duals.
  2. Try the maximum primal step and test acceptance as described below.
  3. If rejected without reducing θ, try up to max_soc second-order
     corrections (SOC). A failed SOC does not replace the original Newton
     direction used for backtracking.
  4. If still rejected, geometrically reduce the original primal step by
     α_red_factor. Keep its bound-dual step fixed. Stop when the next primal
     step would be at or below minimum_step(), or cannot decrease further.

Every accepted trial must be finite, keep slacks strictly interior and finite-
side bound duals positive, and satisfy quaternion validity. A tiny primal
direction with θ₀ ≤ 10⁻⁴ permits a valid finite maximum-step trial without
progress/filter tests or filter augmentation [IPOPT, sec. 3.9]. Repeated tiny
steps with small equality-dual changes trigger barrier reduction, then
unsuccessful TINY_STEP termination if μ cannot decrease further.

Other trials require θ ≤ θₘₐₓ and must pass every stored filter corner:

  θ ≤ θᵢ ∨ φ ≤ φᵢ, ∀i.

For progress relative to the current point (θ₀, φ₀), use the f-type switch

  dφ < 0 ∧ αₜₑₛₜ (−dφ)ˢᵠ > δ θ₀ˢᶿ.                   [IPOPT, eq. (19)]

The exponents s_φ and s_θ correspond to options.s_phi and options.s_theta.
If this holds and θ₀ < θₘᵢₙ, require Armijo decrease:

  φ − φ₀ ≤ η_φ αₜₑₛₜ dφ.                             [IPOPT, eq. (20)]

Otherwise require

  θ ≤ (1 − γ_θ) θ₀ ∨ φ − φ₀ ≤ −γ_φ θ₀,             [IPOPT, eq. (18)]

subject to the obj_max_incr barrier-objective-rise safeguard. For ordinary
trials, αₜₑₛₜ is the current primal step, including reductions during backtracking.
For SOC, αₜₑₛₜ remains the original maximum primal step.

After acceptance, augment the filter with the shifted current-point corner
((1 − γ_θ) θ₀, φ₀ − γ_φ θ₀), unless the step passes both the f-type
switch and Armijo test using the same αₜₑₛₜ as acceptance.

Watchdog trials instead use a frozen reference and may be committed
provisionally without updating the filter. The outer algorithm owns their
rollback snapshots. Soft restoration uses a common primal/dual step and
sufficient reduction of the block-average primal-dual residual before handing
control to elastic ℓ₁ restoration if recovery fails.
"""

from collections.abc import Callable
from typing import Generic, NamedTuple, TypeVar

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_filter, ip_iterate
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas.ip_iterate import (
    Iterate,
    IterateData,
    damping,
    distances,
)
from mujodico.trajopt._src.ippas.options import Options
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel
from mujodico.trajopt._src.types import Params


Point = TypeVar('Point')
System = TypeVar('System')
Bounds = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


class LineSearchState(NamedTuple):
  filter: ip_filter.Filter
  theta_min: jax.Array
  theta_max: jax.Array

  @staticmethod
  def create(
      capacity: int, theta: jax.Array, options: Options
  ) -> 'LineSearchState':
    """Create filter storage and θₘᵢₙ/θₘₐₓ for the current μ."""
    return LineSearchState(
        ip_filter.empty_filter(capacity, theta.dtype),
        options.theta_min_fact * jp.maximum(1.0, theta),
        options.theta_max_fact * jp.maximum(1.0, theta),
    )


class Search(NamedTuple, Generic[Point]):
  """A single trial point and its acceptance decision."""

  alpha: jax.Array
  iterate: Point
  evaluation: dict[str, jax.Array]
  accepted: jax.Array


def detect_tiny_step(
    it: Iterate,
    direction: pd.IpPrimalDualSystem,
    theta: jax.Array,
    options: Options,
) -> jax.Array:
  """Detect an insignificant primal direction [IPOPT, sec. 3.9].

  Require max |Δv|/(1 + |v|) ≤ tiny_step_tol for states, controls, and stage
  and terminal slacks, with θ ≤ 10⁻⁴. For manifold states with different
  tangent and storage dimensions, use unscaled ‖Δx‖∞ instead. Zero tolerance
  disables detection. Multiplier increments are checked by the algorithm's
  repeated-tiny-step policy, separately from this primal test.
  """
  scaled_x = (
      direction.dx / (1 + jp.abs(it.x))
      if direction.dx.shape[-1] == it.x.shape[-1]
      else direction.dx
  )
  size = ip_iterate.max_abs(
      scaled_x,
      direction.du / (1 + jp.abs(it.u)),
      direction.ds / (1 + jp.abs(it.s)),
      direction.dsf / (1 + jp.abs(it.sf)),
  )
  return (
      (options.tiny_step_tol > 0)
      & (size <= options.tiny_step_tol)
      & (theta <= 1e-4)
  )


def _le(left: jax.Array, right: jax.Array, reference: jax.Array) -> jax.Array:
  rounding = (
      10
      * jp.finfo(jp.result_type(left, right)).eps
      * jp.maximum(1.0, jp.abs(reference))
  )
  return left <= right + rounding


def f_type(
    alpha_test: jax.Array,
    theta: jax.Array,
    dphi: jax.Array,
    options: Options,
) -> jax.Array:
  eps = jp.finfo(dphi.dtype).eps
  dphi = jp.where((theta == 0) & (dphi > 0) & (dphi < 100 * eps), -eps, dphi)
  return (dphi < 0) & (
      alpha_test * jp.maximum(-dphi, 0) ** options.s_phi
      > options.delta * theta**options.s_theta
  )


def armijo(
    phi: jax.Array,
    phi0: jax.Array,
    alpha_test: jax.Array,
    dphi: jax.Array,
    options: Options,
) -> jax.Array:
  return _le(phi - phi0, options.eta_phi * alpha_test * dphi, phi0)


def acceptable(
    phi: jax.Array,
    theta: jax.Array,
    phi0: jax.Array,
    theta0: jax.Array,
    dphi: jax.Array,
    alpha_test: jax.Array,
    state: LineSearchState,
    options: Options,
) -> tuple[jax.Array, jax.Array]:
  """Return (accepted, rejected_by_filter) for a trial (θ, φ).

  rejected_by_filter is true only when finite values, θ ≤ θₘₐₓ, and progress
  from (θ₀, φ₀) pass, but the stored filter rejects the point. The caller
  checks iterate validity and masks both flags before using them.
  """
  switching = f_type(alpha_test, theta0, dphi, options)
  sufficient = (
      _le(theta, (1 - options.gamma_theta) * theta0, theta0)
      | _le(phi - phi0, -options.gamma_phi * theta0, phi0)
  )
  scale = jp.where(jp.abs(phi0) > 10, jp.log10(jp.maximum(jp.abs(phi0), 1.0)), 1.0)
  rise_ok = (phi <= phi0) | (
      jp.log10(jp.maximum(phi - phi0, jp.finfo(phi.dtype).tiny))
      <= options.obj_max_incr + scale
  )
  current_ok = jp.where(
      switching & (theta0 < state.theta_min),
      armijo(phi, phi0, alpha_test, dphi, options),
      sufficient & rise_ok,
  )
  filter_ok = ip_filter.filter_acceptable(state.filter, theta, phi)
  eligible = (
      jp.isfinite(phi)
      & jp.isfinite(theta)
      & (theta <= state.theta_max)
      & current_ok
  )
  return eligible & filter_ok, eligible & ~filter_ok


def minimum_step(
    theta: jax.Array,
    dphi: jax.Array,
    state: LineSearchState,
    options: Options,
) -> jax.Array:
  """Return the primal-step cutoff for terminating backtracking [IPOPT, eq. (23)].

  θ₀ (theta) is the current constraint-violation norm, and dφ (dphi) is the
  directional derivative of the barrier objective at the current iterate.
  For θ₀ > 0 ∧ dφ < 0, start with

    min(γ_θ, γ_φ θ₀ / (−dφ)).

  If θ₀ ≤ θₘᵢₙ, also take the minimum with δ θ₀ˢᶿ / (−dφ)ˢᵠ, reflecting
  the f-type switching threshold. When θ₀ = 0 ∨ dφ ≥ 0, use γ_θ alone.
  Multiply the resulting value by α_min_frac (options.alpha_min_frac).

  The caller stops backtracking if the next reduced step is at or below
  this cutoff. This is a termination safeguard, not an acceptance test.
  """
  descent = jp.maximum(-dphi, jp.finfo(dphi.dtype).tiny)
  alpha = jp.minimum(options.gamma_theta, options.gamma_phi * theta / descent)
  alpha = jp.where(
      theta <= state.theta_min,
      jp.minimum(
          alpha,
          options.delta * theta**options.s_theta / descent**options.s_phi,
      ),
      alpha,
  )
  return options.alpha_min_frac * jp.where(
      (theta > 0) & (dphi < 0), alpha, options.gamma_theta
  )


def maximum_step(
    value: jax.Array,
    step: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    tau: jax.Array,
) -> jax.Array:
  """Return the largest fraction-to-boundary step α in [0, 1].

  Write v = value, Δv = step, l = lower, u = upper, and τ = tau.
  For v strictly inside its finite bounds and 0 < τ < 1, require
  v + α Δv to retain at least (1 − τ) of each current distance:

    v + α Δv − l ≥ (1 − τ) (v − l),
    u − v − α Δv ≥ (1 − τ) (u − v).

  Only movement toward a finite bound restricts α. Take the minimum limit
  over all entries and both sides, capped at one. Infinite bounds, zero
  increments, and empty arrays impose no restriction. Used for both primal
  slacks and bound multipliers, with zero as the multiplier lower bound.
  """
  dl, du = distances(value, lower, upper)
  lower_limit = jp.where(
      jp.isfinite(lower) & (step < 0),
      -tau * dl / jp.where(step < 0, step, -1.0),
      1.0,
  )
  upper_limit = jp.where(
      jp.isfinite(upper) & (step > 0),
      tau * du / jp.where(step > 0, step, 1.0),
      1.0,
  )
  return jp.clip(
      jp.minimum(
          jp.min(lower_limit, initial=1.0), jp.min(upper_limit, initial=1.0)
      ),
      0.0,
      1.0,
  )


def maximum_step_lengths(
    it: Iterate,
    direction: pd.IpPrimalDualSystem,
    mu: jax.Array,
    bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    options: Options,
) -> tuple[jax.Array, jax.Array]:
  """Return primal α and dual α_dual limits using τ = max(τₘᵢₙ, 1 − μ)."""
  tau = jp.maximum(options.tau_min, 1 - mu)

  alpha_primal = jp.minimum(
      maximum_step(it.s, direction.ds, *bounds[:2], tau),
      maximum_step(it.sf, direction.dsf, *bounds[2:], tau),
  )

  alpha_dual = jp.asarray(1.0, mu.dtype)
  for z, dz, bound in (
      (it.z_l, direction.dz_l, bounds[0]),
      (it.z_u, direction.dz_u, bounds[1]),
      (it.zf_l, direction.dzf_l, bounds[2]),
      (it.zf_u, direction.dzf_u, bounds[3]),
  ):
    alpha_dual = jp.minimum(
        alpha_dual,
        maximum_step(
            z,
            dz,
            jp.where(jp.isfinite(bound), 0.0, -jp.inf),
            jp.full_like(z, jp.inf),
            tau,
        ),
    )

  return alpha_primal, alpha_dual


def candidate(
    model: StagewiseOcpModel,
    it: Iterate,
    direction: pd.IpPrimalDualSystem,
    alpha_primal: jax.Array,
    alpha_dual: jax.Array,
    mu: jax.Array,
    params: Params,
    bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    options: Options,
) -> Iterate:
  """Retract primals and update equality/bound duals with their own step lengths."""
  s = it.s + alpha_primal * direction.ds
  sf = it.sf + alpha_primal * direction.dsf

  def step_z(z, dz, distance, bound) -> jax.Array:
    """Update a bound multiplier and safeguard its complementarity with μ.

      z⁺ = clip(z + α_dual Δz, μ / (κ_σ dist), κ_σ μ / dist),

    The central-path value μ / dist satisfies dist z = μ;
    clipping keeps dist z⁺ within [μ / κ_σ, κ_σ μ].
    """
    central = mu / distance
    return jp.where(
        jp.isfinite(bound),
        jp.clip(
            z + alpha_dual * dz,
            central / options.kappa_sigma,
            central * options.kappa_sigma,
        ),
        0.0,
    )

  dist_l, dist_u = distances(s, *bounds[:2])
  distf_l, distf_u = distances(sf, *bounds[2:])

  return Iterate(
      model.retract_state(it.x, direction.dx, alpha_primal, params),
      model.retract_control(it.u, direction.du, alpha_primal, params),
      s,
      sf,
      it.lmbda + alpha_primal * direction.dlmbda,
      it.beta + alpha_primal * direction.dbeta,
      it.y + alpha_primal * direction.dy,
      it.yf + alpha_primal * direction.dyf,
      step_z(it.z_l, direction.dz_l, dist_l, bounds[0]),
      step_z(it.z_u, direction.dz_u, dist_u, bounds[1]),
      step_z(it.zf_l, direction.dzf_l, distf_l, bounds[2]),
      step_z(it.zf_u, direction.dzf_u, distf_u, bounds[3]),
  )


def evaluate_trial(
    model: StagewiseOcpModel, iterate: Iterate, params: Params
) -> dict[str, jax.Array]:
  """Evaluate only trial values; never compute gradients or Hessians."""
  m, it = model, iterate
  return {
      'objective': jp.sum(m.objective(it.x, it.u, params)),
      'r0': m.initial_residual(it.x[0], params),
      'r_f': m.dynamics_residual(it.x[:-1], it.u, it.x[1:], params),
      'r_e': m.stage_equality_residual(it.x[:-1], it.u, params, xs_next=it.x[1:]),
      'r_g': m.stage_inequality_residual(it.x[:-1], it.u, params) - it.s,
      'r_gf': m.terminal_inequality_residual(it.x[-1], params) - it.sf,
  }


_EQUALITY_CONSTRAINT_FIELDS = ('r0', 'r_f', 'r_e', 'r_g', 'r_gf')


def merit(
    it: Iterate,
    evaluation: dict[str, jax.Array],
    mu: jax.Array,
    bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    options: Options,
) -> tuple[jax.Array, jax.Array]:
  """Return the barrier objective φ and equality-residual ℓ₁ norm θ."""
  objective = evaluation['objective']

  barrier = jp.asarray(0.0, objective.dtype)
  for s, l, u in ((it.s, *bounds[:2]), (it.sf, *bounds[2:])):
    dist_l, dist_u = distances(s, l, u)
    barrier -= mu * jp.sum(
        jp.where(jp.isfinite(l), jp.log(dist_l), 0)
        + jp.where(jp.isfinite(u), jp.log(dist_u), 0)
    )
    barrier += jp.sum(damping(l, u, mu, options) * s)

  theta = sum(jp.sum(jp.abs(evaluation[name])) for name in _EQUALITY_CONSTRAINT_FIELDS)

  return objective + barrier, theta


def find_acceptable_trial_point(
    model: StagewiseOcpModel,
    it: Iterate,
    data: IterateData,
    direction: pd.IpPrimalDualSystem,
    mu: jax.Array,
    params: Params,
    bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    state: LineSearchState,
    options: Options,
    *,
    skip_first: jax.Array = False,
) -> 'LineSearchResult':
  """Return the selected trial, filter state, rejection history, and tiny flag.

  A valid finite maximum-step tiny trial bypasses progress/filter tests and
  filter augmentation. An invalid tiny trial falls back to normal globalization.
  """
  phi0, theta0 = merit(it, vars(data), mu, bounds, options)
  dphi = directional_derivative(it, data, direction, mu, bounds, options)
  return search(
      it,
      vars(data),
      direction,
      state,
      options,
      phi0,
      theta0,
      dphi,
      lambda d, ap, ad: candidate(
          model, it, d, ap, ad, mu, params, bounds, options
      ),
      lambda point: evaluate_trial(model, point, params),
      lambda point, values: merit(point, values, mu, bounds, options),
      lambda point: ip_iterate.valid(point, bounds, model.quaternion_addrs),
      lambda d: maximum_step_lengths(it, d, mu, bounds, options),
      lambda d: pd.iterative_refinement_solve(
          d, options.linear_ref_tol, options.max_refinement
      ),
      detect_tiny_step(it, direction, theta0, options),
      skip_first,
  )


def directional_derivative(
    it: Iterate,
    data: IterateData,
    direction: pd.IpPrimalDualSystem,
    mu: jax.Array,
    bounds: Bounds,
    options: Options,
) -> jax.Array:
  """Barrier-objective directional derivative at the reference point."""
  dphi = (
      jp.sum(data.objective_gradx * direction.dx)
      + jp.sum(data.objective_gradu * direction.du)
  )
  for s, ds, l, u in (
      (it.s, direction.ds, *bounds[:2]),
      (it.sf, direction.dsf, *bounds[2:]),
  ):
    dist_l, dist_u = distances(s, l, u)
    dphi += jp.sum(
        ds * (
            -mu * jp.where(jp.isfinite(l), 1 / dist_l, 0.0)
            + mu * jp.where(jp.isfinite(u), 1 / dist_u, 0.0)
            + damping(l, u, mu, options)
        )
    )

  return dphi


class LineSearchResult(NamedTuple, Generic[Point]):
  search: Search[Point]
  state: LineSearchState
  filter_rejected: jax.Array
  tiny_step: jax.Array
  backtracked: jax.Array


def search(
    it: Point,
    values: dict[str, jax.Array],
    direction: System,
    state: LineSearchState,
    options: Options,
    phi0: jax.Array,
    theta0: jax.Array,
    dphi: jax.Array,
    make_candidate: Callable[[System, jax.Array, jax.Array], Point],
    evaluate: Callable[[Point], dict[str, jax.Array]],
    merit_value: Callable[
        [Point, dict[str, jax.Array]], tuple[jax.Array, jax.Array]
    ],
    valid: Callable[[Point], jax.Array],
    limits: Callable[[System], tuple[jax.Array, jax.Array]],
    refine: Callable[[System], tuple[System, jax.Array]],
    tiny_direction: bool | jax.Array = False,
    skip_first: bool | jax.Array = False,
) -> LineSearchResult[Point]:
  """Shared filter/SOC search; callbacks are static Python functions under JIT.

  Keep per-search outcome flags outside Search, which describes one trial.
  skip_first resumes a rolled-back watchdog search at the first reduced step.
  """
  max_alpha_primal, max_alpha_dual = limits(direction)

  def trial(
      direction: System,
      alpha_primal: jax.Array,
      alpha_dual: jax.Array,
      alpha_test: jax.Array,
      *,
      allow_tiny: bool | jax.Array = False,
  ) -> tuple[Search[Point], jax.Array, jax.Array, jax.Array]:
    trial_iterate = make_candidate(direction, alpha_primal, alpha_dual)
    evaluation = evaluate(trial_iterate)
    phi, theta = merit_value(trial_iterate, evaluation)
    iterate_valid = valid(trial_iterate)
    accepted, rejected_by_filter = acceptable(
        phi=phi,
        theta=theta,
        phi0=phi0,
        theta0=theta0,
        dphi=dphi,
        alpha_test=alpha_test,
        state=state,
        options=options,
    )
    tiny_taken = (
        allow_tiny
        & tiny_direction
        & iterate_valid
        & jp.isfinite(phi)
        & jp.isfinite(theta)
    )
    return (
        Search(
            alpha=alpha_primal,
            iterate=trial_iterate,
            evaluation=evaluation,
            accepted=iterate_valid & (accepted | tiny_taken),
        ),
        theta,
        iterate_valid & rejected_by_filter & ~tiny_taken,
        tiny_taken,
    )

  first, theta_first, first_filter_rejected, tiny_step_taken = trial(
      direction,
      jp.where(
          skip_first,
          max_alpha_primal * options.alpha_red_factor,
          max_alpha_primal,
      ),
      max_alpha_dual,
      jp.where(
          skip_first,
          max_alpha_primal * options.alpha_red_factor,
          max_alpha_primal,
      ),
      allow_tiny=~jp.asarray(skip_first),
  )

  # SOC modifies only constraint RHSs and reuses the Newton factors. A failed
  # correction never replaces the original Newton direction for backtracking.
  soc_initial = (
      first,
      {name: values[name] for name in _EQUALITY_CONSTRAINT_FIELDS},
      theta_first,
      jp.asarray(jp.inf),
      jp.asarray(0, jp.int32),
      first_filter_rejected,
  )

  def soc_cond(carry) -> jax.Array:
    candidate, _, theta, old_theta, count, _ = carry
    return (
        (~jp.asarray(skip_first))
        & (theta_first >= theta0)  # if theta_first < theta0, skip SOC
        & (~candidate.accepted)
        & (count < options.max_soc)
        & (theta < options.kappa_soc * old_theta)
    )

  def soc_body(carry):
    """Use SOC to compute new direction. For each equality residual, update

      c_acc⁺ = α_prev c_acc + c(v_prev)    [IPOPT, eq.(27) / A-5.9]
    """
    previous, accumulated, theta, _, count, any_filter_rejection = carry
    acc = {
        name: previous.alpha * accumulated[name] + previous.evaluation[name]
        for name in _EQUALITY_CONSTRAINT_FIELDS
    }
    corrected, accurate = refine(direction.replace_rhs(**acc))
    ap_soc, ad_soc = limits(corrected)
    candidate, new_theta, rejected_by_filter, _ = trial(
        corrected, ap_soc, ad_soc, max_alpha_primal
    )
    candidate = candidate._replace(accepted=candidate.accepted & accurate)
    return (
        candidate,
        acc,
        new_theta,
        theta,
        count + 1,
        any_filter_rejection | rejected_by_filter,
    )

  corrected, _, _, _, _, any_filter_rejection = jax.lax.while_loop(
      soc_cond, soc_body, soc_initial
  )

  alpha_min = minimum_step(theta0, dphi, state, options)

  def backtrack_cond(carry) -> jax.Array:
    candidate, _ = carry
    next_alpha = candidate.alpha * options.alpha_red_factor
    return (
        ~candidate.accepted
        & jp.isfinite(next_alpha)
        & (next_alpha < candidate.alpha)
        & (next_alpha > alpha_min)
    )

  def backtrack_body(carry):
    previous, any_filter_rejection = carry
    alpha = previous.alpha * options.alpha_red_factor
    candidate, _, rejected_by_filter, _ = trial(
        direction, alpha, max_alpha_dual, alpha
    )
    return (
        candidate,
        any_filter_rejection | rejected_by_filter,
    )

  found, any_filter_rejection = jax.lax.cond(
      corrected.accepted,
      lambda _: (corrected, any_filter_rejection),
      lambda _: jax.lax.while_loop(
          backtrack_cond,
          backtrack_body,
          (first, any_filter_rejection),
      ),
      None,
  )

  phi, _ = merit_value(found.iterate, found.evaluation)

  alpha_test = jp.where(corrected.accepted, max_alpha_primal, found.alpha)
  augment = (
      found.accepted
      & ~tiny_step_taken
      & (
          ~f_type(alpha_test, theta0, dphi, options)
          | ~armijo(phi, phi0, alpha_test, dphi, options)
      )
  )

  updated_filter = jax.lax.cond(
      augment,
      lambda f: ip_filter.augment_filter(f, theta0, phi0, options),
      lambda f: f,
      state.filter,
  )

  backtracked = jp.asarray(skip_first) | (
      ~corrected.accepted & (found.alpha < max_alpha_primal)
  )
  return LineSearchResult(
      found,
      state._replace(filter=updated_filter),
      any_filter_rejection,
      tiny_step_taken,
      backtracked,
  )


def update_filter_rejections(
    state: LineSearchState,
    accepted: jax.Array,
    any_filter_rejection: jax.Array,
    rejections: jax.Array,
    resets: jax.Array,
    options: Options,
) -> tuple[LineSearchState, jax.Array, jax.Array]:
  """Reset after repeated searches with filter-induced rejections [IPOPT, sec. 3.2].

  any_filter_rejection includes earlier rejected trials even if the selected
  trial was accepted. Reset only after an accepted step and within reset
  limits.
  """
  rejections = jp.where(any_filter_rejection, rejections + 1, 0)
  reset = (
      accepted
      & (rejections >= options.filter_reset_trigger)
      & ((options.max_filter_resets < 0) | (resets < options.max_filter_resets))
  )
  state = state._replace(
      filter=jax.lax.cond(reset, ip_filter.reset, lambda f: f, state.filter)
  )
  return state, jp.where(reset, 0, rejections), resets + reset.astype(jp.int32)


class WatchdogState(NamedTuple):
  """A fixed-shape rollback snapshot; derivative factors are recomputed."""

  active: jax.Array
  shortened: jax.Array
  trials: jax.Array
  iterate: Iterate
  phi: jax.Array
  theta: jax.Array
  dphi: jax.Array
  alpha: jax.Array
  last_delta: jax.Array
  acceptable_count: jax.Array

  @staticmethod
  def create(it: Iterate) -> 'WatchdogState':
    zero = jp.asarray(0, jp.int32)
    scalar = jp.asarray(0.0, it.x.dtype)
    return WatchdogState(
        jp.asarray(False),
        zero,
        zero,
        it,
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        zero,
    )

  def for_barrier_change(self, changed: jax.Array) -> 'WatchdogState':
    """Cancel the episode/history when μ changes, retaining the current point."""
    return self._replace(
        active=self.active & ~changed,
        shortened=jp.where(changed, 0, self.shortened),
    )


def primal_dual_error(
    it: Iterate,
    data: IterateData,
    mu: jax.Array,
    bounds: Bounds,
    options: Options,
) -> jax.Array:
  """FATROP soft-restoration error: sum of mean absolute residual blocks.

  Combine state/control stationarity in one block and stage/terminal slack
  stationarity in another. Each bound-side complementarity block includes
  masked zeros on infinite sides, matching FATROP's vector-size convention.
  """

  def mean(*values):
    return sum(jp.sum(jp.abs(v)) for v in values) / max(
        1, sum(v.size for v in values)
    )

  rs = -it.y - it.z_l + it.z_u + damping(*bounds[:2], mu, options)
  rsf = -it.yf - it.zf_l + it.zf_u + damping(*bounds[2:], mu, options)
  cl, cu = ip_iterate.complementarity(it.s, it.z_l, it.z_u, *bounds[:2], mu)
  cfl, cfu = ip_iterate.complementarity(
      it.sf, it.zf_l, it.zf_u, *bounds[2:], mu
  )
  return (
      mean(data.stationarity_x, data.stationarity_u)
      + mean(rs, rsf)
      + mean(data.r0, data.r_f, data.r_e, data.r_g, data.r_gf)
      + mean(cl, cfl)
      + mean(cu, cfu)
  )


def try_soft_resto_step(
    model: StagewiseOcpModel,
    it: Iterate,
    data: IterateData,
    direction: pd.IpPrimalDualSystem,
    mu: jax.Array,
    params: Params,
    bounds: Bounds,
    state: LineSearchState,
    options: Options,
) -> tuple[Search[Iterate], jax.Array]:
  """Try a common fraction-to-boundary step before full restoration.

  Return (trial, ordinary_acceptance). A valid finite step passing ordinary
  acceptance at α_test=0 exits soft restoration. Otherwise sufficient decrease
  of the FATROP primal-dual error allows a soft continuation.
  """
  ap, ad = maximum_step_lengths(it, direction, mu, bounds, options)
  alpha = jp.minimum(ap, ad)
  point = candidate(
      model, it, direction, alpha, alpha, mu, params, bounds, options
  )
  _, values = pd.evaluate(model, point, params, mu, bounds, options)
  phi, theta = merit(point, vars(values), mu, bounds, options)
  phi0, theta0 = merit(it, vars(data), mu, bounds, options)
  ordinary, _ = acceptable(
      phi,
      theta,
      phi0,
      theta0,
      jp.asarray(0.0, mu.dtype),
      jp.asarray(0.0, mu.dtype),
      state,
      options,
  )
  finite = (
      ip_iterate.valid(point, bounds, model.quaternion_addrs)
      & ip_iterate.all_finite(values)
      & jp.isfinite(phi)
  )
  decrease = primal_dual_error(
      point, values, mu, bounds, options
  ) <= options.soft_rest_pd_error_reduction_factor * primal_dual_error(
      it, data, mu, bounds, options
  )
  enabled = options.soft_rest_pd_error_reduction_factor > 0
  return (
      Search(
          alpha, point, vars(values), finite & enabled & (ordinary | decrease)
      ),
      finite & ordinary & enabled,
  )


def watchdog_trial(
    model: StagewiseOcpModel,
    it: Iterate,
    direction: pd.IpPrimalDualSystem,
    mu: jax.Array,
    params: Params,
    bounds: Bounds,
    state: LineSearchState,
    watchdog: WatchdogState,
    options: Options,
) -> tuple[Search[Iterate], jax.Array, LineSearchState]:
  """Evaluate a watchdog maximum step against its frozen reference.

  The caller may commit an eligible but unaccepted point provisionally; it
  must never commit a nonfinite or out-of-domain trial this way.
  """
  ap, ad = maximum_step_lengths(it, direction, mu, bounds, options)
  point = candidate(model, it, direction, ap, ad, mu, params, bounds, options)
  values = evaluate_trial(model, point, params)
  phi, theta = merit(point, values, mu, bounds, options)
  ordinary, _ = acceptable(
      phi,
      theta,
      watchdog.phi,
      watchdog.theta,
      watchdog.dphi,
      watchdog.alpha,
      state,
      options,
  )
  eligible = (
      ip_iterate.valid(point, bounds, model.quaternion_addrs)
      & jp.isfinite(phi)
      & jp.isfinite(theta)
  )
  augment = ordinary & (
      ~f_type(watchdog.alpha, watchdog.theta, watchdog.dphi, options)
      | ~armijo(phi, watchdog.phi, watchdog.alpha, watchdog.dphi, options)
  )
  updated = state._replace(
      filter=jax.lax.cond(
          eligible & augment,
          lambda f: ip_filter.augment_filter(
              f, watchdog.theta, watchdog.phi, options
          ),
          lambda f: f,
          state.filter,
      )
  )
  return Search(ap, point, values, eligible & ordinary), eligible, updated
