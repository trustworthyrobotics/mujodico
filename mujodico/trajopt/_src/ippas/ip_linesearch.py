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
     α_red_factor. Keep its bound-dual step fixed. Stop at max_ls trials
     or when the next primal step would be at or below minimum_step().

Every accepted trial must be finite, keep slacks strictly interior and finite-
side bound duals positive, satisfy quaternion validity, and have θ ≤ θₘₐₓ.
It must also pass every stored filter corner: θ ≤ θᵢ ∨ φ ≤ φᵢ, ∀i.

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
"""

from typing import NamedTuple

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


class Search(NamedTuple):
  """A single trial point and its acceptance decision."""

  alpha: jax.Array
  iterate: Iterate
  evaluation: dict[str, jax.Array]
  accepted: jax.Array


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
      'r_e': m.stage_equality_residual(it.x[:-1], it.u, params),
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
) -> tuple[Search, LineSearchState, jax.Array]:
  max_alpha_primal, max_alpha_dual = maximum_step_lengths(it, direction, mu, bounds, options)

  phi0, theta0 = merit(it, vars(data), mu, bounds, options)

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

  def trial(
      direction: pd.IpPrimalDualSystem,
      alpha_primal: jax.Array,
      alpha_dual: jax.Array,
      alpha_test: jax.Array,
  ) -> tuple[Search, jax.Array, jax.Array]:
    trial_iterate = candidate(
        model, it, direction, alpha_primal, alpha_dual, mu, params, bounds, options
    )
    evaluation = evaluate_trial(model, trial_iterate, params)
    phi, theta = merit(trial_iterate, evaluation, mu, bounds, options)
    iterate_valid = ip_iterate.valid(
        trial_iterate, bounds, model.quaternion_positions
    )
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
    return (
        Search(
            alpha=alpha_primal,
            iterate=trial_iterate,
            evaluation=evaluation,
            accepted=iterate_valid & accepted,
        ),
        theta,
        iterate_valid & rejected_by_filter,
    )

  first, theta_first, first_filter_rejected = trial(
      direction, max_alpha_primal, max_alpha_dual, max_alpha_primal
  )

  # SOC modifies only constraint RHSs and reuses the Newton factors. A failed
  # correction never replaces the original Newton direction for backtracking.
  soc_initial = (
      first,
      {name: getattr(data, name) for name in _EQUALITY_CONSTRAINT_FIELDS},
      theta_first,
      jp.asarray(jp.inf),
      jp.asarray(0, jp.int32),
      first_filter_rejected,
  )

  def soc_cond(carry) -> jax.Array:
    candidate, _, theta, old_theta, count, _ = carry
    return (
        (theta_first >= theta0)  # if theta_first < theta0, skip SOC
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
    corrected, accurate = pd.iterative_refinement_solve(
        direction.replace_rhs(**acc),
        linear_tol=options.linear_tol,
        max_refinement=options.max_refinement,
    )
    ap_soc, ad_soc = maximum_step_lengths(it, corrected, mu, bounds, options)
    candidate, new_theta, rejected_by_filter = trial(
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
    candidate, count, _ = carry
    return (
        ~candidate.accepted
        & (count < options.max_ls - 1)
        & (candidate.alpha * options.alpha_red_factor > alpha_min)
    )

  def backtrack_body(carry):
    previous, count, any_filter_rejection = carry
    alpha = previous.alpha * options.alpha_red_factor
    candidate, _, rejected_by_filter = trial(direction, alpha, max_alpha_dual, alpha)
    return (
        candidate,
        count + 1,
        any_filter_rejection | rejected_by_filter,
    )

  found, _, any_filter_rejection = jax.lax.cond(
      corrected.accepted,
      lambda _: (corrected, 0, any_filter_rejection),
      lambda _: jax.lax.while_loop(
          backtrack_cond,
          backtrack_body,
          (first, 0, any_filter_rejection),
      ),
      None,
  )

  phi, _ = merit(found.iterate, found.evaluation, mu, bounds, options)

  alpha_test = jp.where(corrected.accepted, max_alpha_primal, found.alpha)
  augment = found.accepted & (
      ~f_type(alpha_test, theta0, dphi, options)
      | ~armijo(phi, phi0, alpha_test, dphi, options)
  )

  updated_filter = jax.lax.cond(
      augment,
      lambda f: ip_filter.augment_filter(f, theta0, phi0, options),
      lambda f: f,
      state.filter,
  )

  return found, state._replace(filter=updated_filter), any_filter_rejection


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
