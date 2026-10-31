# Portions follow FATROP's Ipopt-derived interior-point policies.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
# See licenses/LICENSE-EPL-2.0.txt and NOTICE.md.
"""Line search, SOC, watchdog and soft restoration in device-side control flow."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from . import barrier, iterate as bounds, linear_solver
from . import filter as filters
from . import search_direction
from . import filter as filter_module
from .iterate import Iterate, metrics
from .convergence import Status


class LineSearchResult(NamedTuple):
  alpha: object
  accepted: object
  backtracks: object
  filter_rejected: object


def _le(lhs, rhs, reference):
  dtype = jnp.result_type(lhs, rhs, reference, 1.0)
  rounding = 10.0 * jnp.finfo(dtype).eps * jnp.maximum(1.0, jnp.abs(reference))
  return lhs <= rhs + rounding


def is_f_type(alpha_test, theta0, dphi, options):
  return (dphi < 0.0) & (
      alpha_test * jnp.maximum(-dphi, 0.0) ** options.s_phi
      > options.delta * theta0**options.s_theta
  )


def trial_acceptable(
    phi, theta, finite, alpha_test, phi0, theta0, dphi, state, options
):
  """Return (accepted, rejected_by_filter), without mutating filter memory.

  As in FATROP, the reference maximum step is used for the switching and
  Armijo tests, including when testing a shortened or SOC trial point.
  """
  phi, theta = jnp.asarray(phi), jnp.asarray(theta)
  theta_min = options.theta_min_fact * jnp.maximum(1.0, theta0)
  theta_max = options.theta_max_fact * jnp.maximum(1.0, theta0)
  f_type = is_f_type(alpha_test, theta0, dphi, options) & (theta0 < theta_min)
  armijo = _le(phi - phi0, options.eta_phi * alpha_test * dphi, phi0)
  sufficient_reduction = _le(
      theta, (1.0 - options.gamma_theta) * theta0, theta0
  ) | _le(phi - phi0, -options.gamma_phi * theta0, phi0)
  # This logarithmic comparison avoids overflow in the allowed objective rise.
  scale = jnp.where(
      jnp.abs(phi0) > 10.0, jnp.log10(jnp.maximum(jnp.abs(phi0), 1.0)), 1.0
  )
  rise_ok = (phi <= phi0) | (
      jnp.log10(jnp.maximum(phi - phi0, jnp.finfo(phi.dtype).tiny))
      <= options.obj_max_incr + scale
  )
  current_ok = jnp.where(f_type, armijo, sufficient_reduction & rise_ok)
  valid = (
      finite & jnp.isfinite(phi) & jnp.isfinite(theta) & (theta <= theta_max)
  )
  filter_ok = filter_module.acceptable(state, theta, phi)
  return valid & current_ok & filter_ok, valid & current_ok & ~filter_ok


def minimum_step(theta0, dphi, options):
  descent = jnp.maximum(-dphi, jnp.finfo(jnp.result_type(dphi, 1.0)).tiny)
  alpha_min = jnp.asarray(options.gamma_theta)
  theta_positive = theta0 > 0.0
  alpha_min = jnp.where(
      (dphi < 0.0) & theta_positive,
      jnp.minimum(alpha_min, options.gamma_phi * theta0 / descent),
      alpha_min,
  )
  theta_min = options.theta_min_fact * jnp.maximum(1.0, theta0)
  switching_floor = (
      options.delta * theta0**options.s_theta / descent**options.s_phi
  )
  alpha_min = jnp.where(
      (dphi < 0.0) & theta_positive & (theta0 <= theta_min),
      jnp.minimum(alpha_min, switching_floor),
      alpha_min,
  )
  return options.alpha_min_frac * alpha_min


def accept_trial(phi, theta, alpha, phi0, theta0, dphi, state, options):
  return trial_acceptable(
      phi, theta, jnp.asarray(True), alpha, phi0, theta0, dphi, state, options
  )[0]


def backtrack(evaluate, alpha_max, phi0, theta0, dphi, state, options):
  """Find an acceptable step; ``evaluate(alpha)`` returns phi, theta, finite."""
  alpha_max = jnp.asarray(
      alpha_max, dtype=jnp.result_type(phi0, theta0, dphi, 1.0)
  )
  alpha_min = minimum_step(theta0, dphi, options)
  # Carry both the next proposed step and the last evaluated step. On failure,
  # return the evaluated step rather than an untested extra contraction.
  initial = (
      alpha_max,
      alpha_max,
      jnp.asarray(False),
      jnp.asarray(0, jnp.int32),
      jnp.asarray(False),
  )

  def condition(carry):
    alpha, _, accepted, attempts, _ = carry
    return (
        ~accepted
        & (attempts < options.max_ls)
        & ((attempts == 0) | (alpha > alpha_min))
    )

  def body(carry):
    alpha, _, _, attempts, rejected_before = carry
    phi, theta, finite = evaluate(alpha)
    accepted, filter_rejected = trial_acceptable(
        phi, theta, finite, alpha_max, phi0, theta0, dphi, state, options
    )
    return (
        alpha * options.alpha_red_factor,
        alpha,
        accepted,
        attempts + 1,
        rejected_before | filter_rejected,
    )

  _, tested, accepted, attempts, filter_rejected = jax.lax.while_loop(
      condition, body, initial
  )
  return LineSearchResult(
      tested, accepted, jnp.maximum(attempts - 1, 0), filter_rejected
  )


class SocResult(NamedTuple):
  state: object
  accepted: object
  iterations: object
  theta: object


def should_try(alpha, alpha_max, theta0, theta_trial, finite):
  return finite & (alpha == alpha_max) & (theta_trial >= theta0)


def accumulate_rhs(accumulated, alpha, trial_residual):
  """Accumulate the original and trial constraint defects for a SOC solve."""
  return jax.tree.map(lambda a, b: a + alpha * b, accumulated, trial_residual)


def run_soc(
    make_candidate, accept, initial_state, initial_theta, max_soc, kappa
):
  """Run correction solves while rejected candidates improve feasibility.

  ``make_candidate(state)`` returns a new candidate PyTree and its violation;
  it can close over a cached factorization. ``accept(state)`` applies the
  line-search acceptance test. A failed result must not replace the iterate.
  """
  initial_theta = jnp.asarray(initial_theta)
  initial = (
      initial_state,
      initial_theta,
      initial_theta,
      jnp.asarray(False),
      jnp.asarray(0, jnp.int32),
  )

  def condition(carry):
    _, theta, old_theta, accepted, iterations = carry
    improving = (iterations == 0) | (theta < kappa * old_theta)
    return (iterations < max_soc) & ~accepted & improving & jnp.isfinite(theta)

  def body(carry):
    state, theta, _, _, iterations = carry
    candidate, candidate_theta = make_candidate(state)
    candidate_theta = jnp.asarray(candidate_theta, dtype=initial_theta.dtype)
    accepted = jnp.isfinite(candidate_theta) & accept(candidate)
    return candidate, candidate_theta, theta, accepted, iterations + 1

  state, theta, _, accepted, iterations = jax.lax.while_loop(
      condition, body, initial
  )
  return SocResult(state, accepted, iterations, theta)


class WatchdogState(NamedTuple):
  checkpoint: object
  active: object
  shortened_count: object
  trial_count: object


def watchdog_create(checkpoint):
  return WatchdogState(
      checkpoint,
      jnp.asarray(False),
      jnp.asarray(0, jnp.int32),
      jnp.asarray(0, jnp.int32),
  )


def watchdog_start(state, checkpoint):
  return WatchdogState(
      checkpoint,
      jnp.asarray(True),
      jnp.asarray(0, jnp.int32),
      jnp.asarray(0, jnp.int32),
  )


def watchdog_restore(state):
  """Return the saved iterate and cleared watchdog metadata."""
  return state.checkpoint, watchdog_create(state.checkpoint)


def watchdog_record_shortened(
    state, shortened, checkpoint, trigger, enabled=True
):
  count = jnp.where(shortened, state.shortened_count + 1, 0)
  arm = enabled & ~state.active & (count >= trigger) & (trigger > 0)
  updated = state._replace(shortened_count=count)
  return jax.lax.cond(
      arm,
      lambda _: watchdog_start(updated, checkpoint),
      lambda _: updated,
      operand=None,
  )


def watchdog_record_trial(state, accepted, max_steps):
  """Return (new metadata, rollback_required) for a provisional trial."""
  accepted = jnp.asarray(accepted, jnp.bool_)
  count = jnp.where(state.active & ~accepted, state.trial_count + 1, 0)
  rollback = state.active & ~accepted & (count >= max_steps)
  finish = state.active & (accepted | rollback)
  updated = state._replace(trial_count=count)
  updated = jax.lax.cond(
      finish,
      lambda _: watchdog_create(state.checkpoint),
      lambda _: updated,
      operand=None,
  )
  return updated, rollback


class Step(NamedTuple):
  iterate: Iterate
  accepted: object
  alpha: object
  alpha_dual: object
  backtracks: object
  soc_steps: object
  kind: object
  filter_rejected: object


class SocState(NamedTuple):
  direction: object
  dzl: object
  dzu: object
  alpha: object
  alpha_dual: object
  accumulated: object
  accumulatedT: object
  trial: object
  phi: object
  theta: object
  finite: object


def _merit(problem, primal, params, mu, lower, upper, kappa=1e-5):
  e = problem.evaluate(primal, params)
  b, valid = bounds.barrier(bounds.data(primal, lower, upper), mu, kappa)
  phi = e.objective + b
  theta = jnp.sum(jnp.abs(e.eq)) + jnp.sum(jnp.abs(e.eqT))
  finite = (
      valid
      & problem.valid(primal, params)
      & jnp.isfinite(phi)
      & jnp.isfinite(theta)
  )
  return phi, theta, finite


def _candidate(
    problem,
    current,
    direction,
    dzl,
    dzu,
    alpha,
    ad,
    params,
    mu,
    lower,
    upper,
    options,
):
  primal = problem.retract(current.primal, direction, alpha, params)
  bd = bounds.data(primal, lower, upper)

  def update(z, dz, dist, mask):
    value = z + ad * dz
    central = mu / dist
    value = jnp.clip(
        value, central / options.kappa_sigma, central * options.kappa_sigma
    )
    return jnp.where(mask, value, 0.0)

  zl = bounds.tree_map(update, current.zl, dzl, bd.dl, bd.ml)
  zu = bounds.tree_map(update, current.zu, dzu, bd.du, bd.mu)
  return Iterate(
      primal,
      current.eq + alpha * direction.beta,
      current.eqT + alpha * direction.betaT,
      zl,
      zu,
  )


def _directional_merit(problem, current, direction, params, mu, bd, kappa=1e-5):
  dtype = current.primal.xs.dtype
  derivative = jax.jvp(
      lambda a: problem.evaluate(
          problem.retract(current.primal, direction, a, params), params
      ).objective,
      (jnp.asarray(0.0, dtype),),
      (jnp.asarray(1.0, dtype),),
  )[1]
  dq = search_direction.bound_step(problem, current.primal, direction)
  derivative -= mu * (
      bounds.tree_sum(
          bounds.tree_map(
              lambda v, d, m: jnp.where(m, v / d, 0.0), dq, bd.dl, bd.ml
          )
      )
      - bounds.tree_sum(
          bounds.tree_map(
              lambda v, d, m: jnp.where(m, v / d, 0.0), dq, bd.du, bd.mu
          )
      )
  )
  derivative += bounds.tree_sum(
      bounds.tree_map(lambda a, b: a * b, dq, bounds.damping(bd, mu, kappa))
  )
  return derivative


def _globalize(
    problem, current, search, params, mu, lower, upper, filter_state, options
):
  bd = bounds.data(current.primal, lower, upper)
  tau = barrier.fraction_parameter(mu, options)
  dq = search_direction.bound_step(problem, current.primal, search.direction)
  alpha, ad = bounds.maximum_step(
      bd, current.zl, current.zu, dq, search.dzl, search.dzu, tau
  )
  phi0, theta0, _ = _merit(
      problem, current.primal, params, mu, lower, upper, options.kappa_d
  )
  dphi = _directional_merit(
      problem, current, search.direction, params, mu, bd, options.kappa_d
  )

  def candidate(d, dzl, dzu, ap, adu):
    return _candidate(
        problem,
        current,
        d,
        dzl,
        dzu,
        ap,
        adu,
        params,
        mu,
        lower,
        upper,
        options,
    )

  first = candidate(search.direction, search.dzl, search.dzu, alpha, ad)
  phi, theta, finite = _merit(
      problem, first.primal, params, mu, lower, upper, options.kappa_d
  )
  accepted, rejected = trial_acceptable(
      phi, theta, finite, alpha, phi0, theta0, dphi, filter_state, options
  )
  e = problem.evaluate(current.primal, params)
  initial = SocState(
      search.direction,
      search.dzl,
      search.dzu,
      alpha,
      ad,
      e.eq,
      e.eqT,
      first,
      phi,
      theta,
      finite,
  )
  backend = (
      'parallel' if options.linear_backend == 'pas' else options.linear_backend
  )

  def corrected(c):
    ev = problem.evaluate(c.trial.primal, params)
    ac = c.accumulated + c.alpha * ev.eq
    at = c.accumulatedT + c.alpha * ev.eqT
    rhs = search.rhs._replace(c=ac, cT=at)
    direction = linear_solver.solve_rhs(search.factors, rhs, backend)
    dq = search_direction.bound_step(problem, current.primal, direction)
    dzl, dzu = bounds.dual_step(bd, current.zl, current.zu, dq, mu)
    ap, adu = bounds.maximum_step(bd, current.zl, current.zu, dq, dzl, dzu, tau)
    trial = candidate(direction, dzl, dzu, ap, adu)
    pp, tt, ff = _merit(
        problem, trial.primal, params, mu, lower, upper, options.kappa_d
    )
    return SocState(direction, dzl, dzu, ap, adu, ac, at, trial, pp, tt, ff), tt

  def accept_soc(c):
    return trial_acceptable(
        c.phi,
        c.theta,
        c.finite,
        alpha,
        phi0,
        theta0,
        dphi,
        filter_state,
        options,
    )[0]

  should = (~accepted) & should_try(alpha, alpha, theta0, theta, finite)
  correction = jax.lax.cond(
      should,
      lambda _: run_soc(
          corrected,
          accept_soc,
          initial,
          theta,
          options.max_soc,
          options.kappa_soc,
      ),
      lambda _: SocResult(
          initial, jnp.asarray(False), jnp.asarray(0, jnp.int32), theta
      ),
      None,
  )

  def backtrack_direction(_):
    def evaluate(a):
      primal = problem.retract(current.primal, search.direction, a, params)
      return _merit(problem, primal, params, mu, lower, upper, options.kappa_d)

    found = backtrack(
        evaluate, alpha, phi0, theta0, dphi, filter_state, options
    )
    trial = candidate(search.direction, search.dzl, search.dzu, found.alpha, ad)
    return Step(
        trial,
        found.accepted,
        found.alpha,
        ad,
        found.backtracks,
        correction.iterations,
        jnp.asarray(1, jnp.int32),
        found.filter_rejected,
    )

  ordinary = Step(
      first,
      accepted,
      alpha,
      ad,
      jnp.asarray(0, jnp.int32),
      correction.iterations,
      jnp.asarray(1, jnp.int32),
      rejected,
  )
  corrected_step = Step(
      correction.state.trial,
      correction.accepted,
      correction.state.alpha,
      correction.state.alpha_dual,
      jnp.asarray(0, jnp.int32),
      correction.iterations,
      jnp.asarray(2, jnp.int32),
      rejected,
  )
  step = jax.lax.cond(
      accepted,
      lambda _: ordinary,
      lambda _: jax.lax.cond(
          correction.accepted,
          lambda _: corrected_step,
          backtrack_direction,
          None,
      ),
      None,
  )
  return step, first, alpha, ad, phi0, theta0, dphi


class Globalized(NamedTuple):
  iterate: object
  mu: object
  filter: object
  watchdog: object
  step: object
  kind: object
  accepted: object
  rollback: object
  restoration_iterations: object
  restoration_status: object
  rejections: object
  resets: object


def find_acceptable_trial_point(
    problem,
    st,
    search,
    params,
    new_mu,
    changed,
    lower,
    upper,
    original_lower,
    original_upper,
    options,
    allow_restore=True,
):
  from .restoration import perform_restoration

  old = st.iterate
  dtype = old.primal.xs.dtype
  fs = jax.lax.cond(changed, filters.reset, lambda f: f, st.filter)
  step, full, alpha, ad, phi0, theta0, dphi = _globalize(
      problem, old, search, params, new_mu, lower, upper, fs, options
  )
  rejected = st.filter_rejections + step.filter_rejected.astype(jnp.int32)
  reset_allowed = (options.max_filter_resets < 0) | (
      st.filter_resets < options.max_filter_resets
  )
  do_reset = (
      (~step.accepted)
      & step.filter_rejected
      & (rejected >= options.filter_reset_trigger)
      & reset_allowed
  )

  def reset_search(_):
    ff = filters.reset(fs)
    ss = _globalize(
        problem, old, search, params, new_mu, lower, upper, ff, options
    )[0]
    return ff, ss

  fs, step = jax.lax.cond(do_reset, reset_search, lambda _: (fs, step), None)
  reset_count = st.filter_resets + do_reset.astype(jnp.int32)
  rejected = jnp.where(do_reset | step.accepted, 0, rejected)

  # Watchdog trials compare against the checkpoint, then either commit or
  # restore its primal, dual, filter and barrier state together.
  wd = jax.lax.cond(
      changed,
      lambda _: watchdog_create(st.watchdog.checkpoint),
      lambda _: st.watchdog,
      None,
  )
  cp = wd.checkpoint
  fresh = wd.active & (wd.trial_count == 0)
  cp = cp._replace(
      phi=jnp.where(fresh, phi0, cp.phi),
      theta=jnp.where(fresh, theta0, cp.theta),
      dphi=jnp.where(fresh, dphi, cp.dphi),
      alpha=jnp.where(fresh, alpha, cp.alpha),
  )
  wd = wd._replace(checkpoint=cp)
  full_phi, full_theta, full_finite = _merit(
      problem, full.primal, params, new_mu, lower, upper, options.kappa_d
  )
  watchdog_accept = trial_acceptable(
      full_phi,
      full_theta,
      full_finite,
      cp.alpha,
      cp.phi,
      cp.theta,
      cp.dphi,
      cp.filter,
      options,
  )[0]
  watchdog_active = wd.active
  wd_next, rollback = watchdog_record_trial(
      wd, watchdog_accept, options.watchdog_max
  )
  rollback = wd.active & (rollback | ~full_finite)
  provisional = Step(
      full,
      jnp.asarray(True),
      alpha,
      ad,
      jnp.asarray(0, jnp.int32),
      jnp.asarray(0, jnp.int32),
      jnp.asarray(5, jnp.int32),
      jnp.asarray(False),
  )
  step = jax.lax.cond(wd.active, lambda _: provisional, lambda _: step, None)
  wd = jax.lax.cond(wd.active, lambda _: wd_next, lambda _: wd, None)

  # Soft restoration accepts a maximal Newton step only after a measurable
  # reduction of the barrier KKT error. It has a bounded consecutive count.
  def soft(_):
    common = jnp.minimum(alpha, ad)
    trial = _candidate(
        problem,
        old,
        search.direction,
        search.dzl,
        search.dzu,
        common,
        common,
        params,
        new_mu,
        lower,
        upper,
        options,
    )
    mm = metrics(
        problem,
        trial,
        params,
        new_mu,
        lower,
        upper,
        original_lower,
        original_upper,
        options,
    )
    reference = metrics(
        problem,
        old,
        params,
        new_mu,
        lower,
        upper,
        original_lower,
        original_upper,
        options,
    )
    good = mm.finite & (
        mm.barrier_error
        <= options.soft_resto_reduction * reference.barrier_error
    )
    return jax.lax.cond(
        good,
        lambda _: Step(
            trial,
            good,
            common,
            common,
            step.backtracks,
            step.soc_steps,
            jnp.asarray(3, jnp.int32),
            step.filter_rejected,
        ),
        lambda _: step,
        None,
    )

  can_soft = (
      (~step.accepted)
      & (st.soft_count < options.max_soft_resto_iters)
      & (search.status == 0)
  )
  step = jax.lax.cond(can_soft, soft, lambda _: step, None)

  restored_iters = jnp.asarray(0, jnp.int32)
  restore_status = jnp.asarray(Status.LINE_SEARCH_FAILED, jnp.int32)
  if allow_restore and options.enable_restoration:

    def restore(_):
      resto_filter = filters.augment(fs, theta0, phi0, options)
      it, ok, niter, rs = perform_restoration(
          problem, old, params, new_mu, resto_filter, options
      )
      return (
          Step(
              it,
              ok,
              jnp.asarray(1.0, dtype),
              jnp.asarray(1.0, dtype),
              step.backtracks,
              step.soc_steps,
              jnp.asarray(4, jnp.int32),
              step.filter_rejected,
          ),
          niter,
          rs,
          resto_filter,
      )

    step, restored_iters, restore_status, fs = jax.lax.cond(
        (~step.accepted) & (search.status == 0),
        restore,
        lambda _: (step, restored_iters, restore_status, fs),
        None,
    )

  accepted = step.accepted & (search.status == 0)
  it = jax.tree_util.tree_map(
      lambda a, b: jnp.where(accepted, a, b), step.iterate, old
  )
  it = jax.tree_util.tree_map(
      lambda a, b: jnp.where(rollback, a, b), cp.iterate, it
  )
  fs = jax.lax.cond(rollback, lambda _: cp.filter, lambda _: fs, None)
  new_mu = jnp.where(rollback, cp.mu, new_mu)
  kind = jnp.where(rollback, 6, jnp.where(accepted, step.kind, 0)).astype(
      jnp.int32
  )
  # Store filter corners only for actual filter/SOC steps requiring it.
  augment = (
      accepted
      & ~rollback
      & ((step.kind == 1) | (step.kind == 2))
      & (~is_f_type(alpha, theta0, dphi, options))
  )
  fs = jax.lax.cond(
      augment,
      lambda f: filters.augment(f, theta0, phi0, options),
      lambda f: f,
      fs,
  )
  watchdog_corner = (
      watchdog_active
      & watchdog_accept
      & ~rollback
      & (~is_f_type(cp.alpha, cp.theta, cp.dphi, options))
  )
  fs = jax.lax.cond(
      watchdog_corner,
      lambda f: filters.augment(f, cp.theta, cp.phi, options),
      lambda f: f,
      fs,
  )
  checkpoint = cp._replace(
      iterate=it,
      filter=fs,
      mu=new_mu,
      phi=phi0,
      theta=theta0,
      dphi=dphi,
      alpha=alpha,
  )
  wd = watchdog_record_shortened(
      wd,
      (step.alpha < alpha * 0.99) & accepted & ~rollback,
      checkpoint,
      options.watchdog_trigger,
      options.enable_watchdog,
  )
  wd = jax.lax.cond(rollback, lambda _: watchdog_create(cp), lambda _: wd, None)
  return Globalized(
      it,
      new_mu,
      fs,
      wd,
      step,
      kind,
      accepted,
      rollback,
      restored_iters,
      restore_status,
      rejected,
      reset_count,
  )


def detect_tiny_step(direction, dtype):
  return (
      jnp.maximum(
          jnp.max(jnp.abs(direction.dx), initial=0.0),
          jnp.maximum(
              jnp.max(jnp.abs(direction.dy), initial=0.0),
              jnp.max(jnp.abs(direction.dyT), initial=0.0),
          ),
      )
      < 10 * jnp.finfo(dtype).eps
  )
