"""Behavioral regressions for compiled globalization and stopping policies."""

import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import unittest

from mujodico.trajopt._src.fatrop_ref.barrier import fraction_parameter, update_mu
from mujodico.trajopt._src.fatrop_ref.convergence import Status, check_convergence, max_abs, scaled_error
from mujodico.trajopt._src.fatrop_ref.filter import acceptable, augment, empty_filter, reset
from mujodico.trajopt._src.fatrop_ref.line_search import backtrack, minimum_step
from mujodico.trajopt._src.fatrop_ref.options import Options
from mujodico.trajopt._src.fatrop_ref.line_search import accumulate_rhs, run_soc, should_try
from mujodico.trajopt._src.fatrop_ref.line_search import watchdog_create as create, watchdog_record_shortened as record_shortened, watchdog_record_trial as record_trial, watchdog_restore as restore

jax.config.update('jax_enable_x64', True)


class PolicyTests(unittest.TestCase):

  def test_filter_shifted_corners_reject_only_dominated_trials(self):
    options = Options()
    state = jax.jit(lambda f: augment(f, 2.0, 3.0, options))(empty_filter(3))
    np.testing.assert_allclose(state.theta[0], 2.0 * (1 - options.gamma_theta))
    np.testing.assert_allclose(state.phi[0], 3.0 - options.gamma_phi * 2.0)
    assert not bool(acceptable(state, 2.0, 3.0))
    assert bool(acceptable(state, 1.0, 5.0))
    assert bool(acceptable(state, 4.0, 2.0))
    assert bool(acceptable(state, state.theta[0], state.phi[0]))
    assert bool(acceptable(reset(state), 1e10, 1e10))

  def test_filter_full_capacity_is_bounded(self):
    options = Options()
    state = empty_filter(1)
    state = augment(state, 2.0, 3.0, options)
    state = jax.jit(lambda f: augment(f, 1.0, 2.0, options))(state)
    assert int(state.count) == 1
    assert not bool(acceptable(state, 2.0, 3.0))

  def test_barrier_only_reduces_after_subproblem_is_solved_and_respects_floor(
      self,
  ):
    options = Options()
    update = jax.jit(
        lambda mu, error: update_mu(mu, error, options.tol, options)
    )
    mu, changed = update(0.1, 2.0)
    assert float(mu) == 0.1
    assert not bool(changed)
    mu, changed = update(0.1, 1e-3)
    np.testing.assert_allclose(mu, 0.02)
    assert bool(changed)
    floor = max(options.mu_min, options.tol / 11.0)
    mu, _ = update(floor, 0.0)
    np.testing.assert_allclose(mu, floor)
    assert not bool(update(0.1, jnp.nan)[1])
    np.testing.assert_allclose(fraction_parameter(1e-3, options), 0.999)

  def test_convergence_acceptable_sequence_resets_and_success_precedes_limit(
      self,
  ):
    options = dataclasses.replace(Options(), acceptable_iter=2)
    check = jax.jit(
        lambda err, count, iteration: check_convergence(
            err, 0.0, iteration, count, options
        )
    )
    status, count = check(5e-7, 0, 1)
    assert int(status) == Status.RUNNING and int(count) == 1
    status, count = check(2e-6, count, 2)
    assert int(status) == Status.RUNNING and int(count) == 0
    status, count = check(5e-7, count, 3)
    status, count = check(5e-7, count, 4)
    assert int(status) == Status.ACCEPTABLE
    assert int(check(0.0, 0, options.max_iter)[0]) == Status.SUCCESS
    assert int(check(1.0, 0, options.max_iter)[0]) == Status.MAX_ITER
    assert int(check(jnp.nan, 0, 1)[0]) == Status.NUMERICAL_ERROR

  def test_empty_residual_blocks_and_multiplier_scaling(self):
    assert float(jax.jit(max_abs)(jnp.zeros(0))) == 0.0
    error = jax.jit(
        lambda: scaled_error(
            jnp.array([100.0]),
            jnp.array([0.1]),
            jnp.array([10.0]),
            jnp.array([1e5]),
            Options(),
        )
    )()
    np.testing.assert_allclose(error, 0.1)

  def test_backtracking_rejects_overshoot_then_accepts_descent(self):
    options = Options()
    evaluate = lambda alpha: (
        (1.0 - 4.0 * alpha) ** 2,
        jnp.asarray(0.0),
        jnp.asarray(True),
    )
    result = jax.jit(
        lambda: backtrack(
            evaluate, 1.0, 1.0, 0.0, -8.0, empty_filter(4), options
        )
    )()
    assert bool(result.accepted)
    np.testing.assert_allclose(result.alpha, 0.25)
    assert int(result.backtracks) == 2

  def test_nonfinite_line_search_is_bounded_and_returns_last_tested_step(self):
    options = dataclasses.replace(Options(), max_ls=4)
    evaluate = lambda alpha: (jnp.asarray(jnp.nan), alpha, jnp.asarray(False))
    result = jax.jit(
        lambda: backtrack(
            evaluate, 1.0, 1.0, 0.0, -1.0, empty_filter(4), options
        )
    )()
    assert not bool(result.accepted)
    np.testing.assert_allclose(result.alpha, 0.125)
    assert int(result.backtracks) == 3
    assert float(minimum_step(0.0, -1.0, options)) > 0.0

  def test_filter_rejection_is_distinguished_from_failed_current_progress(self):
    options = dataclasses.replace(Options(), max_ls=3)
    state = augment(empty_filter(4), 0.1, 0.1, options)
    evaluate = lambda alpha: (
        jnp.asarray(5.0),
        jnp.asarray(0.5),
        jnp.asarray(True),
    )
    result = jax.jit(
        lambda: backtrack(evaluate, 1.0, 10.0, 1.0, 1.0, state, options)
    )()
    assert not bool(result.accepted)
    assert bool(result.filter_rejected)

  def test_soc_trigger_and_accumulated_rhs(self):
    assert bool(should_try(1.0, 1.0, 1.0, 1.0, True))
    assert not bool(should_try(0.5, 1.0, 1.0, 2.0, True))
    assert not bool(should_try(1.0, 1.0, 1.0, 2.0, False))
    rhs = jax.jit(
        lambda: accumulate_rhs(
            (jnp.array([1.0]), jnp.zeros(0)),
            0.25,
            (jnp.array([2.0]), jnp.zeros(0)),
        )
    )()
    np.testing.assert_allclose(rhs[0], [1.5])

  def test_soc_retries_while_violation_decreases_and_accepts_corrected_point(
      self,
  ):
    result = jax.jit(
        lambda: run_soc(
            lambda state: (state * 0.5, state * 0.5),
            lambda state: state < 0.3,
            jnp.asarray(1.0),
            1.0,
            4,
            0.99,
        )
    )()
    assert bool(result.accepted)
    assert int(result.iterations) == 2
    np.testing.assert_allclose(result.state, 0.25)

  def test_soc_stops_on_lack_of_improvement_and_can_be_disabled(self):
    solve = lambda count: run_soc(
        lambda state: (state * 1.1, state * 1.1),
        lambda state: state < 0.3,
        jnp.asarray(1.0),
        1.0,
        count,
        0.99,
    )
    assert int(jax.jit(lambda: solve(4))().iterations) == 1
    result = jax.jit(lambda: solve(0))()
    assert int(result.iterations) == 0
    assert not bool(result.accepted)

  def test_watchdog_checkpoints_and_rolls_back_after_bounded_excursion(self):
    checkpoint = {'x': jnp.array([2.0, 3.0]), 'mu': jnp.asarray(0.1)}
    initial = create(checkpoint)
    record = jax.jit(lambda state: record_shortened(state, True, checkpoint, 2))
    state = record(record(initial))
    assert bool(state.active)
    state, rollback = record_trial(state, False, 2)
    assert not bool(rollback)
    state, rollback = jax.jit(lambda s: record_trial(s, False, 2))(state)
    assert bool(rollback) and not bool(state.active)
    saved, state = restore(state)
    np.testing.assert_allclose(saved['x'], checkpoint['x'])
    assert int(state.trial_count) == 0

  def test_watchdog_acceptance_ends_provisional_mode(self):
    checkpoint = jnp.ones(2)
    state = record_shortened(create(checkpoint), True, checkpoint, 1)
    state, rollback = jax.jit(lambda s: record_trial(s, True, 3))(state)
    assert not bool(state.active) and not bool(rollback)

  def test_options_are_hashable_and_reject_invalid_static_settings(self):
    assert isinstance(hash(Options()), int)
    with self.assertRaisesRegex(ValueError, 'linear_backend'):
      Options(linear_backend='unknown')
    with self.assertRaisesRegex(ValueError, 'alpha_red_factor'):
      Options(alpha_red_factor=1.0)


if __name__ == '__main__':
  unittest.main()
