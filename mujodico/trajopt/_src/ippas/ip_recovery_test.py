"""Compiled recovery transitions, rollback, and restoration return/failure."""

from dataclasses import replace
from unittest.mock import patch

from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import IppasSolver, IppasOptions, IppasInitialGuess, IppasStatus
from mujodico.trajopt._src.ippas import ip_filter
from mujodico.trajopt._src.ippas import ip_convergence_check_resto as conv
from mujodico.trajopt._src.ippas import ip_initializer
from mujodico.trajopt._src.ippas import ip_linesearch as ls
from mujodico.trajopt._src.ippas import ip_mu_update
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas import ip_resto_phase_min_cl1 as phase
from mujodico.trajopt._src.ippas import ip_search_dir
from mujodico.trajopt._src.ippas.test_util import simple_model
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel
from mujodico.trajopt._src import manifold


class RecoveryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())

  def quartic(self, **kwargs):
    return simple_model(
        N=1,
        stage_cost=lambda x, u, k, p: 0.25 * (u[0] - 1) ** 4,
        terminal_cost=lambda x, p: jp.array(0.0),
        **kwargs,
    )

  def fixture(self, m, options=None):
    o = options or IppasOptions()
    p = jp.array([0.0, 2.0])
    m.initialize(p)
    guess = IppasInitialGuess(jp.zeros((m.N + 1, m.nx)), jp.zeros((m.N, m.nu)))
    bounds = ip_initializer.get_bounds(m, o)
    it = ip_initializer.initialize(m, p, guess, bounds, o)
    system, data = pd.evaluate(m, it, p, jp.array(0.1), bounds, o)
    direction, good, _ = ip_search_dir.compute(
        system, jp.array(0.1), jp.array(0.0), o
    )
    self.assertTrue(bool(good))
    phi, theta = ls.merit(it, vars(data), jp.array(0.1), bounds, o)
    fs = ls.LineSearchState.create(o.max_iter + 1, theta, o)
    return o, p, guess, bounds, it, data, direction, fs, phi, theta

  def test_soft_error_progress_and_ordinary_exit(self):
    m = self.quartic()
    o, p, _, bounds, it, data, direction, fs, _, _ = self.fixture(m)
    blocked = fs._replace(
        filter=ip_filter.augment_filter(
            fs.filter, jp.array(-1.0), jp.array(-1.0), o
        )
    )
    trial, ordinary = jax.jit(
        lambda: ls.try_soft_resto_step(
            m, it, data, direction, jp.array(0.1), p, bounds, blocked, o
        )
    )()
    self.assertTrue(bool(trial.accepted))
    self.assertFalse(bool(ordinary))
    trial2, ordinary2 = ls.try_soft_resto_step(
        m, it, data, direction, jp.array(0.1), p, bounds, fs, o
    )
    self.assertTrue(bool(trial2.accepted))
    self.assertTrue(bool(ordinary2))
    disabled = replace(o, soft_rest_pd_error_reduction_factor=0)
    trial3, _ = ls.try_soft_resto_step(
        m, it, data, direction, jp.array(0.1), p, bounds, fs, disabled
    )
    self.assertFalse(bool(trial3.accepted))

  def test_watchdog_trial_filter_freeze_and_success(self):
    m = self.quartic()
    o, p, _, bounds, it, data, direction, fs, phi, theta = self.fixture(m)
    w = ls.WatchdogState.create(it)._replace(
        active=jp.array(True),
        phi=phi,
        theta=theta,
        dphi=ls.directional_derivative(
            it, data, direction, jp.array(0.1), bounds, o
        ),
        alpha=jp.array(1.0),
    )
    blocked = fs._replace(
        filter=ip_filter.augment_filter(
            fs.filter, jp.array(-1.0), jp.array(-1.0), o
        )
    )
    trial, eligible, newfs = ls.watchdog_trial(
        m, it, direction, jp.array(0.1), p, bounds, blocked, w, o
    )
    self.assertFalse(bool(trial.accepted))
    self.assertTrue(bool(eligible))
    jax.tree.map(np.testing.assert_array_equal, newfs, blocked)
    trial, eligible, _ = ls.watchdog_trial(
        m, it, direction, jp.array(0.1), p, bounds, fs, w, o
    )
    self.assertTrue(bool(trial.accepted))
    self.assertTrue(bool(eligible))
    bad = replace(direction, dx=jp.full_like(direction.dx, jp.nan))
    _, eligible, _ = ls.watchdog_trial(
        m, it, bad, jp.array(0.1), p, bounds, fs, w, o
    )
    self.assertFalse(bool(eligible))

  def test_watchdog_rolls_back_and_keeps_global_iteration_count(self):
    m = self.quartic()
    guess = IppasInitialGuess(jp.zeros((2, 1)), jp.zeros((1, 1)))
    real_search = ls.find_acceptable_trial_point
    real_watchdog = ls.watchdog_trial

    def shortened(*args, **kwargs):
      result = real_search(*args, **kwargs)
      return result._replace(backtracked=jp.array(True))

    def reject(*args):
      trial, eligible, _ = real_watchdog(*args)
      return trial._replace(accepted=jp.array(False)), eligible, args[6]

    with (
        patch.object(ls, 'find_acceptable_trial_point', shortened),
        patch.object(ls, 'watchdog_trial', reject),
        patch.object(ip_mu_update, 'update', lambda mu, *args, **kw: mu),
    ):
      solver = IppasSolver(
          m,
          IppasOptions(
              max_iter=4,
              watchdog_shortened_iter_trigger=1,
              watchdog_trial_iter_max=3,
          ),
      )
      r = jax.jit(solver.solve)(jp.array([0.0, 2.0]), guess)
    self.assertEqual(int(r.status), IppasStatus.MAX_ITER)
    self.assertEqual(int(r.iterations), 4)
    np.testing.assert_allclose(r.us, 1 / 3, atol=1e-12)
    np.testing.assert_allclose(r.xs[-1], 1 / 3, atol=1e-12)

  def test_soft_continuation_limit_enters_restoration_and_rolls_back(self):
    real = ls.find_acceptable_trial_point

    def reject(*args, **kwargs):
      result = real(*args, **kwargs)
      return result._replace(
          search=result.search._replace(accepted=jp.array(False))
      )

    real_soft = ls.try_soft_resto_step

    def stay_soft(*args):
      trial, _ = real_soft(*args)
      return trial, jp.array(False)

    with (
        patch.object(ls, 'try_soft_resto_step', stay_soft),
        patch.object(ls, 'find_acceptable_trial_point', reject),
        patch.object(ip_mu_update, 'update', lambda mu, *args, **kw: mu),
    ):
      solver = IppasSolver(
          self.quartic(), IppasOptions(max_iter=20, max_soft_resto_iters=0)
      )
      r = jax.jit(solver.solve)(
          jp.array([0.0, 2.0]),
          IppasInitialGuess(jp.zeros((2, 1)), jp.zeros((1, 1))),
      )
    self.assertEqual(int(r.status), IppasStatus.RESTORATION_FAILED)
    self.assertEqual(int(r.iterations), 2)
    np.testing.assert_allclose(r.us, 1 / 3, atol=1e-12)

  def test_restoration_recovers_original_solver_jit_mixed_vmap(self):
    m = simple_model(N=1, stage_equality_constraint=lambda x, u, k, p: u - p[1])
    guess = IppasInitialGuess(jp.zeros((2, 1)), jp.zeros((1, 1)))
    real = ls.find_acceptable_trial_point

    def reject_initial(*args, **kwargs):
      result = real(*args, **kwargs)
      blocked = jp.max(jp.abs(args[2].r_e)) > 0.9
      return result._replace(
          search=result.search._replace(
              accepted=result.search.accepted & ~blocked
          )
      )

    with patch.object(ls, 'find_acceptable_trial_point', reject_initial):
      solver = IppasSolver(
          m, IppasOptions(max_iter=40, soft_rest_pd_error_reduction_factor=0)
      )
      result = jax.jit(jax.vmap(lambda p: solver.solve(p, guess)))(
          jp.array([[0.0, 1.0], [0.0, 0.5]])
      )
    np.testing.assert_array_equal(result.status, IppasStatus.SUCCESS)
    np.testing.assert_allclose(result.us[:, 0, 0], [1.0, 0.5], atol=1e-8)
    self.assertTrue(bool(result.iterations[0] > result.iterations[1]))

  def test_local_infeasibility_and_budget_return_entry(self):
    m = simple_model(
        N=1, stage_equality_constraint=lambda x, u, k, p: jp.ones(1) + 0 * u
    )
    o, p, _, bounds, it, data, _, fs, phi, theta = self.fixture(m)
    fs = fs._replace(filter=ip_filter.augment_filter(fs.filter, theta, phi, o))
    run = jax.jit(
        lambda limit: phase.perform_restoration(
            m,
            it,
            data,
            p,
            jp.array(0.1),
            bounds,
            fs,
            limit,
            replace(o, max_iter=40),
        )
    )
    result = run(jp.array(0, jp.int32))
    self.assertEqual(int(result.status), phase.Status.LOCAL_INFEASIBILITY)
    jax.tree.map(np.testing.assert_array_equal, result.iterate, it)
    result = run(jp.array(40, jp.int32))
    self.assertEqual(int(result.status), phase.Status.MAX_ITER)
    self.assertEqual(int(result.iteration), 40)
    jax.tree.map(np.testing.assert_array_equal, result.iterate, it)

  def test_watchdog_convergence_precedes_budget_rollback(self):
    m = simple_model(
        N=1,
        stage_cost=lambda x, u, k, p: 0.5 * jp.sum((u - 1) ** 2),
        terminal_cost=lambda x, p: jp.array(0.0),
    )
    real_search = ls.find_acceptable_trial_point
    real_watchdog = ls.watchdog_trial

    def half_step(*args, **kwargs):
      result = real_search(*args, **kwargs)
      model, it, data, direction, mu, params, bounds, state, options = args
      point = ls.candidate(
          model,
          it,
          direction,
          jp.array(0.5),
          jp.array(0.5),
          mu,
          params,
          bounds,
          options,
      )
      return result._replace(
          search=result.search._replace(iterate=point),
          backtracked=jp.array(True),
      )

    def reject(*args):
      trial, eligible, _ = real_watchdog(*args)
      return trial._replace(accepted=jp.array(False)), eligible, args[6]

    with (
        patch.object(ls, 'find_acceptable_trial_point', half_step),
        patch.object(ls, 'watchdog_trial', reject),
        patch.object(ip_mu_update, 'update', lambda mu, *a, **kw: mu),
    ):
      solver = IppasSolver(
          m, IppasOptions(max_iter=2, watchdog_shortened_iter_trigger=1)
      )
      result = jax.jit(solver.solve)(
          jp.array([0.0, 2.0]),
          IppasInitialGuess(jp.zeros((2, 1)), jp.zeros((1, 1))),
      )
    self.assertEqual(int(result.status), IppasStatus.SUCCESS)
    np.testing.assert_allclose(result.us, 1.0, atol=1e-10)

  def test_watchdog_barrier_change_cancels_history(self):
    it = self.fixture(self.quartic())[4]
    state = ls.WatchdogState.create(it)._replace(
        active=jp.array(True), shortened=jp.array(12, jp.int32)
    )
    updated = jax.jit(lambda changed: state.for_barrier_change(changed))(
        jp.array(True)
    )
    self.assertFalse(bool(updated.active))
    self.assertEqual(int(updated.shortened), 0)
    self.assertTrue(bool(state.for_barrier_change(jp.array(False)).active))

  def test_quaternion_restoration(self):
    m = StagewiseOcpModel(
        1,
        4,
        3,
        initial_state=lambda p: jp.array([1.0, 0.0, 0.0, 0.0]),
        dynamics=lambda x, u, k, p: manifold.quat_multiply(
            x, manifold.quat_exp(u)
        ),
        stage_cost=lambda x, u, k, p: 0.5 * jp.sum(u * u),
        terminal_cost=lambda x, p: jp.array(0.0),
        stage_equality_constraint=lambda x, u, k, p: u - p,
        quaternion_addrs=(0,),
        full_dynamics_jacobian=True,
    )
    p = jp.array([0.2, 0.0, 0.0])
    m.initialize(p)
    o = IppasOptions(max_iter=40)
    bounds = ip_initializer.get_bounds(m, o)
    entry = ip_initializer.initialize(
        m,
        p,
        IppasInitialGuess(
            jp.tile(jp.array([1.0, 0.0, 0.0, 0.0]), (2, 1)), jp.zeros((1, 3))
        ),
        bounds,
        o,
    )
    _, data = pd.evaluate(m, entry, p, jp.array(0.1), bounds, o)
    phi, theta = ls.merit(entry, vars(data), jp.array(0.1), bounds, o)
    fs = ls.LineSearchState.create(41, theta, o)
    fs = fs._replace(filter=ip_filter.augment_filter(fs.filter, theta, phi, o))
    result = jax.jit(
        lambda: phase.perform_restoration(
            m,
            entry,
            data,
            p,
            jp.array(0.1),
            bounds,
            fs,
            jp.array(0, jp.int32),
            o,
        )
    )()
    self.assertEqual(int(result.status), phase.Status.RETURNED)
    np.testing.assert_allclose(
        jp.linalg.norm(result.iterate.x, axis=1), 1.0, atol=1e-12
    )
    self.assertLess(float(jp.max(jp.abs(result.iterate.u - p))), 0.1)

  def test_restoration_return_requires_reduction_and_filter_acceptance(self):
    m = self.quartic()
    o, p, _, bounds, it, data, _, fs, _, _ = self.fixture(m)
    values = ls.evaluate_trial(m, it, p)
    entry_merit = (jp.array(1.0), jp.array(1.0))

    def test(violation, options=o, filter_state=fs):
      trial = dict(
          values, r_e=jp.array([[violation]]), objective=jp.array(1e12)
      )
      return conv.acceptable_return(
          it,
          trial,
          entry_merit,
          jp.array(1.0),
          jp.array(0.1),
          bounds,
          filter_state,
          (),
          options,
      )

    self.assertFalse(bool(test(0.95)))
    self.assertTrue(
        bool(test(0.5))
    )  # Feasibility-driven return permits objective rise.
    blocked = fs._replace(
        filter=ip_filter.augment_filter(
            fs.filter, jp.array(0.0), jp.array(0.0), o
        )
    )
    self.assertFalse(bool(test(0.5, filter_state=blocked)))
    exact = replace(o, required_infeasibility_reduction=0)
    self.assertFalse(bool(test(1e-16, exact)))
    self.assertTrue(bool(test(0.0, exact)))

  def test_new_option_validation(self):
    for name in (
        'watchdog_shortened_iter_trigger',
        'watchdog_trial_iter_max',
        'max_soft_resto_iters',
        'resto_penalty_parameter',
        'required_infeasibility_reduction',
        'soft_rest_pd_error_reduction_factor',
        'bound_mult_reset_threshold',
    ):
      for bad in (-1.0, True, jp.nan, jp.inf):
        with self.assertRaises(ValueError):
          IppasOptions(**{name: bad})
    for name in (
        'required_infeasibility_reduction',
        'soft_rest_pd_error_reduction_factor',
    ):
      with self.assertRaises(ValueError):
        IppasOptions(**{name: 1.0})
    with self.assertRaises(ValueError):
      IppasOptions(restoration_enabled=1)


if __name__ == '__main__':
  absltest.main()
