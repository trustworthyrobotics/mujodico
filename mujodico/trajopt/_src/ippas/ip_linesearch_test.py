from dataclasses import replace

from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import ip_filter as glob
from mujodico.trajopt._src.ippas import ip_initializer
from mujodico.trajopt._src.ippas import ip_linesearch
from mujodico.trajopt._src.ippas import ip_linesearch as g
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas import ip_search_dir
from mujodico.trajopt._src.ippas import IppasInitialGuess
from mujodico.trajopt._src.ippas import IppasOptions
from mujodico.trajopt._src.ippas.ip_iterate import damping
from mujodico.trajopt._src.ippas.test_util import simple_model


class IpLinesearchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())
    self.options = IppasOptions()

  def test_fraction_to_boundary(self):
    lower = jp.array([0.0, -jp.inf, -2.0, -jp.inf])
    upper = jp.array([1.0, 0.0, 2.0, jp.inf])
    s = jp.array([0.1, -0.1, 1.9, 5.0])
    delta = jp.array([-2.0, 2.0, 0.0, 100.0])
    alpha = jax.jit(g.maximum_step)(s, delta, lower, upper, jp.array(0.99))
    trial = s + alpha * delta
    self.assertTrue(np.all(np.asarray(trial) > np.asarray(lower)))
    self.assertTrue(np.all(np.asarray(trial) < np.asarray(upper)))

  def test_independent_slack_equality_multipliers(self):
    m = simple_model(
        N=1,
        stage_inequality_constraint=lambda x, u, k, p: 1e3 * u,
        stage_inequality_bounds=(np.array([-1.0]), np.array([np.inf])),
    )
    p = jp.array([0.0, 2.0])
    m.initialize(p)
    options = IppasOptions()
    bounds = ip_initializer.get_bounds(m, options)
    it = ip_initializer.initialize(
        m,
        p,
        IppasInitialGuess(jp.zeros((2, 1)), jp.zeros((1, 1))),
        bounds,
        options,
    )
    # A shortened primal step and a larger bound-dual step must leave y and
    # zu-zl distinct. The reduced Newton RHS still accounts for that residual.
    it = replace(it, y=it.y + 0.2)
    sys, data = pd.evaluate(m, it, p, jp.array(0.1), bounds, options)
    direction, good, _ = ip_search_dir.compute(
        sys, jp.array(0.1), jp.array(0.0), options
    )
    self.assertTrue(bool(good))
    rs = -it.y - it.z_l + it.z_u + damping(*bounds[:2], jp.array(0.1), options)
    dy = direction.dz_u - direction.dz_l + rs
    control = (
        data.stationarity_u
        + jp.einsum('tij,tj->ti', sys.Huu, direction.du)
        + jp.einsum('tji,tj->ti', sys.B, direction.dlmbda[1:])
        + jp.einsum('tji,tj->ti', sys.Gu, dy)
    )
    np.testing.assert_allclose(control, 0.0, atol=1e-9)
    candidate = ip_linesearch.candidate(
        m,
        it,
        direction,
        jp.array(0.1),
        jp.array(0.8),
        jp.array(0.1),
        p,
        bounds,
        options,
    )
    np.testing.assert_allclose(candidate.y, it.y + 0.1 * dy, atol=1e-12)
    self.assertGreater(
        float(jp.max(jp.abs(candidate.y - candidate.z_u + candidate.z_l))),
        1e-3,
    )

  def test_create_thresholds(self):
    create = jax.jit(
        lambda theta: g.LineSearchState.create(4, theta, self.options)
    )
    for theta in (0.0, 0.5, 20.0):
      state = create(jp.array(theta))
      self.assertEqual(int(state.filter.count), 0)
      self.assertAlmostEqual(
          float(state.theta_min), self.options.theta_min_fact * max(1, theta)
      )
      self.assertAlmostEqual(
          float(state.theta_max), self.options.theta_max_fact * max(1, theta)
      )

  def test_theta_max_rejects_trial_despite_objective_decrease(self):
    state = g.LineSearchState.create(4, jp.array(1.0), self.options)
    accept = jax.jit(
        lambda theta: g.acceptable(
            jp.array(0.0),
            theta,
            jp.array(1.0),
            jp.array(1.0),
            jp.array(0.0),
            jp.array(1.0),
            state,
            self.options,
        )
    )
    accepted, rejected_by_filter = accept(state.theta_max)
    self.assertTrue(bool(accepted))
    self.assertFalse(bool(rejected_by_filter))
    accepted, rejected_by_filter = accept(state.theta_max + 1)
    self.assertFalse(bool(accepted))
    self.assertFalse(bool(rejected_by_filter))

  def test_soc_overcomes_maratos_step(self):
    m = simple_model(
        N=1,
        nu=2,
        dynamics=lambda x, u, k, p: x,
        stage_cost=lambda x, u, k, p: 0.5 * jp.sum((u - 0.1) ** 2),
        terminal_cost=lambda x, p: jp.array(0.0),
        stage_equality_constraint=lambda x, u, k, p: jp.array(
            [jp.sum(u * u) - 1]
        ),
        gauss_newton=False,
    )
    p = jp.array([0.0, 0.0])
    m.initialize(p)
    options = IppasOptions()
    bounds = ip_initializer.get_bounds(m, options)
    it = ip_initializer.initialize(
        m,
        p,
        IppasInitialGuess(jp.zeros((2, 1)), jp.array([[0.8, 0.6]])),
        bounds,
        options,
    )
    it = replace(it, beta=jp.array([[(np.sqrt(0.02) - 1) / 2]]))
    sys, data = pd.evaluate(m, it, p, jp.array(0.1), bounds, options)
    direction, _, _ = ip_search_dir.compute(
        sys, jp.array(0.1), jp.array(0.0), options
    )
    first = ip_linesearch.candidate(
        m,
        it,
        direction,
        jp.array(1.0),
        jp.array(1.0),
        jp.array(0.1),
        p,
        bounds,
        options,
    )
    self.assertGreater(
        float(ip_linesearch.evaluate_trial(m, first, p)['objective']),
        float(data.objective),
    )
    fs = g.LineSearchState.create(20, jp.array(0.0), options)
    step, _, _ = jax.jit(
        lambda: ip_linesearch.find_acceptable_trial_point(
            m, it, data, direction, jp.array(0.1), p, bounds, fs, options
        )
    )()
    self.assertTrue(bool(step.accepted))
    self.assertEqual(float(step.alpha), 1.0)  # The SOC took the full step.
    self.assertLess(float(step.evaluation['objective']), float(data.objective))

  def test_switching_and_armijo_near_feasibility(self):
    o = self.options
    f = g.LineSearchState.create(4, jp.array(0.0), o)
    common = (
        jp.array(1.0),
        jp.array(0.0),
        jp.array(-1.0),
        jp.array(1.0),
        f,
        o,
    )
    accepted, rejected_by_filter = g.acceptable(
        jp.array(0.9), jp.array(0.0), *common
    )
    self.assertTrue(bool(accepted))
    self.assertFalse(bool(rejected_by_filter))
    accepted, rejected_by_filter = g.acceptable(
        jp.array(1.1), jp.array(0.0), *common
    )
    self.assertFalse(bool(accepted))
    self.assertFalse(bool(rejected_by_filter))
    self.assertGreater(
        float(g.minimum_step(jp.array(0.0), jp.array(-1.0), f, o)), 0
    )
    accepted, rejected_by_filter = g.acceptable(
        jp.array(jp.nan), jp.array(0.0), *common
    )
    self.assertFalse(bool(accepted))
    self.assertFalse(bool(rejected_by_filter))

  def test_filter_reset_after_repeated_rejections(self):
    state = g.LineSearchState.create(10, jp.array(20.0), self.options)
    theta_min, theta_max = state.theta_min, state.theta_max
    state = state._replace(
        filter=glob.augment_filter(
            state.filter, jp.array(1.0), jp.array(1.0), self.options
        )
    )
    update = jax.jit(
        lambda f, n, r: ip_linesearch.update_filter_rejections(
            f, jp.array(True), jp.array(True), n, r, self.options
        )
    )
    count, resets = jp.array(0, jp.int32), jp.array(0, jp.int32)
    for _ in range(self.options.filter_reset_trigger - 1):
      state, count, resets = update(state, count, resets)
      self.assertEqual(int(state.filter.count), 1)
    state, count, resets = update(state, count, resets)
    self.assertEqual(int(state.filter.count), 0)
    self.assertEqual(float(state.theta_min), float(theta_min))
    self.assertEqual(float(state.theta_max), float(theta_max))
    self.assertEqual(int(count), 0)
    self.assertEqual(int(resets), 1)
    disabled = replace(self.options, max_filter_resets=0)
    state = state._replace(
        filter=glob.augment_filter(
            state.filter, jp.array(1.0), jp.array(1.0), self.options
        )
    )
    state, _, resets = ip_linesearch.update_filter_rejections(
        state,
        jp.array(True),
        jp.array(True),
        jp.array(100, jp.int32),
        jp.array(0, jp.int32),
        disabled,
    )
    self.assertEqual(int(state.filter.count), 1)
    self.assertEqual(int(resets), 0)


if __name__ == '__main__':
  absltest.main()
