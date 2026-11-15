from dataclasses import replace

from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import ip_initializer
from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas import IppasOptions
from mujodico.trajopt._src.ippas.ip_linesearch import evaluate_trial
from mujodico.trajopt._src.ippas.test_util import point_mass_model


class IpInitializerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())
    self.model, self.guess = point_mass_model()
    self.options = IppasOptions()
    self.bounds = ip_initializer.get_bounds(self.model, self.options)

  def test_original_bound_interior(self):
    lower = jp.array([0.0, -jp.inf, -2.0, -jp.inf])
    upper = jp.array([1.0, 0.0, 2.0, jp.inf])
    s = ip_initializer.push_interior(
        jp.array([-3.0, 2.0, 4.0, 5.0]), lower, upper, self.options
    )
    self.assertTrue(np.all(np.asarray(s) > np.asarray(lower)))
    self.assertTrue(np.all(np.asarray(s) < np.asarray(upper)))

  def test_eq_mult_initializer(self):
    def initialize(xs, us):
      guess = replace(self.guess, xs=xs, us=us)
      it = ip_initializer.initialize(
          self.model, None, guess, self.bounds, self.options
      )
      system, _ = pd.evaluate(
          self.model, it, None, jp.array(0.0), self.bounds, self.options
      )
      lmbda, beta = ip_initializer.initialize_multipliers(system, self.options)
      return replace(it, lmbda=lmbda, beta=beta)

    result = jax.jit(initialize)(self.guess.xs, self.guess.us)
    self.assertTrue(bool(ip_iterate.all_finite(result)))
    self.assertLessEqual(
        float(ip_iterate.max_abs(result.lmbda, result.beta)),
        self.options.lam_max,
    )

  def _check_bound_push(self, value):
    guess = replace(
        self.guess,
        xs=jp.full_like(self.guess.xs, value),
        us=jp.full_like(self.guess.us, value),
    )
    raw = self.model.stage_inequality_residual(guess.xs[:-1], guess.us, None)
    before = float(jp.sum(jp.abs(raw)))
    result = jax.jit(
        lambda xs, us: ip_initializer.initialize(
            self.model,
            None,
            replace(guess, xs=xs, us=us),
            self.bounds,
            self.options,
        )
    )(guess.xs, guess.us)
    lower, upper = self.bounds[:2]
    self.assertTrue(np.all(np.asarray(result.s) > np.asarray(lower)))
    self.assertTrue(np.all(np.asarray(result.s) < np.asarray(upper)))
    self.assertLess(
        float(jp.sum(jp.abs(evaluate_trial(self.model, result, None)['r_g']))),
        before,
    )
    np.testing.assert_array_equal(result.u, guess.us)
    np.testing.assert_array_equal(result.z_l, jp.isfinite(lower))
    np.testing.assert_array_equal(result.z_u, jp.isfinite(upper))

  def test_bound_push_low(self):
    self._check_bound_push(-100.0)

  def test_bound_push_high(self):
    self._check_bound_push(100.0)


if __name__ == '__main__':
  absltest.main()
