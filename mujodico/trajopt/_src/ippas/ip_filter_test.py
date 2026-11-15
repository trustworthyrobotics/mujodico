from absl.testing import absltest
import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_filter as g
from mujodico.trajopt._src.ippas import IppasOptions


class IpFilterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())
    self.options = IppasOptions()

  def test_default_constructor(self):
    self.assertEqual(int(g.empty_filter(10, jp.float64).count), 0)

  def test_reset(self):
    state = g.empty_filter(10, jp.float64)
    self.assertEqual(int(g.reset(state).count), 0)
    state = g.augment_filter(state, jp.array(0.0), jp.array(1.0), self.options)
    state = jax.jit(g.reset)(state)
    self.assertEqual(int(state.count), 0)
    self.assertTrue(
        bool(g.filter_acceptable(state, jp.array(2.0), jp.array(2.0)))
    )

  def test_reserve(self):
    state = g.empty_filter(10, jp.float64)
    self.assertEqual(state.theta.shape, (10,))
    self.assertEqual(state.phi.shape, (10,))
    state = jax.jit(
        lambda f: jax.lax.fori_loop(
            0,
            10,
            lambda i, f: g.augment_filter(
                f, i.astype(jp.float64), i.astype(jp.float64), self.options
            ),
            f,
        )
    )(state)
    self.assertEqual(int(state.count), 10)

  def test_is_acceptable_empty_filter(self):
    state = g.empty_filter(10, jp.float64)
    self.assertTrue(
        bool(g.filter_acceptable(state, jp.array(0.0), jp.array(1.0)))
    )

  def test_augment_and_is_acceptable(self):
    state = g.empty_filter(10, jp.float64)
    # Upstream data is ordered (objective, violation); our arrays take theta,phi.
    state = g.augment_filter(state, jp.array(0.0), jp.array(1.0), self.options)
    self.assertEqual(int(state.count), 1)
    self.assertTrue(
        bool(g.filter_acceptable(state, jp.array(0.0), jp.array(1.0)))
    )
    self.assertTrue(
        bool(g.filter_acceptable(state, jp.array(0.5), jp.array(0.5)))
    )
    state = g.augment_filter(state, jp.array(0.5), jp.array(0.5), self.options)
    self.assertEqual(int(state.count), 2)
    self.assertFalse(
        bool(g.filter_acceptable(state, jp.array(2.0), jp.array(2.0)))
    )

  def test_size(self):
    state = g.empty_filter(10, jp.float64)
    self.assertEqual(int(state.count), 0)
    state = g.augment_filter(state, jp.array(0.0), jp.array(1.0), self.options)
    self.assertEqual(int(state.count), 1)
    state = g.augment_filter(state, jp.array(0.5), jp.array(0.5), self.options)
    self.assertEqual(int(state.count), 2)

  def test_filter_shift_and_strict_dominance(self):
    o = self.options
    f = g.empty_filter(4, jp.float64)
    f = g.augment_filter(f, jp.array(1.0), jp.array(2.0), o)
    self.assertAlmostEqual(float(f.theta[0]), 1 - o.gamma_theta)
    self.assertAlmostEqual(float(f.phi[0]), 2 - o.gamma_phi)
    # Equal corners are accepted by the filter (FATROP uses strict dominance).
    accept = lambda theta, phi: g.filter_acceptable(
        f, jp.array(theta), jp.array(phi)
    )
    self.assertTrue(bool(accept(f.theta[0], 3.0)))
    self.assertTrue(bool(accept(3.0, f.phi[0])))
    self.assertFalse(bool(accept(2.0, 3.0)))


if __name__ == '__main__':
  absltest.main()
