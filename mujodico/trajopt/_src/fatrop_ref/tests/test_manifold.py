"""Numerical regression tests for manifold primitives."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mujodico.trajopt._src.fatrop_ref.manifold import Euclidean, ProductManifold, Quaternion
from mujodico.trajopt._src.fatrop_ref.manifold import quat_exp, quat_log, quaternion_product

jax.config.update('jax_enable_x64', True)


class ManifoldTest(unittest.TestCase):

  def test_exp_log_identity_derivatives(self):
    z = jnp.zeros(3)
    np.testing.assert_allclose(quat_exp(z), [1, 0, 0, 0], atol=1e-15)
    fn = lambda v: quat_log(quat_exp(v))
    np.testing.assert_allclose(jax.jacfwd(fn)(z), np.eye(3), atol=1e-13)
    np.testing.assert_allclose(jax.jacfwd(jax.jacrev(fn))(z), 0.0, atol=1e-13)
    self.assertTrue(
        np.isfinite(np.asarray(jax.hessian(lambda v: quat_exp(v)[0])(z))).all()
    )

  def test_retract_local_and_sign(self):
    m = Quaternion()
    q = quat_exp(jnp.asarray([0.3, -0.2, 0.1]))
    v = jnp.asarray([0.1, 0.04, -0.03])
    r = jax.jit(m.retract)(q, v)
    np.testing.assert_allclose(m.local(q, r), v, atol=1e-13)
    np.testing.assert_allclose(m.local(q, -r), v, atol=1e-13)
    self.assertAlmostEqual(float(jnp.linalg.norm(r)), 1.0, places=14)

  def test_product_maps_and_dimensions(self):
    m = ProductManifold(Euclidean(2), Quaternion(), Euclidean(1))
    self.assertEqual((m.storage_dim, m.tangent_dim), (7, 6))
    np.testing.assert_array_equal(m.native_map[:2, :2], np.eye(2))
    np.testing.assert_array_equal(m.native_map[2:6], 0.0)
    x = m.identity(jnp.float64)
    v = jnp.arange(6, dtype=jnp.float64) * 0.02
    np.testing.assert_allclose(m.local(x, m.retract(x, v)), v, atol=1e-14)
    p = quaternion_product(7, [2])
    np.testing.assert_array_equal(p.native_map, m.native_map)

  def test_invalid_seed_and_cut_locus(self):
    m = Quaternion()
    self.assertFalse(bool(m.valid(jnp.zeros(4))))
    self.assertFalse(
        bool(
            m.chart_valid(
                jnp.asarray([1.0, 0, 0, 0]), jnp.asarray([0.0, 1.0, 0, 0])
            )
        )
    )
    self.assertTrue(
        bool(
            m.chart_valid(
                quat_exp(jnp.zeros(3)), quat_exp(jnp.asarray([0.3, 0.0, 0.0]))
            )
        )
    )


if __name__ == '__main__':
  unittest.main()
