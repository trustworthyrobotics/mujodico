"""Interior pushing and multiplier-initialization regression tests."""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from mujodico.trajopt._src.fatrop_ref import Problem, InitialGuess, Options
from mujodico.trajopt._src.fatrop_ref import iterate as bounds
from mujodico.trajopt._src.fatrop_ref.initialization import initialize, metrics

jax.config.update('jax_enable_x64', True)


def problem(**kwargs):
  return Problem(
      horizon=1,
      num_states=1,
      num_controls=1,
      initial_state=lambda p: jnp.array([0.0]),
      dynamics=lambda x, u, p: x + u,
      stage_cost=lambda x, u, p: jnp.sum(u * u),
      terminal_cost=lambda x, p: jnp.sum((x - 2.0) ** 2),
      **kwargs,
  )


class InitializationTest(unittest.TestCase):

  def test_pushes_both_bound_sides_and_slacks(self):
    p = problem(
        u_bounds=([-1.0], [1.0]),
        stage_inequality=lambda x, u, q: u,
        stage_inequality_bounds=([-2.0], [-1.0]),
    )
    guess = InitialGuess(jnp.zeros((2, 1)), jnp.array([[10.0]]))
    it, lo, hi, _, _ = jax.jit(
        lambda g: initialize(p, None, g, Options(initialize_dual=False))
    )(guess)
    bd = bounds.data(it.primal, lo, hi)
    self.assertGreater(float(it.primal.us[0, 0]), -1.0)
    self.assertLess(float(it.primal.us[0, 0]), 1.0)
    self.assertGreater(float(it.primal.s[0, 0]), -2.0)
    self.assertLess(float(it.primal.s[0, 0]), -1.0)
    self.assertTrue(bool(bounds.barrier(bd, 0.1)[1]))
    np.testing.assert_allclose(it.zl.us, 1.0)
    np.testing.assert_allclose(it.zu.us, 1.0)

  def test_multiplier_least_squares_and_limit_fallback(self):
    p = problem()
    guess = InitialGuess(jnp.zeros((2, 1)), jnp.zeros((1, 1)))
    it, *_ = initialize(p, None, guess, Options())
    np.testing.assert_allclose(it.eq, [[-2.0]], atol=1e-10)
    limited, *_ = initialize(p, None, guess, Options(lam_max=0.1))
    np.testing.assert_allclose(limited.eq, 0.0)

  def test_preserve_primal_for_restoration_dual_reset(self):
    p = problem(u_bounds=(0.0, 1.0))
    seed = p.initialize(None, us=jnp.array([[1e-10]]))
    it, *_ = initialize(p, None, seed, Options(), preserve_primal=True)
    np.testing.assert_array_equal(it.primal.us, seed.us)

  def test_missing_sides_are_zero_and_finite(self):
    p = problem(u_bounds=(0.0, jnp.inf))
    it, lo, hi, ol, ou = initialize(p, None, None, Options())
    bd = bounds.data(it.primal, lo, hi)
    theta, force = bounds.condense(bd, it.zl, it.zu, 0.1)
    self.assertTrue(bool(jnp.isfinite(bounds.tree_max(theta))))
    self.assertTrue(bool(jnp.isfinite(bounds.tree_max(force))))
    np.testing.assert_allclose(it.zu.us, 0.0)
    np.testing.assert_allclose(it.zl.xs, 0.0)
    m = metrics(p, it, None, 0.1, lo, hi, ol, ou, Options())
    self.assertTrue(bool(m.finite))


if __name__ == '__main__':
  unittest.main()
