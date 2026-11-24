from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
from mujodico.mjx._src import small_la
import numpy as np


def _library_solve(a, b):
  return jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(a, lower=True), b)


class SmallLinearAlgebraTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(jax.enable_x64())

  @parameterized.product(n=(0, 4, 16, 17), dtype=(np.float32, np.float64))
  def test_cholesky_and_products(self, n, dtype):
    rng = np.random.default_rng(0)
    r = rng.standard_normal((n, n))
    a = jp.asarray(r @ r.T + np.eye(n), dtype=dtype)
    rhs = jp.asarray(rng.standard_normal((n, 3)), dtype=dtype)

    @jax.jit
    def evaluate(a, rhs):
      factor = small_la.cho_factor(a)
      vector = rhs[:, 0]
      return (
          small_la.cho_solve(factor, vector), small_la.cho_solve(factor, rhs),
          small_la.mv(a, vector), small_la.mtv(a, vector), small_la.mm(a, rhs),
      )

    actual = evaluate(a, rhs)
    expected = (_library_solve(a, rhs[:, 0]), _library_solve(a, rhs),
                a @ rhs[:, 0], a.T @ rhs[:, 0], a @ rhs)
    tol = 1e-4 if dtype == np.float32 else 1e-10
    for value, reference in zip(actual, expected):
      np.testing.assert_allclose(value, reference, atol=tol, rtol=tol)

  @parameterized.product(n=(4, 17), matrix_rhs=(False, True))
  def test_spd_solve_derivatives(self, n, matrix_rhs):
    rng = np.random.default_rng(0)
    r = rng.standard_normal((n, n))
    a = jp.asarray(r @ r.T + np.eye(n))
    b = jp.asarray(rng.standard_normal((n, 3) if matrix_rhs else (n,)))
    da = jp.asarray(rng.standard_normal(a.shape))
    db = jp.asarray(rng.standard_normal(b.shape))
    cotangent = jp.asarray(rng.standard_normal(b.shape))

    def evaluate(solver):
      def solve(a, b):
        return solver((a + a.T) * 0.5, b)

      primal, tangent = jax.jvp(solve, (a, b), (da, db))
      _, pullback = jax.vjp(solve, a, b)
      return primal, tangent, *pullback(cotangent)

    for actual, expected in zip(evaluate(small_la.spd_solve), evaluate(_library_solve)):
      np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
  absltest.main()
