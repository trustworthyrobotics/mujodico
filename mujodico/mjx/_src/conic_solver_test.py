from absl.testing import absltest
from absl.testing import parameterized
from functools import partial
import jax
from jax import numpy as jp
from jax._src import test_util
from mujodico.mjx._src import conic_solver as conic
import numpy as np


# tolerance for KKT optimality
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  err_msg = f"Failed: {name}"
  np.testing.assert_allclose(
      a, b, err_msg=err_msg, atol=_TOLERANCE, rtol=_TOLERANCE
  )


def _assert_zero(a, name):
  _assert_eq(a, np.zeros(a.size), name)


def _check_jacobians(P, q, A, b, cones, comps, jac_fun):
  check_jac_fun = (
      jax._src.test_util.check_jvp
      if jac_fun == jax.jvp
      else jax._src.test_util.check_vjp
  )

  def f1(P, q, A, b):
    return conic.solve(P, q, A, b, cones, comps)

  def f2(comps):
    return conic.solve(P, q, A, b, cones, comps)

  check_jac_fun(f1, partial(jac_fun, f1), (P, q, A, b), eps=1e-3)

  # only check jacobian with respect to μ when μ ≠ 0
  if comps is not None and jp.all(comps > 0):
    check_jac_fun(f2, partial(jac_fun, f2), (comps,), eps=1e-3)


def _check_jvp(P, q, A, b, cones, comps=None):
  _check_jacobians(P, q, A, b, cones, comps, jax.jvp)


def _check_vjp(P, q, A, b, cones, comps=None):
  _check_jacobians(P, q, A, b, cones, comps, jax.vjp)


class ConicSolverTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    _float = jp.float32

    self.P = jp.array([[3, 1, -1], [1, 4, 2], [-1, 2, 5]], dtype=_float)
    self.q = jp.array([1, 2, -3], dtype=_float)

    self.Aeq = jp.array([[1, 1, -1]], dtype=_float)
    self.beq = jp.array([1], dtype=_float)

    self.Aineq = jp.array([[0, 1, 0], [0, 0, 1]], dtype=_float)
    self.bineq = jp.array([2, 2], dtype=_float)

    self.Asoc = -jp.identity(3, dtype=_float)
    self.bsoc = jp.array([0, 0, 0], dtype=_float)

    self.solve = jax.jit(conic.solve, static_argnames="cones")

  def test_equality_constrained_qp(self):
    P = self.P
    q = self.q
    A = self.Aeq
    b = self.beq
    cones = (conic.ZeroCone(1),)

    x, z, s = self.solve(P=P, q=q, A=A, b=b, cones=cones)

    _assert_zero(P @ x + q + A.T @ z, "stationarity")
    _assert_zero(A @ x - b, "primal feasibility")
    _assert_zero(s, "zero slack")

    _check_jvp(P, q, A, b, cones)
    _check_vjp(P, q, A, b, cones)

  @parameterized.parameters((0.0, 0.0), (0.0, 0.1), (0.1, 0.1))
  def test_inequality_constrained_qp(self, mu1, mu2):
    P = self.P
    q = self.q
    A = self.Aineq
    b = self.bineq
    cones = (conic.NonNegativeCone(1), conic.NonNegativeCone(1))
    comps = jp.array([mu1, mu2])

    x, z, s = self.solve(
        P=P, q=q, A=A, b=b, cones=cones, complementarities=comps
    )

    _assert_zero(P @ x + q + A.T @ z, "stationarity")
    _assert_zero(A @ x + s - b, "primal feasibility")
    self.assertTrue(np.all(s >= 0), "primal feasibility")
    self.assertTrue(np.all(z >= 0), "dual feasibility")
    _assert_eq(s * z, jp.array([mu1, mu2]), "complementarity")

    _check_jvp(P, q, A, b, cones, comps)
    _check_vjp(P, q, A, b, cones, comps)

  @parameterized.parameters((0.0, 0.0), (0.0, 0.1), (0.1, 0.1))
  def test_inequality_equality_constrained_qp(self, mu1, mu2):
    P = self.P
    q = self.q
    A = jp.vstack((self.Aineq, self.Aeq))
    b = jp.concatenate((self.bineq, self.beq))
    cones = (
        conic.NonNegativeCone(1),
        conic.NonNegativeCone(1),
        conic.ZeroCone(1),
    )
    comps = jp.array([mu1, mu2, 0])

    x, z, s = self.solve(
        P=P, q=q, A=A, b=b, cones=cones, complementarities=comps
    )

    _assert_zero(P @ x + q + A.T @ z, "stationarity")
    _assert_zero(A @ x + s - b, "primal feasibility")
    _assert_zero(s[2], "primal feasibility")
    self.assertTrue(np.all(s[:2] >= 0), "primal feasibility")
    self.assertTrue(np.all(z[:2] >= 0), "dual feasibility")
    _assert_eq((s * z)[:2], jp.array([mu1, mu2]), "complementarity")

    _check_jvp(P, q, A, b, cones, comps)
    _check_vjp(P, q, A, b, cones, comps)


if __name__ == "__main__":
  absltest.main()
