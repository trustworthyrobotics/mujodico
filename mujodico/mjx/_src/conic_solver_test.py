from functools import partial
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
from jax._src.test_util import check_vjp
from  mujodico.mjx._src import conic_solver
import numpy as np


# tolerance for KKT optimality
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  err_msg = f"Failed: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=_TOLERANCE, rtol=_TOLERANCE)


def _assert_zero(a, name):
  _assert_eq(a, np.zeros(a.size), name)


def _check_solve_vjp(P, q, A, b, cones):
  tol = _TOLERANCE * 10  # avoid test noise

  def solve(P, q, A, b):
    return conic_solver.solve(P, q, A, b, cones)

  solve_vjp = partial(jax.vjp, solve)

  jax._src.test_util.check_vjp(solve, solve_vjp, (P, q, A, b),
                               eps=1e-3, atol=tol, rtol=tol)


class ConicSolverTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    _float = jp.float32

    self.P = jp.array([[ 3, 1, -1],
                       [ 1, 4,  2],
                       [-1, 2,  5]], dtype=_float)
    self.q = jp.array([ 1,  2, -3], dtype=_float)

    self.Aeq = jp.array([[ 1, 1, -1]], dtype=_float)
    self.beq = jp.array([1], dtype=_float)

    self.Aineq = jp.array([[0, 1, 0],
                           [0, 0, 1]], dtype=_float)
    self.bineq = jp.array([2, 2], dtype=_float)

    self.Asoc = -jp.identity(3, dtype=_float)
    self.bsoc = jp.array([0, 0, 0], dtype=_float)

    self.solve = jax.jit(conic_solver.solve, static_argnames='cones')

  def test_equality_constrained_qp(self):
    P = self.P
    q = self.q
    A = self.Aeq
    b = self.beq
    cones = (conic_solver.ZeroCone(1),)

    x, z, s = self.solve(P=P, q=q, A=A, b=b, cones=cones)

    _assert_zero(P @ x + q + A.T @ z, "stationarity")
    _assert_zero(A @ x - b, "primal feasibility")
    _assert_zero(s, "zero slack")

    _check_solve_vjp(P, q, A, b, cones)

  @parameterized.parameters((0.0, 0.0), (0.1, 0.1))
  def test_inequality_constrained_qp(self, mu1, mu2):
    P = self.P
    q = self.q
    A = self.Aineq
    b = self.bineq
    cones = (
      conic_solver.NonNegativeCone(1, mu1),
      conic_solver.NonNegativeCone(1, mu2),
    )

    x, z, s = self.solve(P=P, q=q, A=A, b=b, cones=cones)

    _assert_zero(P @ x + q + A.T @ z, "stationarity")
    _assert_zero(A @ x + s - b, "primal feasibility")
    self.assertTrue(np.all(s >= 0), "primal feasibility")
    self.assertTrue(np.all(z >= 0), "dual feasibility")
    _assert_eq(s * z, jp.array([mu1, mu2]), "complementarity")

    _check_solve_vjp(P, q, A, b, cones)

  @parameterized.parameters((0.0, 0.0), (0.1, 0.1))
  def test_equality_inequality_constrained_qp(self, mu1, mu2):
    P = self.P
    q = self.q
    A = jp.vstack((self.Aeq, self.Aineq))
    b = jp.concatenate((self.beq, self.bineq))
    cones = (
      conic_solver.ZeroCone(1),
      conic_solver.NonNegativeCone(1, mu1),
      conic_solver.NonNegativeCone(1, mu2),
    )

    x, z, s = self.solve(P=P, q=q, A=A, b=b, cones=cones)

    _assert_zero(P @ x + q + A.T @ z, "stationarity")
    _assert_zero(A @ x + s - b, "primal feasibility")
    _assert_zero(s[0], "primal feasibility")
    self.assertTrue(np.all(s >= 0), "primal feasibility")
    self.assertTrue(np.all(z[1:] >= 0), "dual feasibility")
    _assert_eq((s * z)[1:], jp.array([mu1, mu2]), "complementarity")

    _check_solve_vjp(P, q, A, b, cones)


if __name__ == '__main__':
  absltest.main()
