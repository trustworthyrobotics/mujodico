from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
import numpy as np
from scipy.spatial.transform import Rotation

from mujodico.trajopt._src import manifold


def _skew(v):
  x, y, z = v
  return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _rotation_jacobians(v):
  """Analytical SO(3) left/right Jacobians and their inverses."""
  theta2 = np.dot(v, v)
  if theta2 < 1e-6:
    a = 0.5 - theta2 / 24 + theta2**2 / 720
    b = 1 / 6 - theta2 / 120 + theta2**2 / 5040
    c = 1 / 12 + theta2 / 720 + theta2**2 / 30240
  else:
    theta = np.sqrt(theta2)
    a = (1 - np.cos(theta)) / theta2
    b = (theta - np.sin(theta)) / theta**3
    c = (1 - 0.5 * theta / np.tan(0.5 * theta)) / theta2
  skew = _skew(v)
  identity = np.eye(3)
  return (
      identity + a * skew + b * skew @ skew,
      identity - a * skew + b * skew @ skew,
      identity - 0.5 * skew + c * skew @ skew,
      identity + 0.5 * skew + c * skew @ skew,
  )


class QuaternionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(jax.enable_x64())

  def test_multiply_values_and_order(self):
    identity = jp.array([1.0, 0, 0, 0])
    i, j, k = jp.eye(4)[1:]
    np.testing.assert_array_equal(manifold.quat_multiply(i, j), k)
    np.testing.assert_array_equal(manifold.quat_multiply(j, i), -k)
    q = jp.array([0.3, -0.2, 0.5, 0.7])
    np.testing.assert_array_equal(manifold.quat_multiply(q, identity), q)
    np.testing.assert_array_equal(manifold.quat_multiply(identity, q), q)
    a = Rotation.from_rotvec([0.3, -0.2, 0.5])
    b = Rotation.from_rotvec([-0.1, 0.4, 0.2])
    np.testing.assert_allclose(
        manifold.quat_multiply(
            np.roll(a.as_quat(), 1), np.roll(b.as_quat(), 1)
        ),
        np.roll((a * b).as_quat(), 1),
        atol=1e-15,
    )

  def test_multiply_jacobians(self):
    q, r = jp.array([0.3, -0.2, 0.5, 0.7]), jp.array([0.8, 0.1, -0.4, 0.2])
    right = np.block([
        [np.array([[r[0]]]), -np.asarray(r[1:])[None]],
        [np.asarray(r[1:])[:, None], r[0] * np.eye(3) - _skew(r[1:])],
    ])
    left = np.block([
        [np.array([[q[0]]]), -np.asarray(q[1:])[None]],
        [np.asarray(q[1:])[:, None], q[0] * np.eye(3) + _skew(q[1:])],
    ])
    for differentiate in (jax.jacfwd, jax.jacrev):
      dq, dr = differentiate(manifold.quat_multiply, argnums=(0, 1))(q, r)
      np.testing.assert_allclose(dq, right, atol=1e-15)
      np.testing.assert_allclose(dr, left, atol=1e-15)

  def test_inverse_values_and_jacobian(self):
    q = jp.array([0.3, -0.2, 0.5, 0.7])
    inverse = manifold.quat_inverse(q)
    for product in (
        manifold.quat_multiply(q, inverse),
        manifold.quat_multiply(inverse, q),
    ):
      np.testing.assert_allclose(product, [1, 0, 0, 0], atol=1e-15)
    conjugation = np.diag([1, -1, -1, -1])
    norm2 = np.dot(q, q)
    expected = conjugation / norm2 - 2 * np.outer(conjugation @ q, q) / norm2**2
    for differentiate in (jax.jacfwd, jax.jacrev):
      np.testing.assert_allclose(
          differentiate(manifold.quat_inverse)(q), expected
      )

  @parameterized.parameters(
      [0.0, 0.0, 0.0],
      [1e-12, -2e-12, 3e-12],
      [0.3, -0.5, 0.7],
      [np.pi - 1e-6, 0.0, 0.0],
      [3.8, 0.1, 0.0],
  )
  def test_exp_log_values(self, vx, vy, vz):
    v = np.array([vx, vy, vz])
    rotation = Rotation.from_rotvec(v)
    expected = np.roll(rotation.as_quat(), 1)
    q = manifold.quat_exp(v)
    np.testing.assert_allclose(q, expected, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-15)
    for scale in (1.0, -1.0, 2.5, -2.5):
      np.testing.assert_allclose(
          manifold.quat_log(scale * q), rotation.as_rotvec(), atol=1e-14
      )
    roundtrip = manifold.quat_exp(manifold.quat_log(q))
    np.testing.assert_allclose(
        roundtrip, expected if expected[0] >= 0 else -expected, atol=1e-14
    )

  def test_identity_jacobians(self):
    v = jp.zeros(3)
    identity = jp.array([1.0, 0.0, 0.0, 0.0])
    for differentiate in (jax.jacfwd, jax.jacrev):
      np.testing.assert_allclose(
          jax.jit(differentiate(manifold.quat_exp))(v),
          np.vstack([np.zeros((1, 3)), 0.5 * np.eye(3)]),
      )
      for sign in (1, -1):
        np.testing.assert_allclose(
            jax.jit(differentiate(manifold.quat_log))(sign * identity),
            sign * np.column_stack([np.zeros(3), 2 * np.eye(3)]),
        )
      np.testing.assert_allclose(
          differentiate(manifold.quat_inverse)(identity), -np.eye(4)
      )
    # Exact solver Hessians require the zero-angle extensions to be smooth too.
    np.testing.assert_allclose(
        jax.hessian(lambda v: manifold.quat_exp(v)[0])(v), -0.25 * np.eye(3)
    )
    self.assertTrue(
        np.all(np.isfinite(jax.jacfwd(jax.jacrev(manifold.quat_log))(identity)))
    )

  @parameterized.parameters(
      [0.0, 0.0, 0.0],
      [1e-12, -2e-12, 3e-12],
      [1e-7, -2e-7, 3e-7],
      [0.3, -0.5, 0.7],
      [np.pi - 1e-6, 0.0, 0.0],
      (np.pi - 1e-6) * np.array([1.0, -2.0, 3.0]) / np.sqrt(14),
  )
  def test_exp_log_analytical_left_right_jacobians(self, vx, vy, vz):
    v = np.array([vx, vy, vz])
    left, right, left_inverse, right_inverse = _rotation_jacobians(v)
    q = np.roll(Rotation.from_rotvec(v).as_quat(), 1)
    q_inverse = q * np.array([1, -1, -1, -1])
    zero = jp.zeros(3)
    # Exp(v + dv) = Exp(J_left dv) * Exp(v)
    #             = Exp(v) * Exp(J_right dv), to first order.
    left_tangent = 0.5 * np.vstack([-q[1:], q[0] * np.eye(3) - _skew(q[1:])])
    right_tangent = 0.5 * np.vstack([-q[1:], q[0] * np.eye(3) + _skew(q[1:])])
    for differentiate in (jax.jacfwd, jax.jacrev):
      exp_jacobian = differentiate(manifold.quat_exp)(jp.asarray(v))
      np.testing.assert_allclose(exp_jacobian, left_tangent @ left, atol=1e-13)
      np.testing.assert_allclose(
          exp_jacobian, right_tangent @ right, atol=1e-13
      )
      for fn, expected in (
          (
              lambda dv: manifold.quat_log(
                  manifold.quat_multiply(manifold.quat_exp(v + dv), q_inverse)
              ),
              left,
          ),
          (
              lambda dv: manifold.quat_log(
                  manifold.quat_multiply(q_inverse, manifold.quat_exp(v + dv))
              ),
              right,
          ),
      ):
        np.testing.assert_allclose(
            differentiate(fn)(zero), expected, atol=1e-13
        )
      # Log(Exp(dv) * q) and Log(q * Exp(dv)) have inverse Jacobians.
      for sign in (1, -1):
        for fn, expected in (
            (
                lambda dv: manifold.quat_log(
                    manifold.quat_multiply(manifold.quat_exp(dv), sign * q)
                ),
                left_inverse,
            ),
            (
                lambda dv: manifold.quat_log(
                    manifold.quat_multiply(sign * q, manifold.quat_exp(dv))
                ),
                right_inverse,
            ),
        ):
          np.testing.assert_allclose(
              differentiate(fn)(zero), expected, atol=1e-13
          )


if __name__ == '__main__':
  absltest.main()
