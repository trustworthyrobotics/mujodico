from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import numpy as np

from mujodico.mjx._src.solver_sap_util import (
  smooth_relu,
  smooth_relu_grad,
  soft_normalize,
  soft_normalize_grad,
)
from mujodico.mjx._src.solver_sap import (
  SapNormalContactConstraint,
  SapContactLaggedConstraint,
)


_TOLERANCE = 1e-5


def _assert_eq(a, b):
  np.testing.assert_allclose(a, b, atol=_TOLERANCE, rtol=_TOLERANCE)


class ContactTest(parameterized.TestCase):

  def test_smooth_relu(self):
    x = jp.linspace(-1.0, 1.0, num=1000)
    epsilon = 1e-3

    actual = smooth_relu_grad(x, epsilon)

    expected = jax.vmap(jax.grad(lambda x: smooth_relu(x, epsilon)))(x)
    _assert_eq(actual, expected)

  def test_soft_normalize(self):
    r = jp.linspace(0.0, 1.0, num=100)
    a = jp.linspace(0.0, 2*jp.pi, num=10)
    r, a = jp.meshgrid(r, a)
    x = jp.stack((
      (r * jp.cos(a)).ravel(),
      (r * jp.sin(a)).ravel(),
    ), axis=1)

    epsilon = 0.01

    actual = jax.vmap(lambda x: soft_normalize_grad(x, epsilon))(x)

    expected = jax.vmap(jax.jacobian(lambda x: soft_normalize(x, epsilon)))(x)
    _assert_eq(actual, expected)

  def test_sap_normal_contact_constraint(self):
    N = 1000

    sap_constraint = SapNormalContactConstraint(
      phi0=jp.full(N, 0.0),
      stiffness=jp.full(N, 1e5),
      hunt_crossley_damping=jp.full(N, 50.0),
      dt=jp.array(0.01),
    )

    vn = jp.linspace(-1.0, 1.0, N)

    actual = jp.diag(sap_constraint.calc_impulse_grad(vn))

    expected = jax.jacobian(sap_constraint.calc_impulse)(vn)
    _assert_eq(actual, expected)

  def test_sap_contact_lagged_constraint(self):
    sap_constraint = SapContactLaggedConstraint(
      phi0=jp.array(-0.001),
      vn0=jp.array(0.0),
      stiffness=jp.array(1e5),
      hunt_crossley_damping=jp.array(50.0),
      friction=jp.array([0.2, 0.2]),
      dt=jp.array(0.01),
    )

    vc = jp.linspace(-0.1, 0.1, num=11)
    vn, vt1, vt2 = jp.meshgrid(vc, vc, vc)
    vc = jp.stack((
      vn.ravel(),
      vt1.ravel(),
      vt2.ravel(),
    ), axis=1)

    actual = jax.vmap(sap_constraint.calc_impulse_grad)(vc)

    expected = jax.vmap(jax.jacobian(sap_constraint.calc_impulse))(vc)
    _assert_eq(actual, expected)


if __name__ == "__main__":
  absltest.main()
