from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import numpy as np

from mujodico.mjx._src.solver_sap import (
  _contact_normal_impulse,
  _contact_normal_impulse_grad,
  _contact_normal_impulse_integral,
)


_TOLERANCE = 5e-5


class ContactTest(parameterized.TestCase):

  def test_contact_normal_impulse_grad(self):
    phi0 = 0.0
    k = 1e5
    d = 50.0
    dt = 0.01
    epsilon1 = 0.01
    epsilon2 = 0.1

    vn = jp.linspace(-1.0, 1.0, num=1000)

    actual = _contact_normal_impulse_grad(vn, phi0, k, d, dt, epsilon1, epsilon2)

    desired = jax.vmap(jax.grad(
      lambda vn: _contact_normal_impulse(vn, phi0, k, d, dt, epsilon1, epsilon2)
    ))(vn)

    np.testing.assert_allclose(actual, desired, rtol=_TOLERANCE)

  def test_contact_normal_impulse_integral(self):
    phi0 = 0.0
    k = 1e5
    d = 50.0
    dt = 0.01
    epsilon1 = 0.01
    epsilon2 = 0.1

    vn = jp.linspace(-1.0, 1.0, num=1000)

    desired = jax.vmap(jax.grad(
      lambda vn: _contact_normal_impulse_integral(vn, phi0, k, d, dt, epsilon1, epsilon2)
    ))(vn)

    actual = _contact_normal_impulse(vn, phi0, k, d, dt, epsilon1, epsilon2)

    np.testing.assert_allclose(actual, desired, rtol=_TOLERANCE)


if __name__ == "__main__":
  absltest.main()
