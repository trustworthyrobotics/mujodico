from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import numpy as np

from mujodico.mjx._src.solver_sap_util import (
  smooth_relu,
  smooth_relu_grad,
)
from mujodico.mjx._src.solver_sap import (
  SapNormalContactConstraint,
)


_TOLERANCE = 1e-6


class ContactTest(parameterized.TestCase):

  def test_smooth_relu(self):
    x = jp.linspace(-1.0, 1.0, num=1000)
    epsilon = 1e-3

    actual = smooth_relu_grad(x, epsilon)

    desired = jax.vmap(jax.grad(lambda x: smooth_relu(x, epsilon)))(x)

    np.testing.assert_allclose(actual, desired, rtol=_TOLERANCE)

  def test_sap_normal_contact_constraint(self):
    N = 1000

    sap_constraint = SapNormalContactConstraint(
      phi0=jp.full(N, 0.0),
      dt=jp.array(0.01),
      stiffness=jp.full(N, 1e5),
      hunt_crossley_damping=jp.full(N, 50.0),
    )

    vn = jp.linspace(-1.0, 1.0, N)

    actual = jp.diag(sap_constraint.calc_impulse_grad(vn))

    desired = jax.jacobian(sap_constraint.calc_impulse)(vn)

    np.testing.assert_allclose(actual, desired, rtol=_TOLERANCE)


if __name__ == "__main__":
  absltest.main()
