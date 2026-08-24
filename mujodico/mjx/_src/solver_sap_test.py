from absl.testing import absltest
from absl.testing import parameterized
from functools import partial
from itertools import product
import jax
from jax import numpy as jp
from jax._src import test_util
import mujoco
from mujodico import mjx
from mujodico.mjx._src import solver_sap_util
from mujodico.mjx._src import solver_sap
import numpy as np

jax.config.update('jax_enable_x64', True)


_TOLERANCE = 1e-5


def _assert_eq(a, b):
  np.testing.assert_allclose(a, b, atol=_TOLERANCE, rtol=_TOLERANCE)


class SolverSapTest(parameterized.TestCase):

  def test_soft_normalize(self):
    r = jp.linspace(0.0, 1.0, num=100)
    a = jp.linspace(0.0, 2*jp.pi, num=10)
    r, a = jp.meshgrid(r, a)
    x = jp.stack((
      (r * jp.cos(a)).ravel(),
      (r * jp.sin(a)).ravel(),
    ), axis=1)

    epsilon = 0.01

    grad_actual = jax.vmap(lambda x: solver_sap_util.soft_normalize_grad(x, epsilon))(x)

    grad_expected = jax.vmap(
      jax.jacobian(lambda x: solver_sap_util.soft_normalize(x, epsilon))
    )(x)
    _assert_eq(grad_actual, grad_expected)

  @parameterized.parameters([
    'softplus',
    'algebraic_smooth_relu',
    'local_smooth_relu',
    'reciprocal_approx_relu',
  ])
  def test_smooth_relu(self, name):
    val_fun = eval(f'solver_sap_util.{name}')
    grad_fun = eval(f'solver_sap_util.{name}_grad')

    x = jp.linspace(-1.0, 1.0, num=1000)
    epsilon = 1e-3

    grad_actual = grad_fun(x, epsilon)

    grad_expected = jax.vmap(jax.grad(lambda x: val_fun(x, epsilon)))(x)
    _assert_eq(grad_actual, grad_expected)

  def test_sap_fixed_constraint(self):
    N = 1000

    sap_constraint = solver_sap.SapFixedConstraint(
      phi0=jp.zeros(N),
      delassus_estimation=jp.ones(N),
      dt=jp.array(0.01),
    )

    vc = jp.linspace(-1.0, 1.0, N)

    grad_grad_actual = jp.diag(sap_constraint.calc_impulse_grad(vc))

    grad_grad_expected = jax.jacobian(sap_constraint.calc_impulse)(vc)
    _assert_eq(grad_grad_actual, grad_grad_expected)

  def test_sap_limit_constraint(self):
    N = 1000

    sap_constraint = solver_sap.SapLimitConstraint(
      phi0=jp.full(N, 0.001),
      delassus_estimation=jp.ones(N),
      dt=jp.array(0.01),
    )

    vc = jp.linspace(-1.0, 1.0, N)

    grad_grad_actual = jp.diag(sap_constraint.calc_impulse_grad(vc))

    grad_grad_expected = jax.jacobian(sap_constraint.calc_impulse)(vc)
    _assert_eq(grad_grad_actual, grad_grad_expected)

  def test_sap_dry_friction_constraint(self):
    N = 1000

    sap_constraint = solver_sap.SapDryFrictionConstraint(
      dryfriction=jp.ones(N),
      dt=jp.array(0.01),
    )

    vc = jp.linspace(-1.0, 1.0, N)

    grad_actual = jp.diag(sap_constraint.calc_impulse_grad(vc))

    grad_expected = jax.jacobian(sap_constraint.calc_impulse)(vc)
    _assert_eq(grad_actual, grad_expected)

  def test_sap_logbarrier_normal_contact_constraint(self):
    N = 1000

    sap_constraint = solver_sap.SapLogbarrierNormalContactConstraint(
      phi0=jp.full(N, 0.0),
      tau=jp.zeros(N),
      dt=jp.array(0.01),
      contact_smoothing=jp.array(1e-1),
    )

    vn = jp.linspace(-1.0, 1.0, N)

    grad_actual = jp.diag(sap_constraint.calc_impulse_grad(vn))

    grad_expected = jax.jacobian(sap_constraint.calc_impulse)(vn)
    _assert_eq(grad_actual, grad_expected)

  def test_sap_kelvin_voigt_normal_contact_constraint(self):
    N = 1000

    sap_constraint = solver_sap.SapKelvinVoigtNormalContactConstraint(
      phi0=jp.full(N, 0.0),
      stiffness=jp.full(N, 1e5),
      dissipation_time_scale=jp.full(N, 0.02),
      dt=jp.array(0.01),
      contact_smoothing=jp.array(1e-4),
    )

    vn = jp.linspace(-1.0, 1.0, N)

    grad_actual = jp.diag(sap_constraint.calc_impulse_grad(vn))

    grad_expected = jax.jacobian(sap_constraint.calc_impulse)(vn)
    _assert_eq(grad_actual, grad_expected)

  def test_sap_hunt_crossley_normal_contact_constraint(self):
    N = 1000

    sap_constraint = solver_sap.SapHuntCrossleyNormalContactConstraint(
      phi0=jp.full(N, 0.0),
      stiffness=jp.full(N, 1e5),
      hunt_crossley_damping=jp.full(N, 50.0),
      dt=jp.array(0.01),
      contact_smoothing=jp.array(1e-4),
    )

    vn = jp.linspace(-1.0, 1.0, N)

    grad_actual = jp.diag(sap_constraint.calc_impulse_grad(vn))

    grad_expected = jax.jacobian(sap_constraint.calc_impulse)(vn)
    _assert_eq(grad_actual, grad_expected)

  def test_sap_kelvin_voigt_frictional_contact_constraint(self):
    sap_constraint = solver_sap.SapKelvinVoigtFrictionalContactConstraint(
      phi0=jp.array(-0.001),
      vn0=jp.array(0.0),
      stiffness=jp.array(1e5),
      dissipation_time_scale=jp.array(0.02),
      friction=jp.array([0.2, 0.2]),
      dt=jp.array(0.01),
      contact_smoothing=jp.array(1e-4),
    )

    vc = jp.linspace(-1.0, 1.0, num=11)
    vn, vt1, vt2 = jp.meshgrid(vc, vc, vc)
    vc = jp.stack((
      vn.ravel(),
      vt1.ravel(),
      vt2.ravel(),
    ), axis=1)

    grad_actual = jax.vmap(sap_constraint.calc_impulse_grad)(vc)

    grad_expected = jax.vmap(jax.jacobian(sap_constraint.calc_impulse))(vc)
    _assert_eq(grad_actual, grad_expected)

  def test_sap_hunt_crossley_frictional_contact_constraint(self):
    sap_constraint = solver_sap.SapHuntCrossleyFrictionalContactConstraint(
      phi0=jp.array(-0.001),
      vn0=jp.array(0.0),
      stiffness=jp.array(1e5),
      hunt_crossley_damping=jp.array(50.0),
      friction=jp.array([0.2, 0.2]),
      dt=jp.array(0.01),
      contact_smoothing=jp.array(1e-4),
    )

    vc = jp.linspace(-1.0, 1.0, num=11)
    vn, vt1, vt2 = jp.meshgrid(vc, vc, vc)
    vc = jp.stack((
      vn.ravel(),
      vt1.ravel(),
      vt2.ravel(),
    ), axis=1)

    grad_actual = jax.vmap(sap_constraint.calc_impulse_grad)(vc)

    grad_expected = jax.vmap(jax.jacobian(sap_constraint.calc_impulse))(vc)
    _assert_eq(grad_actual, grad_expected)

  @parameterized.parameters(
      *product(
        ['Kelvin-Voigt', 'Hunt-Crossley'],
        ['pyramidal', 'elliptic'],
        [1, 3, 4, 6],
      )
  )
  def test_solve(self, contact_approximation, cone, condim):
    # Elliptic cone does not support condim=1.
    if cone == 'elliptic' and condim == 1:
      return

    xml = f'''
<mujoco model="1d_pusher">

    <option cone="{cone}" />
    <option integrator="implicitfast" />
    <option timestep="0.02"  />

    <default>
        <joint damping="5"/>
        <geom condim="{condim}" />
    </default>

    <worldbody>

        <!-- Red sphere pusher -->
        <body name="pusher" pos="-0.1 0 0">
            <joint name="pusher_slide"
                   type="slide"
                   axis="1 0 0" />

            <geom name="pusher_geom"
                  type="sphere"
                  size="0.05"
                  mass="1"
                  rgba="1 0 0 1" />
        </body>

        <!-- Blue capsule object -->
        <body name="object" pos="0.0 0 0">
            <joint name="object_slide"
                   type="slide"
                   axis="1 0 0" />

            <geom name="object_geom"
                  type="sphere"
                  size="0.05"
                  mass="1"
                  rgba="0 0 1 1" />
        </body>

    </worldbody>

    <actuator>
        <velocity name="pusher_vel"
                  joint="pusher_slide"
                  kv="100" />
    </actuator>

</mujoco>
'''

    model = mujoco.MjModel.from_xml_string(xml)

    mjx_model = mjx.put_model(
      model,
      device=jax.devices()[0],
      contact_approximation=contact_approximation,
      contact_smoothing=1e-4,
    )
    mjx_data = mjx.make_data(mjx_model, impl=mjx_model.impl)

    @jax.jit
    def step(x, u):
      m = mjx_model
      d = mjx_data.replace(
        qpos=x[:m.nq],
        qvel=x[m.nq:m.nq+m.nv],
        act=x[m.nq+m.nv:],
        ctrl=u,
      )
      d = mjx.step(m, d)

      efc_force = d._impl.efc_force
      state = jp.concatenate([d.qpos, d.qvel, d.act])

      return efc_force, state

    x = jp.array([0.0, 0.01, 0.0, 0.0])
    u = jp.array([0.1])

    efc_force, _ = step(x, u)
    self.assertTrue((efc_force > 0.0).any(), 'inactive contact')

    eps = 1e-5  # finite difference step size
    tol = _TOLERANCE * 50  # looser tolerance when comparing with finite difference

    jax._src.test_util.check_jvp(
      step,
      partial(jax.jvp, step),
      (x, u),
      eps=eps,
      atol=tol,
      rtol=tol,
    )

    jax._src.test_util.check_vjp(
      step,
      partial(jax.vjp, step),
      (x, u),
      eps=eps,
      atol=tol,
      rtol=tol * 1.5,
    )


if __name__ == "__main__":
  absltest.main()
