from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
import mujoco
from mujodico import mjx
from mujodico.mjx._src import support
from mujodico.trajopt._src import transcriptions
from mujodico.trajopt._src.transcriptions import (
  _Contact,
  _extract_contact_indices,
)
import numpy as np


def _contact_model(
    approximation='conic', cone='pyramidal', condim=1, smoothing=None,
    contact=True, integrator='Euler',
):
  approximation_xml = (
      f'<text name="contact_approx" data="{approximation}"/>'
      if approximation is not None else ''
  )
  smoothing_xml = (
      f'<numeric name="contact_smoothing" data="{smoothing}"/>'
      if smoothing is not None else ''
  )
  native = mujoco.MjModel.from_xml_string(f'''
    <mujoco>
      <custom>{approximation_xml}{smoothing_xml}</custom>
      <option cone="{cone}" timestep=".02" gravity="0 0 0"
              integrator="{integrator}"/>
      <default>
        <joint type="slide" range="-10 10"/>
        <geom condim="{condim}" friction=".4 .02 .01"
              solref="-1000 -1" solimp=".5 .5 .01 .5 2"/>
      </default>
      <worldbody>
        <geom type="plane" size="1 1 .1" contype="{int(contact)}"
              conaffinity="{int(contact)}"/>
        <body pos="0 0 .1">
          <joint name="x" axis="1 0 0"/>
          <joint axis="0 1 0"/>
          <joint axis="0 0 1"/>
          <geom type="sphere" size=".1" mass="2"/>
        </body>
      </worldbody>
      <actuator><motor joint="x" ctrlrange="-1 1"/></actuator>
    </mujoco>
  ''')
  return mjx.put_model(native, impl='jax')


class ContactImplicitTranscriptionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(jax.enable_x64())

  @parameterized.product(
      approximation=('conic', 'linear-elastic'),
      contact=(('pyramidal', 1), ('pyramidal', 3), ('pyramidal', 4),
               ('pyramidal', 6), ('elliptic', 3), ('elliptic', 4),
               ('elliptic', 6)),
  )
  def test_contact_law_and_force_mapping(self, approximation, contact):
    cone, dim = contact
    model = _contact_model(approximation, cone, dim)
    lower, upper = np.array([-.4]), np.array([.6])
    dynamics, inequality, complementarity, bounds = (
        transcriptions.contact_implicit_transcription(
            model, (lower, upper), reference_state=np.zeros(6)
        )
    )
    x = jp.array([.1, .2, -.01, .3, -.4, -.2])
    xnext = jp.array([4., -3., .5, .5, -.7, .1])
    if dim == 1:
      nz = 1
    elif cone == 'pyramidal':
      nz = 2 * (dim - 1)
    else:
      nz = 2 * dim
    z = jp.arange(1, nz + 1, dtype=x.dtype) * .2
    # Deliberately exceed actuator limits: the optimizer must enforce bounds.
    w = jp.concatenate((jp.array([2.5]), z))
    dt = float(model.opt.timestep)
    friction = np.array([.4, .4, .02, .01, .01])[:dim - 1]
    vt = np.r_[np.asarray(xnext[3:5]), np.zeros(3)][:dim - 1]
    normal_gap = float(x[2] + dt * xnext[5])
    force = np.zeros(3)
    if dim == 1:
      gap = np.array([normal_gap])
      normal_force = np.asarray(z)
      force[2] = z[0]
    elif cone == 'pyramidal':
      # Edges have Jacobians J_n ± μ_i J_ti in opposing pairs.
      gap = (normal_gap + dt * (friction * vt)[:, None] * [1, -1]).ravel()
      normal_force = np.asarray(z)
      edges = np.asarray(z).reshape(-1, 2)
      force[:2] = friction[:2] * (edges[:2, 0] - edges[:2, 1])
      force[2] = np.sum(z)
    else:
      gap = np.array([normal_gap])
      normal_force = np.asarray(z[:1])
      force[:2] = np.asarray(z[1:3] - z[dim:dim + 2])
      force[2] = z[0]
    # Direct solref gives K=1000/(.5²); force stiffness is 100 K.
    expected = gap
    if approximation == 'linear-elastic':
      expected = normal_force + 400000 * gap
    if dim > 1 and cone == 'elliptic':
      friction_twice = np.tile(friction, 2)
      expected = np.r_[
          expected, np.r_[vt, -vt] + float(z[-1]) / friction_twice,
          float(z[0]) - np.sum(np.asarray(z[1:-1]) / friction_twice),
      ]
    actual = jax.jit(inequality)(x, w, xnext)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    # Only v⁺ from the next state enters these constraints.
    np.testing.assert_allclose(
        inequality(x, w, xnext.at[:3].set(-xnext[:3])), actual, atol=1e-10
    )
    self.assertFalse(np.allclose(
        inequality(x, w, xnext.at[5].add(.1)), actual
    ))
    # Translation has mass 2 and no passive forces.
    force[0] += float(w[0])
    vnext = np.asarray(x[3:]) + dt * force / 2
    expected_state = np.r_[np.asarray(x[:3]) + dt * vnext, vnext]
    np.testing.assert_allclose(
        jax.jit(dynamics)(x, w), expected_state, atol=1e-12
    )
    np.testing.assert_array_equal(bounds[0], np.r_[lower, np.zeros(nz)])
    np.testing.assert_array_equal(bounds[1], np.r_[upper, np.full(nz, np.inf)])
    self.assertEqual(complementarity.pairs, [
        ((complementarity.Side.U, i + 1), (complementarity.Side.INEQ, i))
        for i in range(nz)
    ])
    self.assertFalse(model.opt.disableflags & mjx.DisableBit.CLAMPCTRL)
    self.assertEqual(support.contact_approx(model).value, approximation)
    self.assertEqual(float(support.contact_smoothing(model)), 0.)

  @parameterized.product(
      approximation=('conic', 'linear-elastic'),
      integrator=('Euler', 'implicitfast'),
      separated=(False, True),
  )
  def test_satisfies_complementarity(self, approximation, integrator, separated):
    model = _contact_model(approximation, integrator=integrator)
    dynamics, inequality, _, _ = transcriptions.contact_implicit_transcription(
        model, reference_state=np.zeros(6)
    )
    x = jp.array([0., 0., .02 if separated else -.01, 0., 0., -.2])
    dt = float(model.opt.timestep)
    compliance = 1 / 400000 if approximation == 'linear-elastic' else 0.
    normal = max(-float(x[2] + dt * x[5]) / (dt * dt / 2 + compliance), 0.)
    w = jp.array([0., normal])
    xnext = jax.jit(dynamics)(x, w)
    residual = jax.jit(inequality)(x, w, xnext)
    if separated:
      self.assertEqual(normal, 0.)
      self.assertGreater(float(residual[0]), 0.)
    else:
      self.assertGreater(normal, 0.)
      np.testing.assert_allclose(residual, 0., atol=1e-10)
    np.testing.assert_allclose(normal * residual, 0., atol=1e-8)
    # Check the velocity coefficient and independence from q⁺ in both AD modes.
    expected_jacobian = np.zeros((1, 6))
    coefficient = 400000 if approximation == 'linear-elastic' else 1
    expected_jacobian[0, 5] = dt * coefficient
    for differentiate in (jax.jacfwd, jax.jacrev):
      jacobian = jax.jit(differentiate(inequality, argnums=2))(x, w, xnext)
      np.testing.assert_allclose(jacobian, expected_jacobian, atol=1e-10)


class ExtractContactIndicesTest(parameterized.TestCase):

  @parameterized.parameters(
      ('pyramidal', 1, _Contact.Type.FRICTIONLESS, 1),
      ('pyramidal', 3, _Contact.Type.PYRAMIDAL,    4),
      ('elliptic',  3, _Contact.Type.ELLIPTIC,     3),
  )
  def test_extract_contact_indices(
      self, cone, condim, expected_type, expected_num_addresses
  ):
    xml = f'''
<mujoco>
  <option cone="{cone}" />
  <default>
    <geom condim="{condim}" friction="0.2" />
  </default>
  <worldbody>
    <body name="pusher" pos="0 0 0">
      <joint name="pusher_slide_x" type="slide" axis="1 0 0" range="-1 1" />
      <joint name="pusher_slide_y" type="slide" axis="0 1 0" range="-1 1" />
      <geom name="pusher_geom" type="sphere" size="0.05" />
    </body>
    <body name="object" pos="0 0 0">
      <joint name="object_slide_x" type="slide" axis="1 0 0" range="-1 1" />
      <joint name="object_slide_y" type="slide" axis="0 1 0" range="-1 1" />
      <geom name="object_geom" type="sphere" size="0.08" />
    </body>
  </worldbody>
</mujoco>
'''
    mjmodel = mujoco.MjModel.from_xml_string(xml)
    model = mjx.put_model(mjmodel, impl='jax')
    cf_addresses, contacts = _extract_contact_indices(model)

    self.assertEqual(cf_addresses.size, expected_num_addresses)
    self.assertEqual(cf_addresses[0], 4)  # there are 4 LIMIT_JOINT constraints

    self.assertLen(contacts, 1)
    contact = contacts[0]
    self.assertEqual(contact.type, expected_type)
    self.assertEqual(contact.dim, condim)
    self.assertEqual(contact.efc_index, int(cf_addresses[0]))
    self.assertEqual(contact.cf_index, 0)
    self.assertEqual(contact.friction.size, condim - 1)
    np.testing.assert_allclose(contact.friction, [0.2] * (condim - 1))


if __name__ == '__main__':
  absltest.main()
