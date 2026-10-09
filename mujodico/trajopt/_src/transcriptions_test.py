from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from mujodico import mjx
from mujodico.trajopt._src.transcriptions import (
  _Contact,
  _extract_contact_indices,
)
import numpy as np


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
