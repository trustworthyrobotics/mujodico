from absl.testing import absltest
from absl.testing import parameterized
import jax
import mujoco
from mujodico import mjx
import numpy as np


_TOLERANCE = 5e-5


def _assert_eq(a, b, msg):
  np.testing.assert_allclose(a, b, atol=_TOLERANCE, rtol=_TOLERANCE, err_msg=msg)


def _check_collision(xml: str, test_cases: list[dict]):
  model = mujoco.MjModel.from_xml_string(xml)
  mjx_model = mjx.put_model(model, device=jax.devices()[0])
  mjx_data = mjx.make_data(mjx_model, impl=mjx_model.impl)

  @jax.jit
  def collision(qpos):
    d = mjx_data.replace(qpos=qpos)
    d = mjx.fwd_position(mjx_model, d)
    dist = d._impl.contact.dist[0]
    pos = d._impl.contact.pos[0]
    n = d._impl.contact.frame[0][0, :]
    return dist, pos, n

  for test_case in test_cases:
    dist, pos, n = collision(test_case['qpos'])
    _assert_eq(n, test_case['n'], 'n mismatch')
    _assert_eq(dist, test_case['dist'], 'dist mismatch')
    _assert_eq(pos, test_case['pos'], 'pos mismatch')


def _xpos_to_qpos(xpos: np.ndarray):
  xpos = np.array(xpos)
  qpos = np.hstack((
    xpos,
    np.tile([[1, 0, 0, 0]], (xpos.shape[0], 1)),
  ))
  return qpos.ravel()


def _normalize(x: np.ndarray):
  x = np.array(x)
  return x / np.linalg.norm(x)


class CollisionPrimitiveTest(parameterized.TestCase):

  def test_sphere_capsule(self):
    xml = """
<mujoco>
  <worldbody>
    <body>
      <joint type="free"/>
      <geom size="0.1" type="sphere"/>
    </body>
    <body>
      <joint type="free"/>
      <geom fromto="-0.2 0 0 0.2 0 0" size="0.1" type="capsule"/>
    </body>
  </worldbody>
</mujoco>
"""
    test_cases = []
    test_cases.append({
      'qpos': _xpos_to_qpos([[0, 0, 0.1], [0, 0, 0]]),
      'dist': -0.1,
      'pos': [0, 0, 0.05],
      'n': [0, 0, -1],
    })
    test_cases.append({
      'qpos': _xpos_to_qpos([[0.3, 0, 0], [0, 0, 0]]),
      'dist': -0.1,
      'pos': np.array([0.25, 0, 0]),
      'n': [-1, 0, 0],
    })

    _check_collision(xml, test_cases)

  def test_sphere_box(self):
    xml = """
<mujoco>
  <worldbody>
    <body>
      <joint type="free"/>
      <geom size="0.1" type="sphere"/>
    </body>
    <body>
      <joint type="free"/>
      <geom size="1 1.5 2" type="box"/>
    </body>
  </worldbody>
</mujoco>
"""
    test_cases = []
    test_cases.append({
      'qpos': _xpos_to_qpos([[1.0, 0, 0], [0, 0, 0]]),
      'dist': -0.1,
      'pos': [0.95, 0, 0],
      'n': [-1, 0, 0],
    })

    n = _normalize([-1, -1, 0])
    shift = 0.6e-3
    sphere_pos = np.array([1.0, 1.5, 0]) + n * shift
    dist = 1.0e-3 * (np.sqrt(2) - 1) - shift - 0.1
    test_cases.append({
      'qpos': _xpos_to_qpos([sphere_pos, [0, 0, 0]]),
      'dist': dist,
      'pos': sphere_pos + n * (0.1 + dist / 2),
      'n': n,
    })

    n = _normalize([-1, -1, -1])
    shift = 0.9e-3
    sphere_pos = np.array([1.0, 1.5, 2.0]) + n * shift
    dist = 1.0e-3 * (np.sqrt(3) - 1) - shift - 0.1
    test_cases.append({
      'qpos': _xpos_to_qpos([sphere_pos, [0, 0, 0]]),
      'dist': dist,
      'pos': sphere_pos + n * (0.1 + dist / 2),
      'n': n,
    })

    test_cases.append({
      'qpos': _xpos_to_qpos([[0, 0, 1.5], [0, 0, 0]]),
      'dist': -0.6,
      'pos': [0, 0, 1.7],
      'n': [0, 0, -1],
    })

    _check_collision(xml, test_cases)

  def test_sphere_cylinder(self):
    xml = """
<mujoco>
  <worldbody>
    <body>
      <joint type="free"/>
      <geom size="0.1" type="sphere"/>
    </body>
    <body>
      <joint type="free"/>
      <geom fromto="-0.2 0 0 0.2 0 0" size="0.1" type="cylinder"/>
    </body>
  </worldbody>
</mujoco>
"""
    test_cases = []
    test_cases.append({
      'qpos': _xpos_to_qpos([[0, 0, 0.09], [0, 0, 0]]),
      'dist': -0.11,
      'pos': [0, 0, 0.045],
      'n': [0, 0, -1],
    })
    test_cases.append({
      'qpos': _xpos_to_qpos([[0.2, 0.03, 0], [0, 0, 0]]),
      'dist': -0.1,
      'pos': [0.15, 0.03, 0],
      'n': [-1, 0, 0],
    })

    n = _normalize([-1, -1, 0])
    shift = 0.6e-4
    sphere_pos = np.array([0.2, 0.1, 0]) + n * shift
    dist = 1.0e-4 * (np.sqrt(2) - 1) - shift - 0.1
    test_cases.append({
      'qpos': _xpos_to_qpos([sphere_pos, [0, 0, 0]]),
      'dist': dist,
      'pos': sphere_pos + n * (0.1 + dist / 2),
      'n': n,
    })

    _check_collision(xml, test_cases)


if __name__ == "__main__":
  absltest.main()
