"""Differentiable quaternion operations using scalar-first [w, x, y, z].

The exponential and logarithm use rotation vectors: the vector's norm is
the rotation angle in radians. Each operation acts on a single quaternion.
"""

import jax
import jax.numpy as jp


def quat_multiply(q: jax.Array, r: jax.Array) -> jax.Array:
  """Return the Hamilton product of two quaternions shaped (4,)."""
  q, r = jp.asarray(q), jp.asarray(r)
  assert q.shape == (4,), f'Expected q shape (4,), got {q.shape}.'
  assert r.shape == (4,), f'Expected r shape (4,), got {r.shape}.'
  qw, qv = q[0], q[1:]
  rw, rv = r[0], r[1:]
  scalar = qw * rw - jp.dot(qv, rv)
  vector = qw * rv + rw * qv + jp.cross(qv, rv)
  return jp.concatenate([scalar[None], vector])


def quat_inverse(q: jax.Array) -> jax.Array:
  """Return conjugate(q) / norm(q)**2 for a nonzero quaternion (4,)."""
  q = jp.asarray(q)
  assert q.shape == (4,), f'Expected q shape (4,), got {q.shape}.'
  return q * jp.array([1, -1, -1, -1]) / jp.dot(q, q)


def quat_exp(v: jax.Array) -> jax.Array:
  """Map a rotation vector (3,) to a unit quaternion (4,).

  This is [cos(norm(v)/2), sin(norm(v)/2) * v/norm(v)], with a smooth
  extension at zero. Its Jacobian there is [0; I/2].
  """
  v = jp.asarray(v)
  assert v.shape == (3,), f'Expected v shape (3,), got {v.shape}.'
  theta2 = jp.dot(v, v)
  small = theta2 < 1e-6
  # Mask before sqrt so even inactive branches have finite derivatives at zero.
  theta = jp.sqrt(jp.where(small, 1.0, theta2))
  scalar = jp.where(
      small,
      1 - theta2 / 8 + theta2**2 / 384 - theta2**3 / 46080,
      jp.cos(theta / 2),
  )
  scale = jp.where(
      small,
      0.5 - theta2 / 48 + theta2**2 / 3840 - theta2**3 / 645120,
      jp.sin(theta / 2) / theta,
  )
  return jp.concatenate([scalar[None], scale * v])


def quat_log(q: jax.Array) -> jax.Array:
  """Map a nonzero quaternion (4,) to its shortest rotation vector (3,).

  The result has norm at most pi. Positive scaling and
  antipodal quaternions represent the same rotation. At exactly pi the sign
  follows the input vector part; the map is discontinuous across this cut.
  At [1, 0, 0, 0] the Jacobian is [0, 2*I].
  """
  q = jp.asarray(q)
  assert q.shape == (4,), f'Expected q shape (4,), got {q.shape}.'
  q = jp.where(q[0] < 0, -q, q)
  w, v = q[0], q[1:]
  norm2 = jp.dot(v, v)
  small = norm2 < 1e-6 * w * w
  norm = jp.sqrt(jp.where(small, 1.0, norm2))
  safe_w = jp.where(small, w, 1.0)
  ratio2 = norm2 / safe_w**2
  scale = jp.where(
      small,
      (2 / safe_w) * (1 - ratio2 / 3 + ratio2**2 / 5 - ratio2**3 / 7),
      2 * jp.arctan2(norm, w) / norm,
  )
  return scale * v
