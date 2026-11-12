from dataclasses import dataclass
import jax
from jax import numpy as jp
from mujodico.trajopt._src import manifold
import numpy as np
from scipy.spatial.transform import Rotation


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TranscriptionParams:
  x_0: jax.Array
  x_goal: jax.Array
  Q: jax.Array
  Qf: jax.Array
  u_ref: jax.Array
  R: jax.Array


def state_cost(
    Q: jax.Array | np.ndarray,
    x_goal: jax.Array | np.ndarray,
    x: jax.Array | np.ndarray,
    quaternion_w_idxs: list[int],
):
  """Quadratic state error with squared shortest-rotation quaternion errors.

  Each quaternion uses the mean diagonal weight of its four-coordinate Q
  block, multiplying ||Log(q_goal**-1 * q)||**2 in radians squared.
  """
  backend = 'jax' if isinstance(x, jax.Array) else 'numpy'

  e = x - x_goal

  for i in quaternion_w_idxs:
    if backend == 'jax':
      e = e.at[i : i+4].set(0.0)
    else:
      e[i : i+4] = 0.0

  cost = e.T @ Q @ e

  for i in quaternion_w_idxs:
    q = x[i : i + 4]
    q_goal = x_goal[i : i + 4]

    if backend == 'jax':
      error = manifold.quat_log(
          manifold.quat_multiply(manifold.quat_inverse(q_goal), q)
      )
    else:
      relative = Rotation.from_quat(
          np.roll(q_goal, -1)
      ).inv() * Rotation.from_quat(np.roll(q, -1))
      error = relative.as_rotvec()

    Q_block = Q[i : i + 4, i : i + 4]

    if backend == 'jax':
      weight = jp.mean(jp.diag(Q_block))
    else:
      weight = np.mean(np.diag(Q_block))

    cost += weight * (error @ error)

  return cost


def control_cost(
    R: jax.Array | np.ndarray,
    u_ref: jax.Array | np.ndarray,
    u: jax.Array | np.ndarray,
):
  return (u - u_ref).T @ R @ (u - u_ref)
