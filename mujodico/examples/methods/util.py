from dataclasses import dataclass
import jax
from jax import numpy as jp
from mujodico import mjx
import numpy as np


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
  backend = 'jax' if isinstance(x, jax.Array) else 'numpy'

  e = x - x_goal

  for i in quaternion_w_idxs:
    if backend == 'jax':
      e = e.at[i:i+4].set(0.0)
    else:
      e[i:i+4] = 0.0

  cost = e.T @ Q @ e

  for i in quaternion_w_idxs:
    q = x[i:i+4]
    q_goal = x_goal[i:i+4]

    if backend == 'jax':
      q = mjx._src.math.normalize(q)
      q_goal = mjx._src.math.normalize(q_goal)
    else:
      q = q / np.linalg.norm(q)
      q_goal = q_goal / np.linalg.norm(q_goal)

    Q_block = Q[i:i+4, i:i+4]

    if backend == 'jax':
      weight = jp.mean(jp.diag(Q_block))
    else:
      weight = np.mean(np.diag(Q_block))

    cost += weight * (1.0 - (q.T @ q_goal)**2)

  return cost


def control_cost(
    R: jax.Array | np.ndarray,
    u_ref: jax.Array | np.ndarray,
    u: jax.Array | np.ndarray,
):
  return (u - u_ref).T @ R @ (u - u_ref)
