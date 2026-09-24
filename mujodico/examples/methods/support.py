from dataclasses import dataclass
import enum
import jax
from jax import numpy as jp
from mujodico import mjx
# pylint: disable=g-importing-member
from mujodico.mjx._src.types import Model
# pylint: enable=g-importing-member
import numpy as np


class HessianApproxType(enum.Enum):
  GaussNewton = 'gaussnewton'
  Full = 'full'
  LBFGS = 'lbfgs'

  @classmethod
  def _resolve(cls, value: str | None):
    if isinstance(value, cls):
      return value

    if value is None:
      return cls.GaussNewton

    name = value.lower().strip().translate(str.maketrans('', '', ' -_'))
    for member in cls:
      if member.value.lower() == name:
        return member

    raise ValueError(
        'unknown hessian approximation: value should be one of '
        f'<{"|".join([member.value for member in cls])}>'
    )


def step_given_efc_force(
    m: Model, x: jax.Array, u: jax.Array, efc_force: jax.Array
  ) -> jax.Array:
  """Computes the mapping (x, u, λ) ↦ x⁺ assuming that the constraint force λ is known.
  The mapping satisfies:
    M(q) (v⁺ - v) / δt + c(q⁺, v⁺) = τ(q⁺, v⁺, u) + J(q)ᵀ λ
    q⁺ = q ⊕ δt v⁺
  """

  assert x.size == m.nq + m.nv + m.na
  assert u.size == m.nu

  d = mjx.make_data(m, impl=m.impl)

  d = d.replace(
    qpos=x[:m.nq],
    qvel=x[m.nq:m.nq+m.nv],
    act=x[m.nq+m.nv:],
    ctrl=u
  )

  d = mjx.fwd_position(m, d)
  d = mjx.fwd_velocity(m, d)
  d = mjx.fwd_actuation(m, d)
  d = mjx.fwd_acceleration(m, d)

  d = d.tree_replace({
      'qfrc_constraint': d._impl.efc_J.T @ efc_force,
      '_impl.efc_force': efc_force,
      'qacc': None,
  })

  if m.opt.integrator == mjx.IntegratorType.EULER:
    d = mjx.euler(m, d)
  elif m.opt.integrator == mjx.IntegratorType.IMPLICITFAST:
    d = mjx.implicit(m, d)
  else:
    raise NotImplementedError(f'integrator {m.opt.integrator} not implemented.')

  return jp.concatenate((d.qpos, d.qvel, d.act))


@dataclass(frozen=True)
class SOC:
  start_index: int
  dim: int
  friction: jax.Array


def extract_contact_indices(
    mjx_model: mjx.Model, mjx_data: mjx.Data
  ) -> tuple[np.ndarray, list[SOC]]:
  """
  Returns
    efc_contact_addresses: indices of efc_force that are contact forces
    socs:
      start_index: index of efc_contact_addresses that is the soc normal
      dim: dimension of the soc
      friction: friction coefficients of the soc
  """
  efc_type = np.array(mjx_data._impl.efc_type)
  efc_contact_addresses = np.nonzero(
      (efc_type == mjx.ConstraintType.CONTACT_FRICTIONLESS) |
      (efc_type == mjx.ConstraintType.CONTACT_PYRAMIDAL) |
      (efc_type == mjx.ConstraintType.CONTACT_ELLIPTIC)
  )[0]

  contact_dim = np.array(mjx_data._impl.contact.dim)
  contact_efc_address = np.array(mjx_data._impl.contact.efc_address)
  contact_is_elliptic = (
    efc_type[contact_efc_address] == mjx.ConstraintType.CONTACT_ELLIPTIC
  )

  if np.any(contact_is_elliptic):
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
  conic_friction = mjx_data._impl.contact.friction

  socs = []
  for i in np.nonzero(contact_is_elliptic)[0]:
    start_index = np.nonzero(contact_efc_address[i] == efc_contact_addresses)[0][0]
    dim = contact_dim[i]
    soc = SOC(
        start_index=int(start_index),
        dim=int(dim),
        friction=conic_friction[i, :dim-1],
    )
    socs.append(soc)

    if not np.all(soc.friction > 1e-6):
      raise ValueError('Near zero friction not allowed')

  return efc_contact_addresses, socs


def state_cost(
    Q,
    x_goal,
    x,
    quaternion_w_idxs: tuple[int, ...],
):
  try:
    import casadi as ca
  except ImportError:
    ca = None

  if ca is not None and isinstance(x, (ca.SX, ca.MX, ca.DM)):
    backend = 'casadi'
  elif isinstance(x, jax.Array):
    backend = 'jax'
  elif isinstance(x, np.ndarray):
    backend = 'numpy'
  else:
    raise TypeError(f'Unsupported type: {type(x)}')

  e = x - x_goal

  for i in quaternion_w_idxs:
    if backend == 'jax':
      e = e.at[i:i+4].set(0.0)
    elif backend == 'numpy':
      e[i:i+4] = 0.0
    elif backend == 'csadi':
      e = ca.vertcat(
          e[:i],
          ca.DM.zeros(4, 1),
          e[i+4:],
      )

  cost = e.T @ Q @ e

  for i in quaternion_w_idxs:
    q = x[i:i+4]
    q_goal = x_goal[i:i+4]
    if backend == 'jax':
      q = mjx._src.math.normalize(q)
      q_goal = mjx._src.math.normalize(q_goal)

    Q_block = Q[i:i+4, i:i+4]

    if backend == 'jax':
      weight = jp.mean(jp.diag(Q_block))
    elif backend == 'numpy':
      weight = np.mean(np.diag(Q_block))
    elif backend == 'casadi':
      weight = ca.trace(Q_block) / 4.0

    cost += weight * (1.0 - (q.T @ q_goal)**2)

  return cost
