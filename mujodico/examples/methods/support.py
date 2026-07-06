import enum
import jax
from jax import numpy as jp
from mujodico import mjx
# pylint: disable=g-importing-member
from mujodico.mjx._src.types import Model
# pylint: enable=g-importing-member


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
