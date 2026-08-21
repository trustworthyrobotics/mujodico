"""
Solves next step contact force and velocity/acceleration for a log-barrier
contact approximation. Under the contact approximaiton, the dynamcis are
    M(q) (v⁺ - v) + δt c(q⁺, v⁺) = δt τ(q⁺, v⁺, u) + J(q)ᵀ γ
    q⁺ = q ⊕ δt v⁺
    ν := J(q) v⁺ + ϕ(q) / (δt + τ) ∈ ℱ*
    γ ∈ ℱ
    ν ∘ γ = μ e

By defining the effective mass matrix
    A = M(q) - d[τ(q, v, u) - c(q, v)] / dv δt
and the mass matrix times free-motion velocity
    A v* = A v + δt [τ(q, v, u) - c(q, v)]
we can write the dynamics as
    A v⁺ - A v* = J(q)ᵀ γ
    q⁺ = q ⊕ h v⁺
    ν := J(q) v⁺ + ϕ(q) / (δt + τ)  ∈ ℱ*
    γ ∈ ℱ
    ν ∘ γ = μ e

The complementarity condition can be written as
    ν δt ∘ γ/δt = μ e
where ν δt has unit of length and γ/δt has unit of force.
Thus, provided that μ ∝ (δt)⁰, the contact force is μ divided by the
signed distance and is indepedent of timestep.
"""

import jax
from jax import numpy as jp
from mujodico.mjx._src import conic_solver as conic
from mujodico.mjx._src import constraint
from mujodico.mjx._src import derivative
from mujodico.mjx._src import forward
from mujodico.mjx._src import io
from mujodico.mjx._src import support
# pylint: disable=g-importing-member
from mujodico.mjx._src.types import ConstraintType
from mujodico.mjx._src.types import Data
from mujodico.mjx._src.types import DisableBit
from mujodico.mjx._src.types import IntegratorType
from mujodico.mjx._src.types import Model
from mujodico.mjx._src.types import OptionJAX
# pylint: enable=g-importing-member


def _linearize_unconstrained_dynamics(m: Model, d: Data) -> tuple[jax.Array, jax.Array]:
  """Computes the effective mass matrix and its product with the free-motion velocity
  for the integrator type.
  """

  dt = m.opt.timestep

  qM = support.full_m(m, d) if support.is_sparse(m) else d._impl.M
  if m.opt.integrator == IntegratorType.IMPLICITFAST:
    qM -= derivative.deriv_smooth_vel(m, d) * dt

  qM_qvel_smooth = qM @ d.qvel + d.qfrc_smooth * dt

  return qM, qM_qvel_smooth


def solve(m: Model, d: Data) -> Data:
  if not isinstance(m.opt._impl, OptionJAX):
    raise ValueError('solve requires JAX backend implementation.')

  dt = m.opt.timestep

  qM, qM_qvel_smooth = _linearize_unconstrained_dynamics(m, d)

  efc_J = d._impl.efc_J
  efc_pos = d._impl.efc_pos

  cones, comps, friction_scale = _build_cones(m, d)

  dt = m.opt.timestep
  tau = d._impl.efc_b / jp.maximum(d._impl.efc_k, 1e-8)

  P = qM
  q = -qM_qvel_smooth
  A = -efc_J * friction_scale[:, None]
  b = efc_pos / (dt + tau) * friction_scale

  warm_start = not m.opt.disableflags & DisableBit.WARMSTART

  x_init = d.qvel if warm_start else None
  z_init = d._impl.efc_force * dt if warm_start else None

  x, z, s = conic.solve(
      P=P,
      q=q,
      A=A,
      b=b,
      cones=tuple(cones),
      complementarities=comps,
      x_init=x_init,
      z_init=z_init,
      iterations=m.opt.iterations,
      tolerance=m.opt.tolerance,
  )

  qvel = x
  efc_impulse = z

  qacc = (qvel - d.qvel) / dt
  efc_force = efc_impulse / dt
  qfrc_constraint = efc_J.T @ efc_force

  d = d.tree_replace({
      'qacc': qacc,
      '_impl.efc_force': efc_force,
      'qfrc_constraint': qfrc_constraint,
  })

  return d


def _build_cones(m: Model, d: Data):
  contact = d._impl.contact

  contact_dim_friction = {
      contact.efc_address[i]: (contact.dim[i], contact.friction[i])
      for i in range(len(contact.efc_address))
  }

  efc_type = constraint.make_efc_type(m)
  nefc = efc_type.size

  complementarity = m.opt._impl.contact_smoothing

  cones = []
  comps = []
  friction_scale = jp.ones(nefc)

  i = 0
  while i < nefc:
    t = efc_type[i]

    if t == ConstraintType.EQUALITY:
      cones.append(conic.ZeroCone(dim=1))
      comps.append(0.0)
      i += 1

    elif t in (ConstraintType.LIMIT_JOINT, ConstraintType.LIMIT_TENDON):
      cones.append(conic.NonNegativeCone(dim=1))
      comps.append(0.0)
      i += 1

    elif t in (
        ConstraintType.CONTACT_FRICTIONLESS,
        ConstraintType.CONTACT_PYRAMIDAL,
    ):
      cones.append(conic.NonNegativeCone(dim=1))
      comps.append(complementarity)
      i += 1

    elif t == ConstraintType.CONTACT_ELLIPTIC:
      dim, friction = contact_dim_friction[i]
      cones.append(conic.SecondOrderCone(dim=dim))
      comps.append(complementarity)
      friction_scale = friction_scale.at[i + 1 : i + dim].set(
          friction[: dim - 1]
      )
      i += dim

    else:
      raise RuntimeError(f'Unsupported constraint type {t}')

  return tuple(cones), jp.array(comps), friction_scale


def constitutive_relations(m: Model, efc_contact_mask: jax.Array):
  """Returns the constitutive relations including expression, lower bound, and upper bound."""

  nq, nv, na = m.nq, m.nv, m.na
  nx = nq + nv + na

  nc = int(jp.count_nonzero(efc_contact_mask))

  data = io.make_data(m, impl=m.impl)

  def constitutive_fn(x: jax.Array, u: jax.Array, cf: jax.Array, x_next: jax.Array):
    assert x.size == nx
    assert u.size == m.nu
    assert cf.size == nc
    assert x_next.size == nx

    d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)

    d = forward.fwd_position(m, d)

    dt = m.opt.timestep
    tau = d._impl.efc_b / jp.maximum(d._impl.efc_k, 1e-8)

    qvel_next = x_next[nq : nq + nv]

    cpos = d._impl.efc_J @ qvel_next + d._impl.efc_pos / (dt + tau)
    cpos = cpos[efc_contact_mask]

    return jp.concatenate([
        cpos,
        cpos * cf,
    ])

  mu_e = m.opt._impl.contact_smoothing * jp.ones(nc)

  constitutive_lb = jp.concatenate([
    jp.zeros(nc),
    jp.zeros_like(mu_e),
  ])

  constitutive_ub = jp.concatenate([
    jp.full(nc, jp.inf),
    mu_e,
  ])

  return constitutive_fn, constitutive_lb, constitutive_ub
