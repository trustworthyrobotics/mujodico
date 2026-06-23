import jax
from jax import numpy as jp
from mujodico.mjx._src import conic_solver as conic
from mujodico.mjx._src import constraint
from mujodico.mjx._src import derivative
from mujodico.mjx._src import support
# pylint: disable=g-importing-member
from mujodico.mjx._src.types import ConstraintType
from mujodico.mjx._src.types import Data
from mujodico.mjx._src.types import DisableBit
from mujodico.mjx._src.types import IntegratorType
from mujodico.mjx._src.types import Model
from mujodico.mjx._src.types import OptionJAX
# pylint: enable=g-importing-member
import numpy as np


def _effective_m(m: Model, d: Data) -> tuple[jax.Array, jax.Array]:
  """Computes the effective mass matrix and qacc_smooth for the integrator type."""
  qm = support.full_m(m, d) if support.is_sparse(m) else d._impl.qM
  qacc_smooth = d.qacc_smooth
  if m.opt.integrator == IntegratorType.IMPLICITFAST:
    qderiv = derivative.deriv_smooth_vel(m, d)
    qm -= m.opt.timestep * qderiv

    qh, _ = jax.scipy.linalg.cho_factor(qm)
    qacc_smooth = jax.scipy.linalg.cho_solve((qh, False), d.qfrc_smooth)

  return qm, qacc_smooth


def solve(m: Model, d: Data) -> Data:
  if not isinstance(m.opt._impl, OptionJAX):
    raise ValueError('solve requires JAX backend implementation.')

  qm, qacc_smooth = _effective_m(m, d)

  efc_J = d._impl.efc_J
  efc_pos = d._impl.efc_pos

  cones, friction_scale = _build_cones(m, d)

  dt = m.opt.timestep
  tau = d._impl.efc_b / jp.maximum(d._impl.efc_k, 1e-8)

  P = qm
  q = -qm @ qacc_smooth
  A = -efc_J * friction_scale[:, None]
  b = (efc_J @ d.qvel + efc_pos / (dt + tau)) / dt * friction_scale

  warm_start = not m.opt.disableflags & DisableBit.WARMSTART

  x_init = d.qacc if warm_start else None
  z_init = d._impl.efc_force if warm_start else None

  x, z, s = conic.solve(
    P=P, q=q, A=A, b=b, cones=tuple(cones),
    x_init=x_init, z_init=z_init,
    iterations=m.opt.iterations,
    tolerance=m.opt.tolerance,
  )

  qacc = x
  efc_force = z

  qfrc_constraint = d._impl.efc_J.T @ efc_force

  d = d.tree_replace({
    'qfrc_constraint': qfrc_constraint,
    'qacc': qacc,
    '_impl.efc_force': efc_force,
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

    contact_smoothing = m.opt._impl.contact_smoothing

    cones = []
    friction_scale = jp.ones(nefc)

    i = 0
    while i < nefc:
      t = efc_type[i]

      if t == ConstraintType.EQUALITY:
        cones.append(conic.ZeroCone(dim=1))
        i += 1

      elif t in (ConstraintType.LIMIT_JOINT, ConstraintType.LIMIT_TENDON):
        cones.append(conic.NonNegativeCone(dim=1))
        i += 1

      elif t in (ConstraintType.CONTACT_FRICTIONLESS, ConstraintType.CONTACT_PYRAMIDAL):
        cones.append(conic.NonNegativeCone(dim=1, complementarity=contact_smoothing))
        i += 1

      elif t == ConstraintType.CONTACT_ELLIPTIC:
        dim, friction = contact_dim_friction[i]
        cones.append(conic.SecondOrderCone(dim=dim, complementarity=contact_smoothing))
        friction_scale = friction_scale.at[i+1:i+dim].set(friction[:dim-1])
        i += dim

      else:
        raise RuntimeError(f"Unsupported constraint type {t}")

    return tuple(cones), friction_scale
