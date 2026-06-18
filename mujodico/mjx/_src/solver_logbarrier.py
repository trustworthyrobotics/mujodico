import jax
from jax import numpy as jp
from mujodico.mjx._src import constraint
from mujodico.mjx._src import derivative
from mujodico.mjx._src import math
from mujodico.mjx._src import smooth
from mujodico.mjx._src import support
# pylint: disable=g-importing-member
from mujodico.mjx._src.dataclasses import PyTreeNode
from mujodico.mjx._src.types import ConeType
from mujodico.mjx._src.types import ConstraintType
from mujodico.mjx._src.types import Data
from mujodico.mjx._src.types import DataJAX
from mujodico.mjx._src.types import DisableBit
from mujodico.mjx._src.types import Model
from mujodico.mjx._src.types import ModelJAX
from mujodico.mjx._src.types import OptionJAX
# pylint: enable=g-importing-member

import clarabel
import numpy as np
from scipy import sparse


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

  qacc, efc_force = jax.pure_callback(
      clarabel_solve,
      (
        jax.ShapeDtypeStruct((m.nv,), jp.float32),
        jax.ShapeDtypeStruct((d._impl.nefc,), jp.float32),
      ),
      qm, qacc_smooth, d, m.opt.timestep,
  )

  qfrc_constraint = d._impl.efc_J.T @ efc_force

  d = d.tree_replace({
      'qfrc_constraint': qfrc_constraint,
      'qacc': qacc,
      '_impl.efc_force': efc_force,
  })

  return d


def clarabel_solve(qm: jax.Array, qacc_smooth: jax.Array, d: Data, dt) -> tuple[np.ndarray, np.ndarray]:
  qm = np.asarray(qm)
  qacc_smooth = np.asarray(qacc_smooth)
  efc_J = np.array(d._impl.efc_J)
  efc_pos = np.array(d._impl.efc_pos)

  contact = d._impl.contact
  contact_dim_friction = {}
  for i in range(len(contact.efc_address)):
    contact_dim_friction[contact.efc_address[i]] = (contact.dim[i], contact.friction[i])

  cones = []
  efc_address = 0
  while efc_address < d.nefc:
    efc_type = d._impl.efc_type[efc_address]
    if efc_type == ConstraintType.EQUALITY:
      cones.append(clarabel.ZeroConeT(1))
      efc_address += 1
    elif efc_type in (
      ConstraintType.FRICTION_DOF,
      ConstraintType.FRICTION_TENDON,
    ):
      raise RuntimeError("Solver logbarrier does not support frictionloss")
    elif efc_type in (
      ConstraintType.LIMIT_JOINT,
      ConstraintType.LIMIT_TENDON,
      ConstraintType.CONTACT_FRICTIONLESS,
      ConstraintType.CONTACT_PYRAMIDAL
    ):
      cones.append(clarabel.NonnegativeConeT(1))
      efc_address += 1
    elif efc_type == ConstraintType.CONTACT_ELLIPTIC:
      dim, friction = contact_dim_friction[efc_address]
      cones.append(clarabel.SecondOrderConeT(dim))
      efc_J[efc_address+1 : efc_address+dim] *= friction[:dim-1]
      efc_pos[efc_address+1 : efc_address+dim] *= friction[:dim-1]
      efc_address += dim

  # solve using clarabel
  P = sparse.csc_matrix(qm)
  q = -qm @ qacc_smooth
  A = sparse.csc_matrix(-efc_J)
  b = efc_pos / (dt * dt)

  settings = clarabel.DefaultSettings()
  settings.verbose = False
  settings.time_limit = 5.0

  solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
  solution = solver.solve()

  qacc = np.array(solution.x)
  efc_force = np.array(solution.z)

  return qacc, efc_force
