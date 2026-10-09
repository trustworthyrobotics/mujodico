from dataclasses import dataclass
from enum import Enum
import jax
import jax.numpy as jp
import mujoco
from mujodico import mjx
from mujodico.trajopt._src.types import Bounds
import numpy as np


def shooting_transcription(
    mjmodel: mujoco.MjModel,
    contact_approximation: mjx.ContactApproxType,
    contact_smoothing: float,
):
  """Builds a one-step shooting dynamics function with smoothed derivatives.

  The returned function takes a state ``x`` and control ``u``, and returns
  the next state ``x⁺``. Its value comes from a step with zero contact smoothing;
  its derivatives come from a step using ``contact_smoothing``.

  Args:
    mjmodel: MuJoCo model to transcribe.
    contact_approximation: contact approximation used by both models.
    contact_smoothing: smoothing parameter used to compute dynamics derivatives.

  Returns:
    dynamics: ``dynamics(x, u)`` returns the next state.
  """
  model = mjx.put_model(
      mjmodel,
      impl='jax',
      contact_approximation=contact_approximation,
      contact_smoothing=0.0,
  )
  smooth_model = mjx.put_model(
      mjmodel,
      impl=model.impl,
      contact_approximation=contact_approximation,
      contact_smoothing=contact_smoothing,
  )
  data = mjx.make_data(model, impl=model.impl)

  def dynamics(x: jax.Array, u: jax.Array) -> jax.Array:
    nq, nv = model.nq, model.nv
    d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)

    next_data = mjx.step(model, d)
    xnext = jp.concatenate((next_data.qpos, next_data.qvel, next_data.act))

    smooth_data = mjx.step(smooth_model, d)
    xnext_smooth = jp.concatenate((smooth_data.qpos, smooth_data.qvel, smooth_data.act))

    return (
        jax.lax.stop_gradient(xnext)
        + xnext_smooth
        - jax.lax.stop_gradient(xnext_smooth)
    )

  return dynamics


def lifted_shooting_transcription(
    mjmodel: mujoco.MjModel,
    contact_approximation: mjx.ContactApproxType,
    contact_smoothing: float,
    u_bounds: Bounds | None = None,
):
  """Builds dynamics and constraints with contact forces as lifted controls.

  The returned decision input ``w`` is ``[u, cf]``, where ``u`` is the
  actuator control and ``cf`` contains contact forces. The stage equality
  enforces that ``cf`` matches the forces computed by the forward dynamics.
  The stage inequality enforces the elliptic friction cone for each elliptic
  contact.

  Args:
    mjmodel: MuJoCo model to transcribe.
    contact_approximation: contact approximation used by the dynamics.
    contact_smoothing: contact smoothing parameter used by the dynamics.
    u_bounds: optional pair of lower and upper actuator control bounds.
              Defaults to unbounded controls.

  Returns:
    dynamics: ``dynamics(x, w)`` returns the next state.
    stage_equality: constraint ``stage_equality(x, w) = 0``.
    stage_inequality: constraint ``stage_inequality(x, w) ≥ 0``.
    w_bounds: pair of lower and upper bounds for ``w``.
  """
  nu = mjmodel.nu
  if u_bounds is None:
    u_lb = np.full(nu, -np.inf)
    u_ub = np.full(nu, np.inf)
  else:
    u_lb, u_ub = u_bounds
  if u_lb.size != nu or u_ub.size != nu:
    raise ValueError('u lower and upper bounds have incorrect size')

  model = mjx.put_model(
      mjmodel,
      impl='jax',
      contact_approximation=contact_approximation,
      contact_smoothing=contact_smoothing,
  )
  data = mjx.make_data(model, impl=model.impl)

  cf_addresses, contacts = _extract_contact_indices(model)
  elliptic_contacts = [c for c in contacts if c.type == _Contact.Type.ELLIPTIC]
  ncf = int(cf_addresses.size)

  def step1(x: jax.Array, u: jax.Array) -> jax.Array:
    nq, nv = model.nq, model.nv
    d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)
    d = mjx.forward(model, d)
    return d._impl.efc_force[cf_addresses]

  def step2(x: jax.Array, u: jax.Array, cf: jax.Array) -> jax.Array:
    efc_force = jp.zeros(data._impl.nefc)
    efc_force = efc_force.at[cf_addresses].set(cf)
    return _step_given_efc_force(model, x, u, efc_force)

  def unpack_control(w):
    return w[:nu], w[nu:]

  def dynamics(x: jax.Array, w: jax.Array) -> jax.Array:
    u, cf = unpack_control(w)
    return step2(x, u, cf)

  def stage_equality(x: jax.Array, w: jax.Array) -> jax.Array:
    u, cf = unpack_control(w)
    return cf - step1(x, u)

  def stage_inequality(x: jax.Array, w: jax.Array) -> jax.Array:
    del x
    _, cf = unpack_control(w)
    constraints = jp.zeros(len(elliptic_contacts))
    for i, contact in enumerate(elliptic_contacts):
      normal = cf[contact.cf_index]
      tangent = cf[contact.cf_index+1:contact.cf_index+contact.dim] / jp.asarray(contact.friction)
      constraints = constraints.at[i].set(normal * normal - tangent.dot(tangent))
    return constraints

  cf_lb = np.zeros(ncf)
  cf_ub = np.full(ncf, jp.inf)
  for contact in elliptic_contacts:
    cf_lb[contact.cf_index+1:contact.cf_index+contact.dim] = -np.inf

  w_bounds = (np.concatenate([u_lb, cf_lb]), np.concatenate([u_ub, cf_ub]))

  return dynamics, stage_equality, stage_inequality, w_bounds


def contact_implicit_transcription(
    mjmodel: mujoco.MjModel,
    u_bounds: Bounds | None = None,
):
  pass


@dataclass(frozen=True)
class _Contact:
  class Type(Enum):
    FRICTIONLESS = 'frictionless'
    PYRAMIDAL = 'pyramidal'
    ELLIPTIC = 'elliptic'

  type: Type
  efc_index: int
  cf_index: int
  dim: int
  friction: np.ndarray


def _extract_contact_indices(model: mjx.Model) -> tuple[np.ndarray, list[_Contact]]:
  """
  Returns
    cf_addresses: integer indices into ``efc_force`` for all contact force
                  components, in sorted order.
    contacts: one record per contact, with these fields:
      type: a ``_ContactType``
      efc_index: starting index of this contact in ``efc_force``
      cf_index: starting index of this contact in ``cf_addresses``
      dim: the contact dimension; frictionless contacts always have dimension 1
      friction: friction coefficients (size equals dim-1)
  """
  data = mjx.make_data(model, impl=model.impl)
  data = jax.jit(mjx.fwd_position)(model, data)

  efc_type = np.asarray(data._impl.efc_type)
  cf_addresses = np.nonzero(
      (efc_type == mjx.ConstraintType.CONTACT_FRICTIONLESS) |
      (efc_type == mjx.ConstraintType.CONTACT_PYRAMIDAL) |
      (efc_type == mjx.ConstraintType.CONTACT_ELLIPTIC)
  )[0]

  contact_dim = np.asarray(data._impl.contact.dim)
  contact_efc_address = np.asarray(data._impl.contact.efc_address)
  contact_type = efc_type[contact_efc_address]
  contact_friction = np.asarray(data._impl.contact.friction)

  contacts = []
  for i, efc_index in enumerate(contact_efc_address):
    dim = int(contact_dim[i])
    if contact_type[i] == mjx.ConstraintType.CONTACT_FRICTIONLESS:
      type_ = _Contact.Type.FRICTIONLESS
    elif contact_type[i] == mjx.ConstraintType.CONTACT_PYRAMIDAL:
      type_ = _Contact.Type.PYRAMIDAL
    elif contact_type[i] == mjx.ConstraintType.CONTACT_ELLIPTIC:
      type_ = _Contact.Type.ELLIPTIC
    else:
      continue

    contact = _Contact(
        type=type_,
        dim=dim,
        efc_index=int(efc_index),
        cf_index=int(np.searchsorted(cf_addresses, efc_index)),
        friction=contact_friction[i, :dim-1],
    )
    contacts.append(contact)

    if not np.all(contact.friction > 1e-4):
      raise ValueError('near zero friction not allowed')

  return cf_addresses, contacts


def _step_given_efc_force(
    m: mjx.Model, x: jax.Array, u: jax.Array, efc_force: jax.Array
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
