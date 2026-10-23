from dataclasses import dataclass
from enum import Enum
import jax
import jax.numpy as jp
import mujoco
from mujodico import mjx
from mujodico.trajopt._src.types import Bounds, LocalComplementarityConstraint
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
  model = mjx.put_model(
      mjmodel,
      impl='jax',
      contact_approximation=contact_approximation,
      contact_smoothing=contact_smoothing,
  )
  data = mjx.make_data(model, impl=model.impl)

  cf_addresses, contacts = _extract_contact_indices(model)
  elliptic_contacts = [c for c in contacts if c.type == _Contact.Type.ELLIPTIC]

  nu = model.nu
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
    assert w.size == nu + ncf
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
      tangent = cf[contact.cf_index+1:contact.cf_index+contact.dim] / contact.friction
      constraints = constraints.at[i].set(normal * normal - tangent.dot(tangent))
    return constraints

  cf_lb = np.zeros(ncf)
  cf_ub = np.full(ncf, np.inf)
  for contact in elliptic_contacts:
    cf_lb[contact.cf_index+1:contact.cf_index+contact.dim] = -np.inf

  u_lb, u_ub = _check_u_bounds(u_bounds, nu)
  w_bounds = (np.concatenate([u_lb, cf_lb]), np.concatenate([u_ub, cf_ub]))

  return dynamics, stage_equality, stage_inequality, w_bounds


def contact_implicit_transcription(
    mjmodel: mujoco.MjModel,
    u_bounds: Bounds | None = None,
    *,
    contact_at_next_state: bool = True,
):
  """Builds a contact-implicit transcription.

  The returned decision input ``w`` is ``[u, z]``. ``u`` is the actuator
  control; ``z`` contains nonnegative contact force variables.
  Uses Anitescu convex formulation for pyramidal contact.
  Uses Stewart-Trinkle formulation for elliptic contact.

  Args:
    mjmodel: MuJoCo model to transcribe.
    u_bounds: optional pair of lower and upper actuator control bounds.
              Defaults to unbounded controls.
    contact_at_next_state: if True, contact laws use the next state supplied
      by the optimizer. If False, they use the current state and the returned
      inequality takes only ``(x, w)``.

  Returns:
    dynamics: ``dynamics(x, w)`` returns the next state.
    inequality: ``local_inequality(x, w, xnext) ≥ 0`` by default, or
      ``stage_inequality(x, w) ≥ 0`` when contact_at_next_state is False.
    local_complementarity: local complementarity constraints.
    w_bounds: pair of lower and upper bounds for ``w``.
  """
  model = mjx.put_model(
      mjmodel,
      impl='jax',
      # As long as contact_approximation is not 'mujoco', efc_J and efc_pos
      # will not be masked out (see mjx/constraint.py).
      contact_approximation='conic',
  )
  data = mjx.make_data(model, impl=model.impl)

  cf_addresses, contacts = _extract_contact_indices(model)

  def z_dim(contact: _Contact) -> int:
    if contact.type == _Contact.Type.FRICTIONLESS:
      return contact.dim
    elif contact.type == _Contact.Type.PYRAMIDAL:
      return (contact.dim - 1) * 2
    elif contact.type == _Contact.Type.ELLIPTIC:
      return contact.dim * 2

  nu = model.nu
  nz = 0
  z_indexes = []
  for contact in contacts:
    z_indexes.append(nz)
    nz += z_dim(contact)

  def efc_info(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    nq, nv = model.nq, model.nv
    d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:])
    d = mjx.fwd_position(model, d)
    return d._impl.efc_J, d._impl.efc_pos

  def unpack_control(w):
    assert w.size == nu + nz
    return w[:nu], w[nu:]

  def dynamics(x: jax.Array, w: jax.Array) -> jax.Array:
    u, z = unpack_control(w)

    cf = jp.zeros(cf_addresses.size)
    for z_index, contact in zip(z_indexes, contacts):
      cf_dim = z_dim(contact) if contact.type != _Contact.Type.ELLIPTIC else contact.dim
      z_k = z[z_index : z_index + z_dim(contact)]
      cf_k = z_k[:cf_dim]
      if contact.type == _Contact.Type.ELLIPTIC:
        cf_k = cf_k.at[1:].add(-z_k[cf_dim:-1])
      cf = cf.at[contact.cf_index : contact.cf_index + cf_dim].set(cf_k)

    efc_force = jp.zeros(data._impl.nefc)
    efc_force = efc_force.at[cf_addresses].set(cf)
    return _step_given_efc_force(model, x, u, efc_force)

  def local_inequality(x: jax.Array, w: jax.Array, x_next: jax.Array) -> jax.Array:
    _, z = unpack_control(w)
    v_next = x_next[model.nq:model.nq+model.nv]
    efc_J, efc_pos = efc_info(x)
    efc_J_next, efc_pos_next = efc_info(x_next)

    constraints = []
    for z_index, contact in zip(z_indexes, contacts):
      if contact.type == _Contact.Type.FRICTIONLESS:
        # 0 ≤ λₙ ⊥ ϕ(q⁺) ≥ 0
        constraints.append(efc_pos_next[contact.efc_index][None])

      elif contact.type == _Contact.Type.PYRAMIDAL:
        # 0 ≤ λ ⊥ ϕ(q) + δt J(q) v⁺ ≥ 0
        dt = model.opt.timestep
        efc_indexs = jp.arange(contact.efc_index, contact.efc_index + z_dim(contact))
        constraints.append(efc_pos[efc_indexs] + dt * efc_J[efc_indexs] @ v_next)

      elif contact.type == _Contact.Type.ELLIPTIC:
        # 0 ≤ λₙ ⊥ ϕ(q⁺) ≥ 0
        # 0 ≤ λₜ₁ ⊥ Jₜ(q⁺) v⁺ + ν 1 ≥ 0
        # 0 ≤ λₜ₂ ⊥ -Jₜ(q⁺) v⁺ + ν 1 ≥ 0
        # 0 ≤ ν ⊥ λₙ - [λₜ₁ᵀ/μ  λₜ₂ᵀ/μ] 1 ≥ 0
        vt_next = efc_J_next[contact.efc_index + 1 : contact.efc_index + contact.dim] @ v_next
        z_k = z[z_index : z_index + z_dim(contact)]
        lambdan = z_k[0]
        lambdat = z_k[1:-1]
        nu = z_k[-1]
        friction = jp.tile(contact.friction, 2)
        constraints.append(efc_pos_next[contact.efc_index][None])
        constraints.append(jp.concatenate([vt_next, -vt_next]) + nu)
        constraints.append((lambdan - jp.sum(lambdat / friction))[None])

    return jp.concatenate(constraints) if constraints else jp.zeros((0,))

  local_complementarity = LocalComplementarityConstraint()
  if nz:
    local_complementarity.add_ctrl_ineq_complementarity(
        ctrl_index=np.arange(nu, nu + nz),
        ineq_index=np.arange(0, nz),
    )

  u_lb, u_ub = _check_u_bounds(u_bounds, nu)
  z_lb = np.zeros(nz)
  z_ub = np.full(nz, np.inf)
  w_bounds = (np.concatenate([u_lb, z_lb]), np.concatenate([u_ub, z_ub]))

  def stage_inequality(x: jax.Array, w: jax.Array) -> jax.Array:
    return local_inequality(x, w, x)

  inequality = local_inequality if contact_at_next_state else stage_inequality

  return dynamics, inequality, local_complementarity, w_bounds


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
  friction: jax.Array


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
  contact_friction = data._impl.contact.friction

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

  d = mjx.make_data(m, impl=m.impl)

  assert x.size == m.nq + m.nv + m.na
  assert u.size == m.nu
  assert efc_force.size == d._impl.nefc

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


def _check_u_bounds(u_bounds: Bounds | None, nu: int):
  if u_bounds is None:
    u_bounds = np.full(nu, -np.inf), np.full(nu, np.inf)
  if u_bounds[0].size != nu or u_bounds[1].size != nu:
    raise ValueError('u lower and upper bounds have incorrect size')
  return u_bounds
