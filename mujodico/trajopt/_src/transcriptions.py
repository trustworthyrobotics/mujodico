from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
import jax
import jax.numpy as jp
from jax.typing import ArrayLike
from mujodico import mjx
from mujodico.mjx._src import math
from mujodico.mjx._src import solver_sap
from mujodico.mjx._src import support
from mujodico.trajopt._src.types import Bounds, LocalComplementarityConstraint
import numpy as np


def _disable_control_clipping(model: mjx.Model) -> mjx.Model:
  """Control clipping is disabled to avoid the plant saturating controls.
  It is up to the optimizer to enforce control bounds explicitly."""
  return model.tree_replace({
      'opt.disableflags': model.opt.disableflags | mjx.DisableBit.CLAMPCTRL
  })


def _float32_model(model: mjx.Model) -> mjx.Model:
  return jax.tree.map(
      lambda a: a.astype(jp.float32) if jp.issubdtype(a.dtype, jp.floating) else a,
      model,
  )


def _seed_next_velocity(
    model: mjx.Model, data: mjx.Data, x_next: jax.Array | None,
) -> mjx.Data:
  if x_next is None or support.contact_approx(model) == support.ContactApproxType.MUJOCO:
    return data
  velocity = x_next[model.nq:model.nq + model.nv]
  return data.replace(qacc_warmstart=jax.lax.stop_gradient(
      (velocity - data.qvel) / model.opt.timestep
  ))


def _make_transcription_data(
    model: mjx.Model,
    planar_normal: ArrayLike | None,
    reference_state: ArrayLike | None,
) -> mjx.Data:
  """Choose fixed world-space contact charts once, before tracing dynamics."""
  data = mjx.make_data(model, impl=model.impl)
  if not data._impl.ncon:
    return data

  if not ((planar_normal is None) ^ (reference_state is None)):
    raise ValueError('Specify either planar_normal or reference_stat')

  if reference_state is not None:
    reference = np.asarray(reference_state)
    if (
        reference.shape != (model.nq + model.nv + model.na,)
        or not np.isfinite(reference).all()
    ):
      raise ValueError('reference_state must be a finite [qpos, qvel, act] state.')
    reference = jp.asarray(reference, dtype=data.qpos.dtype)
    data = data.replace(
        qpos=reference[:model.nq],
        qvel=reference[model.nq:model.nq + model.nv],
        act=reference[model.nq + model.nv:],
    )

  if planar_normal is not None:
    normal = np.asarray(planar_normal, dtype=float)
    if (
        normal.shape != (3,)
        or not np.isfinite(normal).all()
        or not np.any(normal)
    ):
      raise ValueError('planar_normal must be a finite nonzero vector of shape (3,).')
    normal = normal / np.max(np.abs(normal))
    normal /= np.linalg.norm(normal)
    normal = jp.asarray(normal, dtype=data.qpos.dtype)

  data = jax.jit(mjx.fwd_position)(model, data)
  contact = data._impl.contact
  if planar_normal is not None:
    rotation = math.make_frame(normal)[jp.array([1, 2, 0])].T
    rotations = jp.broadcast_to(rotation, contact.frame.shape)
  else:
    rotations = contact.frame[:, jp.array([1, 2, 0]), :].swapaxes(-1, -2)

  return data.tree_replace({'_impl.contact.frame_chart_rotation': rotations})


def shooting_transcription(
    model: mjx.Model,
    *,
    planar_normal: ArrayLike | None = None,
    reference_state: ArrayLike | None = None,
    mjx_float32: bool = False,
):
  """Builds a one-step shooting dynamics function with smoothed derivatives.

  The returned function takes a state ``x`` and control ``u``, and returns
  the next state ``x⁺``. Its value comes from a step with zero contact smoothing;
  its derivatives use the XML custom ``contact_smoothing`` value.

  Args:
    model: MJX model to transcribe.
    planar_normal: world-space normal to the plane of contact normals. Chooses
      a shared chart with its pole perpendicular to that plane.
    reference_state: finite [qpos, qvel, act] state used to center each contact
      chart. Mutually exclusive with planar_normal. With neither option, use
      the model's default state. Charts remain fixed for the returned dynamics.
      Reference charts move singularities away from the reference normals;
      they do not guarantee coverage of every future contact normal.
    mjx_float32: evaluate the MJX steps in float32 and return the next state
      in the caller's dtype.

  Returns:
    dynamics: ``dynamics(x, u, x_next=None)`` returns the predicted next state.
      When supplied, ``x_next`` seeds the contact solve with its velocity.
  """
  smooth_model = _disable_control_clipping(model)
  with jax.enable_x64(False) if mjx_float32 else nullcontext():
    if mjx_float32:
      smooth_model = _float32_model(smooth_model)
    model = support._with_contact_config(smooth_model, smoothing=0.0)
    data = _make_transcription_data(model, planar_normal, reference_state)

  def step(x, u, x_next):
    nq, nv = model.nq, model.nv
    d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)
    d = _seed_next_velocity(model, d, x_next)

    next_data = mjx.step(model, d)
    xnext = jp.concatenate((next_data.qpos, next_data.qvel, next_data.act))
    smooth_data = mjx.step(smooth_model, d)
    xnext_smooth = jp.concatenate((smooth_data.qpos, smooth_data.qvel, smooth_data.act))
    return xnext, xnext_smooth

  def dynamics(
      x: jax.Array, u: jax.Array, x_next: jax.Array | None = None,
  ) -> jax.Array:
    if mjx_float32:
      with jax.enable_x64(False):
        if x_next is not None:
          x_next = x_next.astype(jp.float32)
        xnext, xnext_smooth = step(x.astype(jp.float32), u.astype(jp.float32), x_next)
      xnext, xnext_smooth = xnext.astype(x.dtype), xnext_smooth.astype(x.dtype)
    else:
      xnext, xnext_smooth = step(x, u, x_next)
    return (
        jax.lax.stop_gradient(xnext)
        + xnext_smooth
        - jax.lax.stop_gradient(xnext_smooth)
    )

  return dynamics


def lifted_shooting_transcription(
    model: mjx.Model,
    u_bounds: Bounds | None = None,
    *,
    planar_normal: ArrayLike | None = None,
    reference_state: ArrayLike | None = None,
    mjx_float32: bool = False,
):
  """Builds dynamics and constraints with contact forces as lifted controls.

  The returned decision input ``w`` is ``[u, cf]``, where ``u`` is the
  actuator control and ``cf`` contains contact forces. The stage equality
  enforces that ``cf`` matches the forces computed by the forward dynamics.
  The stage inequality enforces the elliptic friction cone for each elliptic
  contact.

  Args:
    model: MJX model to transcribe.
    u_bounds: optional pair of lower and upper actuator control bounds.
              Defaults to unbounded controls.
    planar_normal: world-space normal to the plane of contact normals. Chooses
      a shared chart with its pole perpendicular to that plane.
    reference_state: finite [qpos, qvel, act] state used to center each contact
      chart. Mutually exclusive with planar_normal. With neither option, use
      the model's default state. Charts remain fixed for the returned functions.
      Reference charts move singularities away from the reference normals;
      they do not guarantee coverage of every future contact normal.
    mjx_float32: evaluate MJX dynamics and contact forces in float32. Cast
      predictions back to the caller's dtype before assembling residuals.

  Returns:
    dynamics: ``dynamics(x, w)`` returns the next state.
    stage_equality: constraint ``stage_equality(x, w, x_next=None) = 0``.
      When supplied, ``x_next`` seeds the contact solve with its velocity.
    stage_inequality: constraint ``stage_inequality(x, w) ≥ 0``.
    w_bounds: pair of lower and upper bounds for ``w``.
  """
  model = _disable_control_clipping(model)
  with jax.enable_x64(False) if mjx_float32 else nullcontext():
    if mjx_float32:
      model = _float32_model(model)
    data = _make_transcription_data(model, planar_normal, reference_state)

  cf_addresses, contacts = _extract_contact_indices(model, data)
  elliptic_contacts = [c for c in contacts if c.type == _Contact.Type.ELLIPTIC]

  nu = model.nu
  ncf = int(cf_addresses.size)

  def step1(x: jax.Array, u: jax.Array, x_next: jax.Array | None) -> jax.Array:
    nq, nv = model.nq, model.nv
    d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)
    d = _seed_next_velocity(model, d, x_next)
    d = mjx.forward(model, d)
    return d._impl.efc_force[cf_addresses]

  def step2(x: jax.Array, u: jax.Array, cf: jax.Array) -> jax.Array:
    efc_force = jp.zeros(data._impl.nefc)
    efc_force = efc_force.at[cf_addresses].set(cf)
    return _step_given_efc_force(model, data, x, u, efc_force)

  def unpack_control(w):
    assert w.size == nu + ncf
    return w[:nu], w[nu:]

  def dynamics(x: jax.Array, w: jax.Array) -> jax.Array:
    u, cf = unpack_control(w)
    if mjx_float32:
      with jax.enable_x64(False):
        x_next = step2(x.astype(jp.float32), u.astype(jp.float32), cf.astype(jp.float32))
      return x_next.astype(x.dtype)
    return step2(x, u, cf)

  def stage_equality(
      x: jax.Array, w: jax.Array, x_next: jax.Array | None = None,
  ) -> jax.Array:
    u, cf = unpack_control(w)
    if mjx_float32:
      with jax.enable_x64(False):
        if x_next is not None:
          x_next = x_next.astype(jp.float32)
        force = step1(x.astype(jp.float32), u.astype(jp.float32), x_next)
      return cf - force.astype(cf.dtype)
    return cf - step1(x, u, x_next)

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
    model: mjx.Model,
    u_bounds: Bounds | None = None,
    *,
    planar_normal: ArrayLike | None = None,
    reference_state: ArrayLike | None = None,
):
  """Builds a contact-implicit transcription.

  The returned decision input ``w`` is ``[u, z]``. ``u`` is the actuator
  control; ``z`` contains nonnegative contact force variables.
  Uses Anitescu convex formulation for pyramidal contact.
  Uses Stewart-Trinkle formulation for elliptic contact.
  Only ``conic`` and ``linear-elastic`` contact approximations are supported,
  and the model's ``contact_smoothing`` must be zero.
  With g = ϕ(q) + δt J(q) v⁺, the force complementarity is λ ⊥ g for
  conic contact and λ ⊥ λ + k g for linear-elastic contact. The latter
  is equivalent to the unilateral spring law λ = k max(-g, 0).

  Args:
    model: MJX model to transcribe.
    u_bounds: optional pair of lower and upper actuator control bounds.
              Defaults to unbounded controls.
    planar_normal: world-space normal to the plane of contact normals. Chooses
      a shared chart with its pole perpendicular to that plane.
    reference_state: finite [qpos, qvel, act] state used to center each contact
      chart. Mutually exclusive with planar_normal. With neither option, use
      the model's default state. Charts remain fixed for the returned functions.
      Reference charts move singularities away from the reference normals;
      they do not guarantee coverage of every future contact normal.

  Returns:
    dynamics: ``dynamics(x, w)`` returns the next state.
    local_inequality: constraint ``local_inequality(x, w, xnext) ≥ 0``.
    local_complementarity: local complementarity constraints.
    w_bounds: pair of lower and upper bounds for ``w``.
  """
  model = _disable_control_clipping(model)
  contact_approx = support.contact_approx(model)
  if contact_approx not in (
      support.ContactApproxType.CONIC,
      support.ContactApproxType.LINEAR_ELASTIC,
  ):
    raise ValueError(
        'contact_implicit_transcription supports only conic or linear-elastic '
        f'contact approximations; got {contact_approx.value}.'
    )
  if support.contact_smoothing(model) != 0:
    raise ValueError(
        'contact_implicit_transcription requires contact_smoothing == 0.'
    )
  data = _make_transcription_data(model, planar_normal, reference_state)

  cf_addresses, contacts = _extract_contact_indices(model, data)

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

  def contact_data(x: jax.Array) -> mjx.Data:
    nq, nv = model.nq, model.nv
    d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:])
    return mjx.fwd_position(model, d)

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
    return _step_given_efc_force(model, data, x, u, efc_force)

  def local_inequality(x: jax.Array, w: jax.Array, x_next: jax.Array) -> jax.Array:
    _, z = unpack_control(w)
    v_next = x_next[model.nq:model.nq+model.nv]

    d = contact_data(x)
    efc_J = d._impl.efc_J
    predicted_gap = d._impl.efc_pos + model.opt.timestep * (efc_J @ v_next)
    stiffness = d._impl.efc_k * solver_sap._STIFFNESS_SCALE

    def contact_residual(indices, force):
      gap = predicted_gap[indices]
      if contact_approx == support.ContactApproxType.LINEAR_ELASTIC:
        return force + stiffness[indices] * gap
      return gap

    constraints = []
    for z_index, contact in zip(z_indexes, contacts):
      if contact.type == _Contact.Type.FRICTIONLESS:
        # 0 ≤ λₙ ⊥ gₙ (conic) or λₙ + kₙ gₙ (linear-elastic) ≥ 0
        lambdan = z[z_index]
        constraints.append(contact_residual(contact.efc_index, lambdan)[None])

      elif contact.type == _Contact.Type.PYRAMIDAL:
        # 0 ≤ λₙ ⊥ gₙ (conic) or λₙ + kₙ gₙ (linear-elastic) ≥ 0
        indices = jp.arange(contact.efc_index, contact.efc_index + z_dim(contact))
        lambdan = z[z_index:z_index + z_dim(contact)]
        constraints.append(contact_residual(indices, lambdan))

      elif contact.type == _Contact.Type.ELLIPTIC:
        # 0 ≤ λₙ ⊥ gₙ (conic) or λₙ + kₙ gₙ (linear-elastic) ≥ 0
        # 0 ≤ λₜ₁ ⊥ Jₜ(q) v⁺ + ν / μ ≥ 0
        # 0 ≤ λₜ₂ ⊥ -Jₜ(q) v⁺ + ν / μ ≥ 0
        # 0 ≤ ν ⊥ λₙ - [λₜ₁ᵀ/μ  λₜ₂ᵀ/μ] 1 ≥ 0
        vt_next = efc_J[contact.efc_index + 1 : contact.efc_index + contact.dim] @ v_next
        z_k = z[z_index : z_index + z_dim(contact)]
        lambdan = z_k[0]
        lambdat = z_k[1:-1]
        nu = z_k[-1]
        friction = jp.tile(contact.friction, 2)
        constraints.append(contact_residual(contact.efc_index, lambdan)[None])
        constraints.append(jp.concatenate([vt_next, -vt_next]) + nu / friction)
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

  return dynamics, local_inequality, local_complementarity, w_bounds


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


def _extract_contact_indices(
    model: mjx.Model, data: mjx.Data | None = None
) -> tuple[np.ndarray, list[_Contact]]:
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
  if data is None:
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
    m: mjx.Model,
    data: mjx.Data,
    x: jax.Array,
    u: jax.Array,
    efc_force: jax.Array,
) -> jax.Array:
  """Computes the mapping (x, u, λ) ↦ x⁺ assuming that the constraint force λ is known.
  For implicit Euler integration, The mapping satisfies:
    M(q) (v⁺ - v) / δt + c(q⁺, v⁺) = τ(q⁺, v⁺, u) + J(q)ᵀ λ
    q⁺ = q ⊕ δt v⁺
  """

  assert x.size == m.nq + m.nv + m.na
  assert u.size == m.nu
  assert efc_force.size == data._impl.nefc

  d = data.replace(
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
      'qacc': None,  # Setting (qacc = None) forces integration from
                     # qfrc_constraint, see mjx.euler and mjx.implicit.
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
