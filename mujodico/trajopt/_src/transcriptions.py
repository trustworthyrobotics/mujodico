from dataclasses import dataclass
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

  addresses, socs = _extract_contact_indices(model, data)
  ncf = int(addresses.size)

  def step1(x: jax.Array, u: jax.Array) -> jax.Array:
    nq, nv = model.nq, model.nv
    d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)
    d = mjx.forward(model, d)
    return d._impl.efc_force[addresses]

  def step2(x: jax.Array, u: jax.Array, cf: jax.Array) -> jax.Array:
    efc_force = jp.zeros(data._impl.nefc)
    efc_force = efc_force.at[addresses].set(cf)
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
    constraints = jp.zeros(len(socs))
    for i, soc in enumerate(socs):
      normal = cf[soc.start_index]
      tangent = cf[soc.start_index+1:soc.start_index+soc.dim] / soc.friction
      constraints = constraints.at[i].set(normal * normal - tangent.dot(tangent))
    return constraints

  cf_lb = np.zeros(ncf)
  cf_ub = np.full(ncf, jp.inf)
  for soc in socs:
    cf_lb[soc.start_index+1:soc.start_index+soc.dim] = -np.inf

  w_bounds = (np.concatenate([u_lb, cf_lb]), np.concatenate([u_ub, cf_ub]))

  return dynamics, stage_equality, stage_inequality, w_bounds


@dataclass(frozen=True)
class _SOC:
  start_index: int
  dim: int
  friction: jax.Array


def _extract_contact_indices(
    mjx_model: mjx.Model, mjx_data: mjx.Data
  ) -> tuple[np.ndarray, list[_SOC]]:
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
    soc = _SOC(
        start_index=int(start_index),
        dim=int(dim),
        friction=conic_friction[i, :dim-1],
    )
    socs.append(soc)

    if not np.all(soc.friction > 1e-6):
      raise ValueError('Near zero friction not allowed')

  return efc_contact_addresses, socs


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
