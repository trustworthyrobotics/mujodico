"""
Solves next step contact force and velocity/acceleration for a log-barrier
contact approximation. Under the contact approximaiton, the dynamcis are
  M(q) (v⁺ - v) / δt + c(q⁺, v⁺) = τ(q⁺, v⁺, u) + J(q)ᵀ λ
  q⁺ = q ⊕ δt v⁺

By defining the effective mass matrix
  Mₑ = M(q) - d[τ(q, v, u) - c(q, v)] / dv δt
and the mass matrix free-motion velocity product
  Mₑ v* = Mₑ v + δt [τ(q, v, u) - c(q, v)]
we can write the dynamics as
  Mₑ v⁺ - Mₑ v* = J(q)ᵀ λ δt
  q⁺ = q ⊕ δt v⁺
"""

from abc import ABC, abstractmethod
import jax
from jax import numpy as jp
from mujodico.mjx._src import constraint
from mujodico.mjx._src import derivative
from mujodico.mjx._src import forward
from mujodico.mjx._src import io
from mujodico.mjx._src import support
# pylint: disable=g-importing-member
from mujodico.mjx._src.dataclasses import PyTreeNode
from mujodico.mjx._src.types import ConstraintType
from mujodico.mjx._src.types import ContactApproxType
from mujodico.mjx._src.types import Data
from mujodico.mjx._src.types import DataJAX
from mujodico.mjx._src.types import DisableBit
from mujodico.mjx._src.types import IntegratorType
from mujodico.mjx._src.types import Model
from mujodico.mjx._src.types import OptionJAX
# pylint: enable=g-importing-member


class SapConstraint(PyTreeNode, ABC):
  @abstractmethod
  def calc_cost(self, vc: jax.Array) -> jax.Array:
    pass

  @abstractmethod
  def calc_cost_gradient(self, vc: jax.Array) -> jax.Array:
    pass

  @abstractmethod
  def calc_cost_hessian(self, vc: jax.Array) -> jax.Array:
    pass


class SapFixedConstraint(SapConstraint):
  def __init__(
      self,
      phi0: jax.Array,
      dt: float,
      delassus_estimation: jax.Array
    ):
    pass

  def calc_cost(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()

  def calc_cost_gradient(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()

  def calc_cost_hessian(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()


class SapLimitConstraint(SapConstraint):
  def __init__(
      self,
      phi0: jax.Array,
      dt: float,
      delassus_estimation: jax.Array
    ):
    pass

  def calc_cost(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()

  def calc_cost_gradient(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()

  def calc_cost_hessian(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()


class SapFrictionConstraint(SapConstraint):
  def __init__(
      self,
      dt: float,
      frictionloss: jax.Array
    ):
    pass

  def calc_cost(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()

  def calc_cost_gradient(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()

  def calc_cost_hessian(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()


def _smooth_relu(x, epsilon):
  assert epsilon > 0
  return jp.where(
    x <= -epsilon,
    0.0,
    jp.where(
      x < epsilon,
      (x + epsilon) ** 2 / (4.0 * epsilon),
      x,
    ),
  )

def _smooth_relu_grad(x, epsilon):
  assert epsilon > 0
  return jp.where(
    x <= -epsilon,
    0.0,
    jp.where(
      x < epsilon,
      (x + epsilon) / (2.0 * epsilon),
      1.0,
    ),
  )


def _hunt_crossley_contact_force(phi, vn, k, d, epsilon1, epsilon2):
    """Compute the smoothed Hunt-Crossley normal contact force:
        f(φ, vₙ) = k g_ε₁(-φ) g_ε₂(1 - d vₙ),
    where g_ε(x) is a C1-smooth approximation of ReLU:
        g_ε(x) =
            0                    if x <= -ε,
            (x + ε)^2 / (4 ε)    if |x| < ε,
            x                    if x >= ε.
    """

    g1 = _smooth_relu(-phi, epsilon1)
    g2 = _smooth_relu(1.0 - d * vn, epsilon2)

    return k * g1 * g2


def _contact_normal_impulse(vn, phi0, k, d, dt, epsilon1, epsilon2):
  return dt * _hunt_crossley_contact_force(
    phi=phi0 + vn * dt,
    vn=vn,
    k=k,
    d=d,
    epsilon1=epsilon1,
    epsilon2=epsilon2,
  )


def _contact_normal_impulse_grad(vn, phi0, k, d, dt, epsilon1, epsilon2):
  x1 = -phi0 - dt * vn
  x2 = 1.0 - d * vn

  g1 = _smooth_relu(x1, epsilon1)
  g2 = _smooth_relu(x2, epsilon2)

  dg1_dvn = _smooth_relu_grad(x1, epsilon1) * -dt
  dg2_dvn = _smooth_relu_grad(x2, epsilon2) * -d

  return dt * k * (dg1_dvn * g2 + g1 * dg2_dvn)


def _contact_normal_impulse_integral(vn, phi0, k, d, dt, epsilon1, epsilon2):
  def term(s1, s2):
    A = -phi0 + s1 * epsilon1
    B = -dt
    C = 1 + s2 * epsilon2
    D = -d

    vn_thresh = jp.minimum(-A / B, -C / D)
    x = jp.minimum(vn, vn_thresh)

    return dt * k * (s1 * s2) / (16 * epsilon1 * epsilon2) * (
      (((((
          (B**2 * D**2 / 5)) * x
        + (A * B * D**2 + B**2 * C * D) / 2) * x
        + (A**2 * D**2 + 4 * A * B * C * D + B**2 * C**2) / 3) * x
        + (A**2 * C * D + A * B * C**2)) * x
        + (A**2 * C**2)) * x
    )

  return term(+1, +1) + term(+1, -1) + term(-1, +1) + term(-1, -1)


class SapNormalContactConstraint(SapConstraint):
  phi0: jax.Array
  dt: jax.Array
  stiffness: jax.Array
  hunt_crossley_damping: jax.Array

  @property
  def _params(self):
    return {
      'phi0': self.phi0,
      'dt': self.dt,
      'k': self.stiffness,
      'd': self.hunt_crossley_damping,
      'epsilon1': 0.01,
      'epsilon2': 0.1,
    }

  def calc_cost(self, vc: jax.Array) -> jax.Array:
    return -_contact_normal_impulse_integral(vn=vc, **self._params)

  def calc_cost_gradient(self, vc: jax.Array) -> jax.Array:
    return -_contact_normal_impulse(vn=vc, **self._params)

  def calc_cost_hessian(self, vc: jax.Array) -> jax.Array:
    return -_contact_normal_impulse_grad(vn=vc, **self._params)


class SapContactLaggedConstraint(SapConstraint):
  def __init__(
      self,
      phi0: float,
      vn0: float,
      dt: float,
      stiffness: float,
      hunt_crossley_damping: float,
      friction: jax.Array,
    ):
    pass

  def calc_cost(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()

  def calc_cost_gradient(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()

  def calc_cost_hessian(self, vc: jax.Array) -> jax.Array:
    raise NotImplementedError()


class Context(PyTreeNode):
  """Data updated during each solver iteration.

  Attributes:
    qvel: next step velocity                                (nv,)
    delta_qvel: step direction for qvel                     (nv,)
    qm: mass matrix                                         (nv, nv)
    qm_qvel_smooth: mass matrix times free-motion velocity  (nv,)
    efc_J: constraint Jacobian                              (nefc, nv)
    efc_pos: constraint position                            (nefc,)
    efc_masks: efc mask of each constraint
    sap_constraints: objects of type SapConstraint
  """

  qvel: jax.Array
  delta_qvel: jax.Array
  qm: jax.Array
  qm_qvel_smooth: jax.Array
  efc_J: jax.Array
  efc_pos: jax.Array
  efc_masks: list
  sap_constraints: list[jax.Array]

  @classmethod
  def create(cls, m: Model, d: Data) -> 'Context':
    qm, qm_qvel_smooth = cls._linearize_unconstrained_dynamics(m, d)

    efc_masks, sap_constraints = cls._build_constraints(m, d)

    warm_start = not m.opt.disableflags & DisableBit.WARMSTART
    qvel_init = d.qvel if warm_start else jp.zeros(m.nv)

    ctx = Context(
      qvel=qvel_init,
      delta_qvel=jp.zeros_like(qvel_init),
      qm=qm,
      qm_qvel_smooth=qm_qvel_smooth,
      efc_J=d._impl.efc_J,
      efc_pos=d._impl.efc_pos,
      efc_masks=efc_masks,
      sap_constraints=sap_constraints,
    )

    return ctx

  @staticmethod
  def _linearize_unconstrained_dynamics(m: Model, d: Data) -> tuple[jax.Array, jax.Array]:
    """Computes the effective mass matrix and its product with the free-motion velocity
    for the integrator type.
    """

    dt = m.opt.timestep

    qm = support.full_m(m, d) if support.is_sparse(m) else d._impl.M
    if m.opt.integrator == IntegratorType.IMPLICITFAST:
      qm -= derivative.deriv_smooth_vel(m, d) * dt

    qm_qvel_smooth = qm @ d.qvel + d.qfrc_smooth * dt

    return qm, qm_qvel_smooth

  @staticmethod
  def _build_constraints(m: Model, d: Data):
    dt = m.opt.timestep

    efc_type = constraint.make_efc_type(m)

    efc_masks = []
    constraints = []

    # Fixed constraint
    mask = (efc_type == ConstraintType.EQUALITY)
    if mask.any():
      efc_masks.append(mask)
      constraints.append(
        SapFixedConstraint(
          phi0=d._impl.efc_pos[mask],
          dt=dt,
          delassus_estimation=None,
        )
      )

    # Limit constraint
    mask = jp.logical_or(
      efc_type == ConstraintType.LIMIT_JOINT,
      efc_type == ConstraintType.LIMIT_TENDON
    )
    if mask.any():
      efc_masks.append(mask)
      constraints.append(
        SapLimitConstraint(
          phi0=d._impl.efc_pos[mask],
          dt=dt,
          delassus_estimation=None,
        )
      )

    # Friction constraint
    mask = jp.logical_or(
      efc_type == ConstraintType.FRICTION_DOF,
      efc_type == ConstraintType.FRICTION_TENDON
    )
    if mask.any():
      efc_masks.append(mask)
      constraints.append(
        SapFrictionConstraint(
          dt=dt,
          frictionloss=d._impl.efc_frictionloss[mask],
        )
      )

    # Normal contact constraint
    mask = jp.logical_or(
      efc_type == ConstraintType.CONTACT_FRICTIONLESS,
      efc_type == ConstraintType.CONTACT_PYRAMIDAL
    )
    if mask.any():
      efc_masks.append(mask)
      constraints.append(
        SapNormalContactConstraint(
          phi0=d._impl.efc_pos[mask],
          dt=dt,
          stiffness=d._impl.efc_k[mask],
          hunt_crossley_damping=d._impl.efc_b[mask],
        )
      )

    # Elliptic contact constraint
    contact = d._impl.contact
    for k in range(len(contact.efc_address)):
      idx = contact.efc_address[k]
      if efc_type[idx] != ConstraintType.CONTACT_ELLIPTIC:
        continue

      dim = contact.dim[k]
      mask = jp.zeros(efc_type.size, type=bool)
      mask = mask.at[jp.arange(idx, idx + dim)].set(1)
      efc_masks.append(mask)

      if m.opt._impl.contact_approx == ContactApproxType.LAGGED:
        constraints.append(
          SapContactLaggedConstraint(
            phi0=d._impl.efc_pos[mask],
            vn0=d._impl.efc_J[mask] @ d.qvel,
            dt=dt,
            stiffness=d._impl.efc_k[mask],
            hunt_crossley_damping=d._impl.efc_b[mask],
            friction=contact.friction[k][:dim-1],
          )
        )

      else:
        raise ValueError(f'Contact approximation {m.opt._impl.contact_approx} '
                         f'is not supported by SAP solver')

    return efc_masks, constraints


def _calc_cost(ctx: Context) -> jax.Array:
  qvel = ctx.qvel
  cost = 0.5 * qvel.dot(ctx.qm @ qvel) - qvel.dot(ctx.qm_qvel_smooth)

  for efc_mask, sap_constraint in zip(ctx.efc_masks, ctx.sap_constraints):
    vc = ctx.efc_J[efc_mask] @ qvel
    cost += sap_constraint.calc_cost(vc)

  return cost


def _calc_cost_gradient(ctx: Context) -> jax.Array:
  qvel = ctx.qvel
  grad = ctx.qm @ qvel - ctx.qm_qvel_smooth

  for efc_mask, sap_constraint in zip(ctx.efc_masks, ctx.sap_constraints):
    Jc = ctx.efc_J[efc_mask]
    vc = Jc @ qvel
    grad += Jc.T @ sap_constraint.calc_cost_gradient(vc)

  return grad


def _calc_cost_hessian(ctx: Context) -> jax.Array:
  qvel = ctx.qvel
  hess = ctx.qm

  for efc_mask, sap_constraint in zip(ctx.efc_masks, ctx.sap_constraints):
    Jc = ctx.efc_J[efc_mask]
    vc = Jc @ qvel

    cost_hess = sap_constraint.calc_cost_hessian(vc)
    if cost_hess.ndim == 1:
      hess += Jc.T @ (cost_hess[:, None] * Jc)
    else:
      hess += Jc.T @ cost_hess @ Jc

  return hess


def _calc_constraint_impulse(ctx: Context) -> jax.Array:
  qvel = ctx.qvel
  efc_impulse = jp.zeros_like(ctx.efc_pos)

  for efc_mask, sap_constraint in zip(ctx.efc_masks, ctx.sap_constraints):
    Jc = ctx.efc_J[efc_mask]
    vc = Jc @ qvel
    efc_impulse = efc_impulse.at[efc_mask].set(-sap_constraint.calc_cost_gradient(vc))

  return efc_impulse


def _calc_step_direction(ctx: Context) -> Context:
  H = _calc_cost_hessian(ctx)
  g = _calc_cost_gradient(ctx)

  choH = jax.scipy.linalg.cho_factor(H)
  delta_qvel = jax.scipy.linalg.cho_solve(choH, -g)

  return ctx.replace(delta_qvel=delta_qvel)


def _eval_line_search_derivatives(
    ctx: Context, alpha: float
) -> tuple[jax.Array, jax.Array]:
  """Evaluates dℓ(v + α Δv) / dα and d²ℓ(v + α Δv) / dα²
  """
  qvel = ctx.qvel

  delta_qvel = ctx.delta_qvel
  delta_p = ctx.qm @ delta_qvel

  dell_dalpha = delta_p.dot(qvel + alpha * delta_qvel) - delta_qvel.dot(ctx.qm_qvel_smooth)

  d2ell_dalpha2 = delta_qvel.dot(delta_p)

  for efc_mask, sap_constraint in zip(ctx.efc_masks, ctx.sap_constraints):
    Jc = ctx.efc_J[efc_mask]
    vc = Jc @ (qvel + alpha * delta_qvel)
    delta_vc = Jc @ delta_qvel

    dell_dalpha += delta_vc.dot(sap_constraint.calc_cost_gradient(vc))

    cost_hess = sap_constraint.calc_cost_hessian(vc)
    if cost_hess.ndim == 1:
      d2ell_dalpha2 += delta_vc.dot(cost_hess * delta_vc)
    else:
      d2ell_dalpha2 += delta_vc.dot(cost_hess @ delta_vc)

  return dell_dalpha, d2ell_dalpha2


def _perform_line_search(
    ctx: Context,
    ls_tolerance: float,
    ls_iter: int,
) -> jax.Array:
  """Finds α where dℓ(v + α Δv) / dα = 0
  """

  alpha_lb = 0.0
  alpha_ub = 1.5

  g_ub, _ = _eval_line_search_derivatives(ctx, alpha_ub)

  def no_line_search(_):
    return alpha_ub

  def do_line_search(_):
    alpha = 0.5 * (alpha_lb + alpha_ub)

    g, dg = _eval_line_search_derivatives(ctx, alpha)

    carry = (
        0,                      # iteration
        alpha_lb,               # lower bracket
        alpha_ub,               # upper bracket
        alpha,                  # current alpha
        alpha_ub - alpha_lb,    # previous step
        alpha_ub - alpha_lb,    # current step
        g,                      # ell'(alpha)
        dg,                     # ell''(alpha)
    )

    def cond_fn(carry):
      i, _, _, _, _, _, g, _ = carry
      return (i < ls_iter) & (jp.abs(g) >= ls_tolerance)

    def body_fn(carry):
      (
          i,
          alpha_lb,
          alpha_ub,
          alpha,
          dalpha_old,
          dalpha,
          g,
          dg,
      ) = carry

      # Would Newton leave the bracket?
      newton_out_of_bracket = (
          ((alpha - alpha_ub) * dg - g)
          * ((alpha - alpha_lb) * dg - g)
          > 0.0
      )

      # Would Newton fail to make sufficient progress?
      newton_too_slow = (2.0 * jp.abs(g) > jp.abs(dalpha_old * dg))

      # Newton is unreliable when the second derivative is tiny.
      derivative_too_small = jp.abs(dg) < 1e-6

      use_bisection = newton_out_of_bracket | newton_too_slow | derivative_too_small

      # Bisection step.
      bisect_dalpha = 0.5 * (alpha_ub - alpha_lb)
      bisect_alpha = alpha_lb + bisect_dalpha

      # Newton step.
      newton_dalpha = g / dg
      newton_alpha = alpha - newton_dalpha

      # Select the step.
      next_dalpha = jp.where(use_bisection, bisect_dalpha, newton_dalpha)
      next_alpha = jp.where(use_bisection, bisect_alpha, newton_alpha)

      next_g, next_dg = _eval_line_search_derivatives(ctx, next_alpha)

      # Maintain g(alpha_lb) < 0 and g(alpha_ub) >= 0.
      next_alpha_lb = jp.where(next_g < 0.0, next_alpha, alpha_lb)
      next_alpha_ub = jp.where(next_g < 0.0, alpha_ub, next_alpha)

      return (
          i + 1,
          next_alpha_lb,
          next_alpha_ub,
          next_alpha,
          dalpha,
          next_dalpha,
          next_g,
          next_dg,
      )

    carry = jax.lax.while_loop(cond_fn, body_fn, carry)
    return carry[3]

  return jax.lax.cond(
      g_ub <= 0.0,
      no_line_search,
      do_line_search,
      operand=None,
  )


def _solve_impl(
    ctx: Context,
    tolerance: float,
    iter: int,
    ls_tolerance: float,
    ls_iter: int,
) -> Context:

  def cond_fn(carry):
    ctx, i = carry
    return ((i < iter) & (jp.linalg.norm(ctx.delta_qvel, ord=jp.inf) > tolerance)) | (i == 0)

  def body_fn(carry):
    ctx, i = carry

    ctx = _calc_step_direction(ctx)

    alpha = _perform_line_search(
        ctx,
        ls_tolerance=ls_tolerance,
        ls_iter=ls_iter,
    )

    ctx = ctx.replace(qvel=ctx.qvel + alpha * ctx.delta_qvel)

    return ctx, i + 1

  ctx, _ = jax.lax.while_loop(cond_fn, body_fn, (ctx, 0))

  return ctx


def solve(m: Model, d: Data) -> Data:
  if not isinstance(m.opt._impl, OptionJAX):
    raise ValueError('solve requires JAX backend implementation.')

  if m.opt.integrator == IntegratorType.RK4:
    raise ValueError('SAP solver does not support RK4 integrator.')

  ctx = Context.create(m, d)

  ctx = _solve_impl(
    ctx=ctx,
    tolerance=m.opt.tolerance,
    iter=m.opt.iterations,
    ls_tolerance=m.opt.ls_tolerance,
    ls_iter=m.opt.ls_iterations,
  )

  dt = m.opt.timestep

  qacc = (ctx.qvel - d.qvel) / dt
  efc_force = _calc_constraint_impulse(ctx) / dt
  qfrc_constraint = ctx.efc_J.T @ efc_force

  d = d.tree_replace({
      'qacc': qacc,
      '_impl.efc_force': efc_force,
      'qfrc_constraint': qfrc_constraint,
  })

  return d
