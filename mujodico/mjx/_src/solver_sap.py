"""
Solves next step velocity and constraint impulse, where the contact
force satisfies e.g. e.g. the Hunt-Crossley model. The dynamcis satisfy
    M(q) (v⁺ - v) / δt + c(q⁺, v⁺) = τ(q⁺, v⁺, u) + J(q)ᵀ γ
    q⁺ = q ⊕ δt v⁺
where γ is the constraint impulse.

By defining the effective mass matrix
    A = M(q) - d[τ(q, v, u) - c(q, v)] / dv δt
and the mass matrix times free-motion velocity as
    A v* = A v + δt [τ(q, v, u) - c(q, v)]
we can write the dynamics as
    A v⁺ - A v* = J(q)ᵀ γ
    q⁺ = q ⊕ δt v⁺

The next step velocity v⁺ is obtained by solving a convex program
    minimize ½ (v⁺ - v*)ᵀ A (v⁺ - v*) + ℓ_c(v_c)
with v_c := J(q) v⁺.
The convex program has optimality
    A (v⁺ - v*) + (∂ℓ_c / ∂v⁺) = 0
    ⟹ A (v⁺ - v*) + J(q)ᵀ (∂ℓ_c / ∂v_c) = 0
Hence the constraint impulse γ and constraint cost ℓ_c are related by
    γ = -∂ℓ_c / ∂v_c
It turns out that for some contact constitutive relation γ = γ(v_c),
Hunt-Crossley contact model included, ℓ_c is convex. Furthermore,
since A is positive-definite, the convex program is strongly convex.
"""

from abc import ABC, abstractmethod
from functools import partial
import jax
from jax import numpy as jp
from mujodico.mjx._src import collision_driver
from mujodico.mjx._src import constraint
from mujodico.mjx._src import derivative
from mujodico.mjx._src import solver_sap_util
from mujodico.mjx._src import support
# pylint: disable=g-importing-member
from mujodico.mjx._src.dataclasses import PyTreeNode
from mujodico.mjx._src.types import ConeType
from mujodico.mjx._src.types import ConstraintType
from mujodico.mjx._src.types import ContactApproxType
from mujodico.mjx._src.types import Data
from mujodico.mjx._src.types import DisableBit
from mujodico.mjx._src.types import IntegratorType
from mujodico.mjx._src.types import Model
from mujodico.mjx._src.types import OptionJAX
# pylint: enable=g-importing-member
import numpy as np


_SIGNED_DIST_TOL = 1e-2   # Unit: m
_BOUNCE_SPEED_TOL = 1e-3  # Unit: m/s
_STICTION_TOL = 1e-3      # Unit: m/s or rad/s
_HC_STIFFNESS_SCALE = 1e2  # Scale from mujoco stiffness to Hunt-Crossley stiffness
_HC_DAMPING_SCALE = 1e-1   # Scale from mujoco damping to Hunt-Crossley damping


class SapConstraint(PyTreeNode, ABC):
  dt: jax.Array

  @abstractmethod
  def calc_impulse(self, vc: jax.Array) -> jax.Array:
    pass

  @abstractmethod
  def calc_impulse_grad(self, vc: jax.Array) -> jax.Array:
    pass


class SapFixedConstraint(SapConstraint):
  phi0: jax.Array
  delassus_estimation: jax.Array

  def __post_init__(self):
    assert self.phi0.shape == self.delassus_estimation.shape

  @property
  def _R_inv(self):
    beta = 0.1
    R = (beta / (2 * jp.pi))**2 * jax.lax.stop_gradient(self.delassus_estimation)
    return jp.minimum(1 / R, 1e6)

  def calc_impulse(self, vc: jax.Array) -> jax.Array:
    assert vc.shape == self.phi0.shape
    return -self._R_inv * (vc + self.phi0 / self.dt)

  def calc_impulse_grad(self, vc: jax.Array) -> jax.Array:
    assert vc.shape == self.phi0.shape
    return -self._R_inv


class SapLimitConstraint(SapConstraint):
  phi0: jax.Array
  delassus_estimation: jax.Array

  def __post_init__(self):
    assert self.phi0.shape == self.delassus_estimation.shape

  @property
  def _R_inv(self):
    beta = 0.1
    R = (beta / (2 * jp.pi))**2 * jax.lax.stop_gradient(self.delassus_estimation)
    return jp.minimum(1 / R, 1e6)

  def calc_impulse(self, vc: jax.Array) -> jax.Array:
    assert vc.shape == self.phi0.shape
    return -self._R_inv * jp.minimum(vc + self.phi0 / self.dt, 0.0)

  def calc_impulse_grad(self, vc: jax.Array) -> jax.Array:
    assert vc.shape == self.phi0.shape
    return -self._R_inv * (vc + self.phi0 / self.dt < 0)


class SapFrictionConstraint(SapConstraint):
  frictionloss: jax.Array

  def __post_init__(self):
    assert self.frictionloss.ndim == 1

  def calc_impulse(self, vc: jax.Array) -> jax.Array:
    assert vc.shape == self.frictionloss.shape
    return (
      self.dt * self.frictionloss *
      -jax.vmap(
        lambda x: solver_sap_util.soft_normalize(x, epsilon=_STICTION_TOL)
      )(vc).ravel()
    )

  def calc_impulse_grad(self, vc: jax.Array) -> jax.Array:
    assert vc.shape == self.frictionloss.shape
    return (
      self.dt * self.frictionloss *
      -jax.vmap(
        lambda x: solver_sap_util.soft_normalize_grad(x, epsilon=_STICTION_TOL)
      )(vc).ravel()
    )


class SapLogbarrierNormalContactConstraint(SapConstraint):
  phi0: jax.Array
  tau: jax.Array
  dt: jax.Array
  contact_smoothing: jax.Array

  def __post_init__(self):
    assert self.phi0.shape == self.tau.shape
    assert self.dt.ndim == 0
    assert jp.isscalar(self.contact_smoothing)

  @property
  def _delta(self):
    return self.contact_smoothing

  def calc_impulse(self, vc: jax.Array) -> jax.Array:
    assert vc.shape == self.phi0.shape

    x = vc + self.phi0 / (self.dt + self.tau)
    eps = self.contact_smoothing
    delta = self._delta
    k = eps / delta / delta

    return jp.where(
      x >= delta,
      eps / x,
      eps / delta - k * (x - delta),
    )

  def calc_impulse_grad(self, vc: jax.Array) -> jax.Array:
    assert vc.shape == self.phi0.shape

    x = vc + self.phi0 / (self.dt + self.tau)
    eps = self.contact_smoothing
    delta = self._delta
    k = eps / delta / delta

    return jp.where(
      x >= delta,
      -eps / x**2,
      -k,
    )


class SapNormalContactConstraint(SapConstraint):
  phi0: jax.Array
  stiffness: jax.Array
  hunt_crossley_damping: jax.Array

  def __post_init__(self):
    assert self.phi0.shape == self.stiffness.shape
    assert self.phi0.shape == self.hunt_crossley_damping.shape

  @property
  def _params(self):
    return {
      'phi0': self.phi0,
      'dt': self.dt,
      'k': self.stiffness,
      'd': self.hunt_crossley_damping,
      'epsilon1': _SIGNED_DIST_TOL,
      'epsilon2': _BOUNCE_SPEED_TOL,
    }

  def calc_impulse(self, vc: jax.Array) -> jax.Array:
    assert vc.shape == self.phi0.shape
    return solver_sap_util.contact_normal_impulse(vn=vc, **self._params)

  def calc_impulse_grad(self, vc: jax.Array) -> jax.Array:
    assert vc.shape == self.phi0.shape
    return solver_sap_util.contact_normal_impulse_grad(vn=vc, **self._params)


class SapContactLaggedConstraint(SapConstraint):
  phi0: jax.Array
  vn0: jax.Array
  stiffness: jax.Array
  hunt_crossley_damping: jax.Array
  friction: jax.Array

  def __post_init__(self):
    assert self.phi0.ndim == 0
    assert self.vn0.ndim == 0
    assert self.stiffness.ndim == 0
    assert self.hunt_crossley_damping.ndim == 0
    assert self.friction.ndim == 1

  @property
  def _params_n(self):
    return {
      'phi0': self.phi0,
      'dt': self.dt,
      'k': self.stiffness,
      'd': self.hunt_crossley_damping,
      'epsilon1': _SIGNED_DIST_TOL,
      'epsilon2': _BOUNCE_SPEED_TOL,
    }

  @property
  def gamma_n0(self):
    return solver_sap_util.hunt_crossley_contact_force(
      phi=self.phi0,
      vn=self.vn0,
      k=self.stiffness,
      d=self.hunt_crossley_damping,
      epsilon1=_SIGNED_DIST_TOL,
      epsilon2=_BOUNCE_SPEED_TOL,
    ) * self.dt

  @property
  def _blocks_t(self):
    blocks = [(1, 3, self.friction[0])]

    if self.friction.size >= 3:
      blocks.append((3, 4, self.friction[2]))

    if self.friction.size >= 5:
      blocks.append((4, 6, self.friction[3]))

    return blocks

  def calc_impulse(self, vc: jax.Array) -> jax.Array:
    assert vc.size == self.friction.size + 1

    gamma = jp.zeros(vc.size)

    gamma = gamma.at[0].set(
      solver_sap_util.contact_normal_impulse(vn=vc[0], **self._params_n)
    )

    gamma_n0 = self.gamma_n0
    for start, end, friction in self._blocks_t:
      gamma = gamma.at[start:end].set(
        friction * gamma_n0 *
        -solver_sap_util.soft_normalize(vc[start:end], epsilon=_STICTION_TOL)
      )

    return gamma

  def calc_impulse_grad(self, vc: jax.Array) -> jax.Array:
    assert vc.size == self.friction.size + 1

    grad = jp.zeros((vc.size, vc.size))

    grad = grad.at[0, 0].set(
      solver_sap_util.contact_normal_impulse_grad(vn=vc[0], **self._params_n)
    )

    gamma_n0 = self.gamma_n0
    for start, end, friction in self._blocks_t:
      grad = grad.at[start:end, start:end].set(
        friction * gamma_n0 *
        -solver_sap_util.soft_normalize_grad(vc[start:end], epsilon=_STICTION_TOL)
      )

    return grad


class Context(PyTreeNode):
  """Data updated during each solver iteration.

  Attributes:
    qvel: next step velocity                                (nv,)
    delta_qvel: step direction for qvel                     (nv,)
    qM: mass matrix                                         (nv, nv)
    qM_qvel_smooth: mass matrix times free-motion velocity  (nv,)
    efc_J: constraint Jacobian                              (nefc, nv)
    efc_pos: constraint position                            (nefc,)
    sap_constraints: objects of type SapConstraint
    efc_masks: efc mask of each constraint
  """

  qvel: jax.Array
  delta_qvel: jax.Array
  qM: jax.Array
  qM_qvel_smooth: jax.Array
  efc_J: jax.Array
  efc_pos: jax.Array
  sap_constraints: list[SapConstraint]
  efc_masks: list[np.ndarray]

  @classmethod
  def create(cls, m: Model, d: Data) -> 'Context':
    qM, qM_qvel_smooth = cls._linearize_unconstrained_dynamics(m, d)

    cho_qM = jax.scipy.linalg.cho_factor(qM)

    sap_constraints, efc_masks = cls._build_constraints(m, d, cho_qM)

    warm_start = not m.opt.disableflags & DisableBit.WARMSTART
    qvel_init = (
      d.qvel
      if warm_start
      else jax.scipy.linalg.cho_solve(cho_qM, qM_qvel_smooth)
    )

    ctx = Context(
      qvel=qvel_init,
      delta_qvel=jp.zeros_like(qvel_init),
      qM=qM,
      qM_qvel_smooth=qM_qvel_smooth,
      efc_J=d._impl.efc_J,
      efc_pos=d._impl.efc_pos,
      sap_constraints=sap_constraints,
      efc_masks=efc_masks,
    )

    return ctx

  @staticmethod
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

  @staticmethod
  def _build_constraints(m: Model, d: Data, cho_qM: tuple[jax.Array, bool]):
    '''Create the SAP constraints and their correcponding efc masks.
    '''
    dt = m.opt.timestep

    efc_type = constraint.make_efc_type(m)

    efc_masks = []
    sap_constraints = []

    # Fixed constraint
    mask = np.nonzero(
      efc_type == ConstraintType.EQUALITY
    )[0]
    if mask.size:
      efc_masks.append(mask)

      Jc = d._impl.efc_J[mask]
      delassus = Jc @ jax.scipy.linalg.cho_solve(cho_qM, Jc.T)

      sap_constraints.append(
        SapFixedConstraint(
          phi0=d._impl.efc_pos[mask],
          delassus_estimation=jp.diag(delassus),
          dt=dt,
        )
      )

    # Limit constraint
    mask = np.nonzero(
      (efc_type == ConstraintType.LIMIT_JOINT) |
      (efc_type == ConstraintType.LIMIT_TENDON)
    )[0]
    if mask.size:
      efc_masks.append(mask)

      Jc = d._impl.efc_J[mask]
      delassus = Jc @ jax.scipy.linalg.cho_solve(cho_qM, Jc.T)

      sap_constraints.append(
        SapLimitConstraint(
          phi0=d._impl.efc_pos[mask],
          delassus_estimation=jp.diag(delassus),
          dt=dt,
        )
      )

    # Friction constraint
    mask = np.nonzero(
      (efc_type == ConstraintType.FRICTION_DOF) |
      (efc_type == ConstraintType.FRICTION_TENDON)
    )[0]
    if mask.size:
      efc_masks.append(mask)

      sap_constraints.append(
        SapFrictionConstraint(
          frictionloss=d._impl.efc_frictionloss[mask],
          dt=dt,
        )
      )

    # Normal contact constraint
    contact_approx = m.opt._impl.contact_approx
    mask = np.nonzero(
      (efc_type == ConstraintType.CONTACT_FRICTIONLESS) |
      (efc_type == ConstraintType.CONTACT_PYRAMIDAL)
    )[0]
    if mask.size:
      efc_masks.append(mask)

      if contact_approx == ContactApproxType.LAGGED:
        sap_constraints.append(
          SapNormalContactConstraint(
            phi0=d._impl.efc_pos[mask],
            stiffness=d._impl.efc_k[mask] * _HC_STIFFNESS_SCALE,
            hunt_crossley_damping=d._impl.efc_b[mask] * _HC_DAMPING_SCALE,
            dt=dt,
          )
        )

      elif contact_approx == ContactApproxType.LOG_BARRIER:
        sap_constraints.append(
          SapLogbarrierNormalContactConstraint(
            phi0=d._impl.efc_pos[mask],
            tau=d._impl.efc_b[mask] / jp.maximum(d._impl.efc_k[mask], 1e-8),
            dt=dt,
            contact_smoothing=m.opt._impl.contact_smoothing,
          )
        )

      else:
        raise ValueError(f'Contact approximation {contact_approx} not supported by SAP solver')

    # Elliptic contact constraint
    if m.opt.cone == ConeType.PYRAMIDAL:
      return sap_constraints, efc_masks

    contact_dim = collision_driver.make_condim(m)
    contact_efc_address = constraint.make_efc_address(m, contact_dim, efc_type)

    for k in range(len(contact_efc_address)):
      dim = contact_dim[k]
      efc_address = contact_efc_address[k]

      mask = np.arange(efc_address, efc_address + dim)
      efc_masks.append(mask)

      if contact_approx == ContactApproxType.LAGGED:
        sap_constraints.append(
          SapContactLaggedConstraint(
            phi0=d._impl.efc_pos[efc_address],
            vn0=d._impl.efc_J[efc_address] @ d.qvel,
            stiffness=d._impl.efc_k[efc_address] * _HC_STIFFNESS_SCALE,
            hunt_crossley_damping=d._impl.efc_b[efc_address] * _HC_DAMPING_SCALE,
            friction=d._impl.contact.friction[k][:dim-1],
            dt=dt,
          )
        )

      elif contact_approx == ContactApproxType.LOG_BARRIER:
        raise NotImplementedError()

      else:
        raise ValueError(f'Contact approximation {contact_approx} not supported by SAP solver')

    return sap_constraints, efc_masks


def _calc_constraint_impulse(ctx: Context) -> jax.Array:
  qvel = ctx.qvel
  efc_impulse = jp.zeros_like(ctx.efc_pos)

  for efc_mask, sap_constraint in zip(ctx.efc_masks, ctx.sap_constraints):
    Jc = ctx.efc_J[efc_mask]
    vc = Jc @ qvel
    efc_impulse = efc_impulse.at[efc_mask].set(sap_constraint.calc_impulse(vc))

  return efc_impulse


def _calc_cost_gradient(ctx: Context) -> jax.Array:
  qvel = ctx.qvel
  grad = ctx.qM @ qvel - ctx.qM_qvel_smooth

  for efc_mask, sap_constraint in zip(ctx.efc_masks, ctx.sap_constraints):
    Jc = ctx.efc_J[efc_mask]
    vc = Jc @ qvel

    cost_grad = -sap_constraint.calc_impulse(vc)
    grad += Jc.T @ cost_grad

  return grad


def _calc_cost_hessian(ctx: Context) -> jax.Array:
  qvel = ctx.qvel
  hess = ctx.qM

  for efc_mask, sap_constraint in zip(ctx.efc_masks, ctx.sap_constraints):
    Jc = ctx.efc_J[efc_mask]
    vc = Jc @ qvel

    cost_hess = -sap_constraint.calc_impulse_grad(vc)
    if cost_hess.ndim == 1:
      hess += Jc.T @ (cost_hess[:, None] * Jc)
    else:
      hess += Jc.T @ cost_hess @ Jc

  return hess


def _calc_step_direction(ctx: Context) -> Context:
  g = _calc_cost_gradient(ctx)
  H = _calc_cost_hessian(ctx)

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
  delta_p = ctx.qM @ delta_qvel

  dell_dalpha = delta_p.dot(qvel + alpha * delta_qvel) - delta_qvel.dot(ctx.qM_qvel_smooth)

  d2ell_dalpha2 = delta_qvel.dot(delta_p)

  for efc_mask, sap_constraint in zip(ctx.efc_masks, ctx.sap_constraints):
    Jc = ctx.efc_J[efc_mask]
    vc = Jc @ (qvel + alpha * delta_qvel)
    delta_vc = Jc @ delta_qvel

    cost_grad = -sap_constraint.calc_impulse(vc)
    dell_dalpha += delta_vc.dot(cost_grad)

    cost_hess = -sap_constraint.calc_impulse_grad(vc)
    if cost_hess.ndim == 1:
      d2ell_dalpha2 += delta_vc.dot(cost_hess * delta_vc)
    else:
      d2ell_dalpha2 += delta_vc.dot(cost_hess @ delta_vc)

  return dell_dalpha, d2ell_dalpha2


def _perform_line_search(
    ctx: Context,
    ls_tolerance: float,
    ls_iterations: int,
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
      return (i < ls_iterations) & (jp.abs(g) >= ls_tolerance)

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
    iterations: int,
    ls_tolerance: float,
    ls_iterations: int,
) -> Context:

  def cond_fn(carry):
    ctx, i = carry
    not_converged = jp.linalg.norm(ctx.delta_qvel, ord=jp.inf) > tolerance
    return (i == 0) | ((i < iterations) & not_converged)

  def body_fn(carry):
    ctx, i = carry

    ctx = _calc_step_direction(ctx)

    alpha = _perform_line_search(
        ctx,
        ls_tolerance=ls_tolerance,
        ls_iterations=ls_iterations,
    )

    ctx = ctx.replace(qvel=ctx.qvel + alpha * ctx.delta_qvel)

    return ctx, i + 1

  ctx, _ = jax.lax.while_loop(cond_fn, body_fn, (ctx, 0))

  return ctx


@partial(jax.custom_jvp, nondiff_argnums=(1,2,3,4))
def _solve(
    ctx: Context,
    tolerance: float,
    iterations: int,
    ls_tolerance: float,
    ls_iterations: int,
) -> jax.Array:

  ctx = _solve_impl(
    ctx=ctx,
    tolerance=tolerance,
    iterations=iterations,
    ls_tolerance=ls_tolerance,
    ls_iterations=ls_iterations,
  )
  qvel = ctx.qvel

  return qvel


@_solve.defjvp
def _solve_jvp(
    tolerance: float,
    iterations: int,
    ls_tolerance: float,
    ls_iterations: int,
    primals,
    tangents,
) -> tuple[jax.Array, jax.Array]:

  ctx = _solve_impl(
    ctx=primals[0],
    tolerance=tolerance,
    iterations=iterations,
    ls_tolerance=ls_tolerance,
    ls_iterations=ls_iterations,
  )
  qvel = ctx.qvel

  def f(*primals):
    return _calc_cost_gradient(ctx=primals[0].replace(qvel=qvel))

  _, df = jax.jvp(f, primals, tangents)

  H = _calc_cost_hessian(ctx)
  choH = jax.scipy.linalg.cho_factor(H)

  dqvel = jax.scipy.linalg.cho_solve(choH, -df)

  return qvel, dqvel


def solve(m: Model, d: Data) -> Data:
  if not isinstance(m.opt._impl, OptionJAX):
    raise ValueError('solve requires JAX backend implementation.')

  if m.opt.integrator == IntegratorType.RK4:
    raise ValueError('SAP solver does not support RK4 integrator.')

  ctx = Context.create(m, d)

  qvel = _solve(
    ctx=ctx,
    tolerance=m.opt.tolerance,
    iterations=m.opt.iterations,
    ls_tolerance=m.opt.ls_tolerance,
    ls_iterations=m.opt.ls_iterations,
  )
  ctx = ctx.replace(qvel=qvel)

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
