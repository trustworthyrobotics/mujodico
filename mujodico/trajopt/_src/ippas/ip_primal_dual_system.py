"""
Compute a primal-dual interior-point Newton step for the bounded OCP

  minimizeₓ,ᵤ ∑ₖ₌₀ᵀ⁻¹ ℓₖ(xₖ, uₖ) + ℓₜ(xₜ)          (t=T denotes terminal)
  subject to  h⁰(x₀) = 0
              xₖ₊₁ = fₖ(xₖ, uₖ)                   for k = 0,…,T−1
              hₖ(xₖ, uₖ) = 0                      for k = 0,…,T−1
              gˡₖ ≤ gₖ(xₖ, uₖ) ≤ gᵘₖ               for k = 0,…,T−1
              gˡₜ ≤ gₜ(xₜ) ≤ gᵘₜ.

Introduce slacks sₖ with gₖ - sₖ = 0 and gˡₖ≤ sₖ ≤gᵘₖ, including the terminal
slack sₜ (sf). The independent multipliers yₖ (terminal yf) enforce gₖ - sₖ = 0;
zˡₖ and zᵘₖ enforce the slack bounds.

For barrier parameter μ, let dₖ be the one-sided slack damping gradient
[IPOPT, sec. 3.7]:

  dₖ = kappa_d μ (1[gˡₖ finite] - 1[gᵘₖ finite]).

The slack-form Lagrangian, including this linear damping term, is

  L = ∑ₖ₌₀ᵀ⁻¹ [ℓₖ + λₖ₊₁ᵀ(fₖ-xₖ₊₁) + βₖᵀhₖ + yₖᵀ(gₖ-sₖ)
              - (zˡₖ)ᵀ(sₖ-gˡₖ) - (zᵘₖ)ᵀ(gᵘₖ-sₖ) + dₖᵀsₖ]
      + ℓₜ + yₜᵀ(gₜ-sₜ)
      - (zˡₜ)ᵀ(sₜ-gˡₜ) - (zᵘₜ)ᵀ(gᵘₜ-sₜ) + dₜᵀsₜ + λ₀ᵀh⁰(x₀).

Hxx, Hxu, Huu are the Lagrangian Hessian blocks, Hxx includes the terminal
block. A, B are the Jacobians of f; Ex, Eu those of h; and Gx, Gu those of g.
Gf is the terminal inequality Jacobian. E0 = ∇h⁰(x₀) must be square and nonsingular.

For barrier parameter μ, residuals are

  rˣₜ = ∇ℓₜ + Gfᵀyₜ - λₜ
  rˣₖ = ∇ₓℓₖ + Aₖᵀλₖ₊₁ + (Eˣₖ)ᵀβₖ + (Gˣₖ)ᵀyₖ - λₖ
  rᵘₖ = ∇ᵤℓₖ + Bₖᵀλₖ₊₁ + (Eᵘₖ)ᵀβₖ + (Gᵘₖ)ᵀyₖ
  rˢₖ = -yₖ - zˡₖ + zᵘₖ + dₖ
  rᶠₖ = fₖ-xₖ₊₁,   rᵉₖ = hₖ,   rᵍₖ = gₖ-sₖ,   rᵍₜ = gₜ-sₜ
  rᶜ˒ˡₖ = Zˡₖ(sₖ-gˡₖ) - μ1,   rᶜ˒ᵘₖ = Zᵘₖ(gᵘₖ-sₖ) - μ1
  r₀ = h⁰(x₀).

At k=0, replace -λ₀ in rˣ₀ with +E0ᵀλ₀.
Write Zˡ=diag(zˡ), Zᵘ=diag(zᵘ), Sˡ=diag(s-gˡ), Sᵘ=diag(gᵘ-s).
The terminal Newton equations are

  Hxxₜ δxₜ + Gfᵀδyₜ - δλₜ = -rˣₜ
  -δyₜ - δzˡₜ + δzᵘₜ = -rˢₜ
  Gf δxₜ - δsₜ = -rᵍₜ
  Zˡₜ δsₜ + Sˡₜ δzˡₜ = -rᶜ˒ˡₜ
  -Zᵘₜ δsₜ + Sᵘₜ δzᵘₜ = -rᶜ˒ᵘₜ.

For k=0,…,T−1 the stage Newton equations are

  Hxxₖ δxₖ + Hxuₖ δuₖ + Aₖᵀδλₖ₊₁ + (Eˣₖ)ᵀδβₖ + (Gˣₖ)ᵀδyₖ - δλₖ = -rˣₖ
  Hxuₖᵀδxₖ + Huuₖ δuₖ + Bₖᵀδλₖ₊₁ + (Eᵘₖ)ᵀδβₖ + (Gᵘₖ)ᵀδyₖ = -rᵘₖ
  -δyₖ - δzˡₖ + δzᵘₖ = -rˢₖ
  Aₖδxₖ + Bₖδuₖ - δxₖ₊₁ = -rᶠₖ
  Eˣₖδxₖ + Eᵘₖδuₖ = -rᵉₖ
  Gˣₖδxₖ + Gᵘₖδuₖ - δsₖ = -rᵍₖ
  Zˡₖδsₖ + Sˡₖδzˡₖ = -rᶜ˒ˡₖ
  -Zᵘₖδsₖ + Sᵘₖδzᵘₖ = -rᶜ˒ᵘₖ.

At k=0, replace -δλ₀ with +E0ᵀδλ₀ in state stationarity.
The initial equation is E0 δx₀ = -r₀. For delta_c > 0, only the stage-equality
row is regularized: Eˣₖδxₖ + Eᵘₖδuₖ - delta_c δβₖ = -rᵉₖ.

Eliminate δy internally using δyₖ = δzᵘₖ - δzˡₖ + rˢₖ, then eliminate
slacks and bound multipliers. The resulting LQR coefficients are

  Dₖ = (Sˡₖ)⁻¹Zˡₖ + (Sᵘₖ)⁻¹Zᵘₖ
  ξₖ = Dₖ rᵍₖ + (Sˡₖ)⁻¹rᶜ˒ˡₖ - (Sᵘₖ)⁻¹rᶜ˒ᵘₖ + rˢₖ
  Qₖ = Hxxₖ + (Gˣₖ)ᵀDₖGˣₖ,     Qₜ = Hxxₜ + GfᵀDₜGf
  Nₖ = Hxuₖ + (Gˣₖ)ᵀDₖGᵘₖ,     Rₖ = Huuₖ + (Gᵘₖ)ᵀDₖGᵘₖ
  qₖ = rˣₖ + (Gˣₖ)ᵀξₖ,         qₜ = rˣₜ + Gfᵀξₜ
  rₖ = rᵘₖ + (Gᵘₖ)ᵀξₖ.

EqualityConstrainedLQR solves this reduced system with c=r_f, e=r_e, and x0=-E0⁻¹r₀.
and the selected delta_c. Recover the initial multiplier using δλ₀=-E0⁻ᵀδλ̄₀;
later dynamics multipliers transfer unchanged. Recover the remaining increments using

  δsₖ = Gˣₖδxₖ + Gᵘₖδuₖ + rᵍₖ,   δsₜ = Gf δxₜ + rᵍₜ
  δzˡₖ = -(Sˡₖ)⁻¹(Zˡₖδsₖ + rᶜ˒ˡₖ)
  δzᵘₖ = (Sᵘₖ)⁻¹(Zᵘₖδsₖ - rᶜ˒ᵘₖ)
  δyₖ = δzᵘₖ - δzˡₖ + rˢₖ,

Finite bounds require strictly interior slacks and nonnegative multipliers.
Infinite-bound sides omit the corresponding multiplier and complementarity
equation: their inverse distances, contributions, and returned δz are zero.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import equality_constrained_lqr as lqr
from mujodico.trajopt._src.ippas.ip_iterate import (
  Iterate,
  IterateData,
  _mv,
  complementarity,
  damping,
  distances,
  max_abs,
)
from mujodico.trajopt._src.ippas.options import Options
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel
from mujodico.trajopt._src.types import Params


_STEPS: tuple[str, ...] = (
    'dx',
    'du',
    'dlmbda',
    'dbeta',
    'ds',
    'dsf',
    'dz_l',
    'dz_u',
    'dzf_l',
    'dzf_u',
    'dy',
    'dyf',
)


def _matvec(matrix, vector):
  return jax.vmap(jp.matmul)(matrix, vector)


def _scale_bound(inverse_distance, value):
  # Mask omitted equations before multiplying, even for nonfinite residuals.
  return inverse_distance * jp.where(inverse_distance != 0, value, 0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BoundCache:
  inverse_l: jax.Array
  inverse_u: jax.Array
  D: jax.Array


def _build_bound_cache(lower, upper, slack, z_l, z_u):
  finite_l = jp.isfinite(lower)
  finite_u = jp.isfinite(upper)
  inverse_l = jp.where(finite_l, 1 / jp.where(finite_l, slack - lower, 1), 0)
  inverse_u = jp.where(finite_u, 1 / jp.where(finite_u, upper - slack, 1), 0)
  D = _scale_bound(inverse_l, z_l) + _scale_bound(inverse_u, z_u)
  return _BoundCache(inverse_l=inverse_l, inverse_u=inverse_u, D=D)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _ReductionCache:
  stage: _BoundCache
  terminal: _BoundCache
  initial_lu: tuple[jax.Array, jax.Array]
  reduced: lqr.EqualityConstrainedLQR


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IpPrimalDualSystem:
  """Newton system with stagewise arrays and optional solved increments.

  Hxx and r_x have T+1 entries; other stage arrays have T entries. Stage
  equalities, stage inequalities, and terminal inequalities may each be empty.
  The terminal inequality count may differ from the stage inequality count.
  E0 must have shape (nx, nx) and be nonsingular; r0 has shape (nx,).
  """

  Hxx: jax.Array
  Hxu: jax.Array
  Huu: jax.Array
  A: jax.Array
  B: jax.Array
  Ex: jax.Array
  Eu: jax.Array
  E0: jax.Array
  Gx: jax.Array
  Gu: jax.Array
  g_l: jax.Array
  g_u: jax.Array
  Gf: jax.Array
  gf_l: jax.Array
  gf_u: jax.Array
  s: jax.Array
  sf: jax.Array
  z_l: jax.Array
  z_u: jax.Array
  zf_l: jax.Array
  zf_u: jax.Array
  r_x: jax.Array
  r_u: jax.Array
  r_f: jax.Array
  r_e: jax.Array
  r_g: jax.Array
  r_c_l: jax.Array
  r_c_u: jax.Array
  r_gf: jax.Array
  r_cf_l: jax.Array
  r_cf_u: jax.Array
  r_s: jax.Array
  r_sf: jax.Array
  r0: jax.Array
  dx: jax.Array | None = None
  du: jax.Array | None = None
  dlmbda: jax.Array | None = None
  dbeta: jax.Array | None = None
  ds: jax.Array | None = None
  dsf: jax.Array | None = None
  dz_l: jax.Array | None = None
  dz_u: jax.Array | None = None
  dzf_l: jax.Array | None = None
  dzf_u: jax.Array | None = None
  dy: jax.Array | None = None
  dyf: jax.Array | None = None
  delta_c: float | jax.Array = 0.0
  _cache: _ReductionCache | None = None

  @staticmethod
  def create(
      Hxx: jax.Array,
      Hxu: jax.Array,
      Huu: jax.Array,
      A: jax.Array,
      B: jax.Array,
      Ex: jax.Array,
      Eu: jax.Array,
      E0: jax.Array,
      Gx: jax.Array,
      Gu: jax.Array,
      g_l: jax.Array,
      g_u: jax.Array,
      Gf: jax.Array,
      gf_l: jax.Array,
      gf_u: jax.Array,
      s: jax.Array,
      sf: jax.Array,
      z_l: jax.Array,
      z_u: jax.Array,
      zf_l: jax.Array,
      zf_u: jax.Array,
      r_x: jax.Array,
      r_u: jax.Array,
      r_f: jax.Array,
      r_e: jax.Array,
      r_g: jax.Array,
      r_c_l: jax.Array,
      r_c_u: jax.Array,
      r_gf: jax.Array,
      r_cf_l: jax.Array,
      r_cf_u: jax.Array,
      r_s: jax.Array,
      r_sf: jax.Array,
      r0: jax.Array,
      delta_c: float | jax.Array = 0.0,
  ) -> 'IpPrimalDualSystem':
    assert jp.shape(delta_c) == ()
    T, nx, nu = Hxu.shape
    ne = r_e.shape[1]
    ng = r_g.shape[1]
    ngf = r_gf.shape[0]
    for value, shape in (
        (Hxx, (T + 1, nx, nx)),
        (Hxu, (T, nx, nu)),
        (Huu, (T, nu, nu)),
        (A, (T, nx, nx)),
        (B, (T, nx, nu)),
        (Ex, (T, ne, nx)),
        (Eu, (T, ne, nu)),
        (E0, (nx, nx)),
        (Gx, (T, ng, nx)),
        (Gu, (T, ng, nu)),
        (g_l, (T, ng)),
        (g_u, (T, ng)),
        (Gf, (ngf, nx)),
        (gf_l, (ngf,)),
        (gf_u, (ngf,)),
        (s, (T, ng)),
        (sf, (ngf,)),
        (z_l, (T, ng)),
        (z_u, (T, ng)),
        (zf_l, (ngf,)),
        (zf_u, (ngf,)),
        (r_x, (T + 1, nx)),
        (r_u, (T, nu)),
        (r_f, (T, nx)),
        (r_e, (T, ne)),
        (r_g, (T, ng)),
        (r_c_l, (T, ng)),
        (r_c_u, (T, ng)),
        (r_gf, (ngf,)),
        (r_cf_l, (ngf,)),
        (r_cf_u, (ngf,)),
        (r_s, (T, ng)),
        (r_sf, (ngf,)),
        (r0, (nx,)),
    ):
      assert value.shape == shape
    return IpPrimalDualSystem(
        Hxx=Hxx,
        Hxu=Hxu,
        Huu=Huu,
        A=A,
        B=B,
        Ex=Ex,
        Eu=Eu,
        E0=E0,
        Gx=Gx,
        Gu=Gu,
        g_l=g_l,
        g_u=g_u,
        Gf=Gf,
        gf_l=gf_l,
        gf_u=gf_u,
        s=s,
        sf=sf,
        z_l=z_l,
        z_u=z_u,
        zf_l=zf_l,
        zf_u=zf_u,
        r_x=r_x,
        r_u=r_u,
        r_f=r_f,
        r_e=r_e,
        r_g=r_g,
        r_c_l=r_c_l,
        r_c_u=r_c_u,
        r_gf=r_gf,
        r_cf_l=r_cf_l,
        r_cf_u=r_cf_u,
        r_s=r_s,
        r_sf=r_sf,
        r0=r0,
        delta_c=jp.asarray(delta_c, Hxx.dtype),
    )

  def replace_rhs(
      self,
      *,
      r_x=None,
      r_u=None,
      r_f=None,
      r_e=None,
      r_g=None,
      r_c_l=None,
      r_c_u=None,
      r_gf=None,
      r_cf_l=None,
      r_cf_u=None,
      r_s=None,
      r_sf=None,
      r0=None,
  ) -> 'IpPrimalDualSystem':
    updates = {}
    for name, value in dict(
        r_x=r_x,
        r_u=r_u,
        r_f=r_f,
        r_e=r_e,
        r_g=r_g,
        r_c_l=r_c_l,
        r_c_u=r_c_u,
        r_gf=r_gf,
        r_cf_l=r_cf_l,
        r_cf_u=r_cf_u,
        r_s=r_s,
        r_sf=r_sf,
        r0=r0,
    ).items():
      if value is None:
        continue
      value = jp.asarray(value)
      original = getattr(self, name)
      assert value.shape == original.shape
      assert value.dtype == original.dtype
      updates[name] = value
    return replace(
        self,
        **updates,
        dx=None,
        du=None,
        dlmbda=None,
        dbeta=None,
        ds=None,
        dsf=None,
        dz_l=None,
        dz_u=None,
        dzf_l=None,
        dzf_u=None,
        dy=None,
        dyf=None,
    )


def _bound_rhs(cache, r_g, r_c_l, r_c_u):
  return (
      cache.D * r_g
      + _scale_bound(cache.inverse_l, r_c_l)
      - _scale_bound(cache.inverse_u, r_c_u)
  )


def _recover_bound_steps(cache, ds, z_l, z_u, r_c_l, r_c_u):
  dz_l = -_scale_bound(cache.inverse_l, z_l * ds + r_c_l)
  dz_u = _scale_bound(cache.inverse_u, z_u * ds - r_c_u)
  return dz_l, dz_u


def solve(prob: IpPrimalDualSystem) -> IpPrimalDualSystem:
  assert isinstance(prob, IpPrimalDualSystem)

  if prob._cache is None:
    stage = _build_bound_cache(prob.g_l, prob.g_u, prob.s, prob.z_l, prob.z_u)
    terminal = _build_bound_cache(
        prob.gf_l,
        prob.gf_u,
        prob.sf,
        prob.zf_l,
        prob.zf_u,
    )
    initial_lu = jax.scipy.linalg.lu_factor(prob.E0)
  else:
    stage = prob._cache.stage
    terminal = prob._cache.terminal
    initial_lu = prob._cache.initial_lu

  xi = _bound_rhs(stage, prob.r_g, prob.r_c_l, prob.r_c_u) + prob.r_s
  xif = _bound_rhs(terminal, prob.r_gf, prob.r_cf_l, prob.r_cf_u) + prob.r_sf
  GxT = prob.Gx.swapaxes(-1, -2)
  GuT = prob.Gu.swapaxes(-1, -2)
  rhs = dict(
      q=prob.r_x
        + jp.concatenate((
            _matvec(GxT, xi),
            (prob.Gf.T @ xif)[None],
        )),
      r=prob.r_u + _matvec(GuT, xi),
      c=prob.r_f,
      e=prob.r_e,
      x0=-jax.scipy.linalg.lu_solve(initial_lu, prob.r0),
  )

  if prob._cache is None:
    DGx = stage.D[..., None] * prob.Gx
    DGu = stage.D[..., None] * prob.Gu
    reduced = lqr.EqualityConstrainedLQR.create(
        Q=prob.Hxx
          + jp.concatenate((
              GxT @ DGx,
              (prob.Gf.T @ (terminal.D[:, None] * prob.Gf))[None],
          )),
        N=prob.Hxu + GxT @ DGu,
        R=prob.Huu + GuT @ DGu,
        A=prob.A,
        B=prob.B,
        Ex=prob.Ex,
        Eu=prob.Eu,
        delta_c=prob.delta_c,
        **rhs,
    )
  else:
    reduced = prob._cache.reduced.replace_rhs(**rhs)
  reduced = lqr.solve(reduced)

  # −δλ̄₀ = E₀ᵀδλ₀ in the original initial-state stationarity equation.
  dlmbda = reduced.lmbda.at[0].set(
      -jax.scipy.linalg.lu_solve(initial_lu, reduced.lmbda[0], trans=1)
  )

  ds = _matvec(prob.Gx, reduced.x[:-1]) + _matvec(prob.Gu, reduced.u) + prob.r_g
  dsf = prob.Gf @ reduced.x[-1] + prob.r_gf

  dz_l, dz_u = _recover_bound_steps(
      stage,
      ds,
      prob.z_l,
      prob.z_u,
      prob.r_c_l,
      prob.r_c_u,
  )
  dzf_l, dzf_u = _recover_bound_steps(
      terminal,
      dsf,
      prob.zf_l,
      prob.zf_u,
      prob.r_cf_l,
      prob.r_cf_u,
  )

  dy = dz_u - dz_l + prob.r_s
  dyf = dzf_u - dzf_l + prob.r_sf

  return replace(
      prob,
      dx=reduced.x,
      du=reduced.u,
      dlmbda=dlmbda,
      dbeta=reduced.beta,
      ds=ds,
      dsf=dsf,
      dz_l=dz_l,
      dz_u=dz_u,
      dzf_l=dzf_l,
      dzf_u=dzf_u,
      dy=dy,
      dyf=dyf,
      _cache=_ReductionCache(
          stage=stage,
          terminal=terminal,
          initial_lu=initial_lu,
          reduced=reduced,
      ),
  )


def evaluate(
    model: StagewiseOcpModel,
    iterate: 'Iterate',
    params: Params,
    mu: jax.Array,
    bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    options: Options,
) -> tuple[IpPrimalDualSystem, 'IterateData']:
  """Evaluate a Newton system and nonlinear data.

  Constraint residuals are evaluated once and shared with the system's RHS.
  The returned IterateData retains these nonlinear values when subsequent
  Newton solves replace their RHS for SOC or iterative refinement.
  """

  m = model
  it = iterate

  gx, gu = m.objective_gradient(it.x, it.u, params)
  A, B = m.dynamics_jacobian(it.x[:-1], it.u, it.x[1:], params)
  Ex, Eu = m.stage_equality_jacobian(it.x[:-1], it.u, params)
  Gx, Gu = m.stage_inequality_jacobian(it.x[:-1], it.u, params)
  Gf = m.terminal_inequality_jacobian(it.x[-1], params)
  E0 = m.initial_jacobian(it.x[0], params)

  # rˣₜ = ∇ℓₜ + Gfᵀyₜ - λₜ
  # rˣₖ = ∇ₓℓₖ + Aₖᵀλₖ₊₁ + (Eˣₖ)ᵀβₖ + (Gˣₖ)ᵀyₖ - λₖ
  # at k=0, replace -λ₀ in rˣ₀ with +E0ᵀλ₀.
  r_x = gx.at[:-1].add(_mv(A.swapaxes(-1, -2), it.lmbda[1:]))
  r_x = r_x.at[1:].add(-it.lmbda[1:])
  r_x = r_x.at[0].add(E0.T @ it.lmbda[0])
  r_x = r_x.at[:-1].add(
      _mv(Ex.swapaxes(-1, -2), it.beta) + _mv(Gx.swapaxes(-1, -2), it.y)
  )
  r_x = r_x.at[-1].add(Gf.T @ it.yf)

  # rᵘₖ = ∇ᵤℓₖ + Bₖᵀλₖ₊₁ + (Eᵘₖ)ᵀβₖ + (Gᵘₖ)ᵀyₖ
  r_u = gu + _mv(B.swapaxes(-1, -2), it.lmbda[1:])
  r_u += _mv(Eu.swapaxes(-1, -2), it.beta) + _mv(Gu.swapaxes(-1, -2), it.y)

  Hxx, Hxu, Huu = m.lagrangian_hessian(
      it.x,
      it.u,
      it.lmbda[0],
      it.lmbda[1:],
      it.beta,
      it.y,
      jp.zeros(0),
      it.yf,
      1.0,
      params,
  )

  g_l, g_u, gf_l, gf_u = bounds

  # rˢₖ = -yₖ - zˡₖ + zᵘₖ + dₖ
  r_s = -it.y - it.z_l + it.z_u + damping(g_l, g_u, mu, options)
  r_sf = -it.yf - it.zf_l + it.zf_u + damping(gf_l, gf_u, mu, options)

  # rᶜ˒ˡₖ = Zˡₖ(sₖ-gˡₖ) - μ1,   rᶜ˒ᵘₖ = Zᵘₖ(gᵘₖ-sₖ) - μ1
  r_c_l, r_c_u = complementarity(it.s, it.z_l, it.z_u, g_l, g_u, mu)
  r_cf_l, r_cf_u = complementarity(it.sf, it.zf_l, it.zf_u, gf_l, gf_u, mu)

  data = IterateData(
      objective=jp.sum(m.objective(it.x, it.u, params)),
      objective_gradx=gx,
      objective_gradu=gu,
      stationarity_x=r_x,
      stationarity_u=r_u,
      r0=m.initial_residual(it.x[0], params),
      r_f=m.dynamics_residual(it.x[:-1], it.u, it.x[1:], params),
      r_e=m.stage_equality_residual(it.x[:-1], it.u, params),
      r_g=m.stage_inequality_residual(it.x[:-1], it.u, params) - it.s,
      r_gf=m.terminal_inequality_residual(it.x[-1], params) - it.sf,
  )
  system = IpPrimalDualSystem.create(
      Hxx=Hxx,
      Hxu=Hxu,
      Huu=Huu,
      A=A,
      B=B,
      Ex=Ex,
      Eu=Eu,
      E0=E0,
      Gx=Gx,
      Gu=Gu,
      Gf=Gf,
      g_l=g_l,
      g_u=g_u,
      gf_l=gf_l,
      gf_u=gf_u,
      s=it.s,
      sf=it.sf,
      z_l=it.z_l,
      z_u=it.z_u,
      zf_l=it.zf_l,
      zf_u=it.zf_u,
      r_x=r_x,
      r_u=r_u,
      r_s=r_s,
      r_sf=r_sf,
      r0=data.r0,
      r_f=data.r_f,
      r_e=data.r_e,
      r_g=data.r_g,
      r_gf=data.r_gf,
      r_c_l=r_c_l,
      r_c_u=r_c_u,
      r_cf_l=r_cf_l,
      r_cf_u=r_cf_u,
  )
  return system, data


def refresh_barrier_terms(
    system: IpPrimalDualSystem,
    iterate: 'Iterate',
    mu: jax.Array,
    bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    options: Options,
) -> IpPrimalDualSystem:
  p = system
  it = iterate

  g_l, g_u, gf_l, gf_u = bounds

  # rˢₖ = -yₖ - zˡₖ + zᵘₖ + dₖ
  r_s = -it.y - it.z_l + it.z_u + damping(g_l, g_u, mu, options)
  r_sf = -it.yf - it.zf_l + it.zf_u + damping(gf_l, gf_u, mu, options)

  # rᶜ˒ˡₖ = Zˡₖ(sₖ-gˡₖ) - μ1,   rᶜ˒ᵘₖ = Zᵘₖ(gᵘₖ-sₖ) - μ1
  c_l, c_u = complementarity(it.s, it.z_l, it.z_u, g_l, g_u, mu)
  cf_l, cf_u = complementarity(it.sf, it.zf_l, it.zf_u, gf_l, gf_u, mu)

  return replace(
      p,
      r_s=r_s,
      r_sf=r_sf,
      r_c_l=c_l,
      r_c_u=c_u,
      r_cf_l=cf_l,
      r_cf_u=cf_u,
      delta_c=jp.asarray(0.0, p.Hxx.dtype),
      _cache=None,
      dx=None,
      du=None,
      dlmbda=None,
      dbeta=None,
      ds=None,
      dsf=None,
      dz_l=None,
      dz_u=None,
      dzf_l=None,
      dzf_u=None,
      dy=None,
      dyf=None,
  )


def system_defect(p: IpPrimalDualSystem) -> dict[str, jax.Array]:
  """Return every Newton-equation defect, including selected regularization."""

  # Hxxₜ δxₜ + Gfᵀδyₜ - δλₜ + rˣₜ
  # Hxxₖ δxₖ + Hxuₖ δuₖ + Aₖᵀδλₖ₊₁ + (Eˣₖ)ᵀδβₖ + (Gˣₖ)ᵀδyₖ - δλₖ + rˣₖ
  # Hxuₖᵀδxₖ + Huuₖ δuₖ + Bₖᵀδλₖ₊₁ + (Eᵘₖ)ᵀδβₖ + (Gᵘₖ)ᵀδyₖ + rᵘₖ
  # at k=0, replace -δλ₀ with +E0ᵀδλ₀
  rhs_x = p.r_x + _mv(p.Hxx, p.dx)
  rhs_x = rhs_x.at[:-1].add(
      _mv(p.Hxu, p.du)
      + _mv(p.A.swapaxes(-1, -2), p.dlmbda[1:])
      + _mv(p.Ex.swapaxes(-1, -2), p.dbeta)
      + _mv(p.Gx.swapaxes(-1, -2), p.dy)
  )
  rhs_x = rhs_x.at[0].add(p.E0.T @ p.dlmbda[0])
  rhs_x = rhs_x.at[1:].add(-p.dlmbda[1:])
  rhs_x = rhs_x.at[-1].add(p.Gf.T @ p.dyf)

  # Hxuₖᵀδxₖ + Huuₖ δuₖ + Bₖᵀδλₖ₊₁ + (Eᵘₖ)ᵀδβₖ + (Gᵘₖ)ᵀδyₖ + rᵘₖ
  rhs_u = p.r_u + _mv(p.Hxu.swapaxes(-1, -2), p.dx[:-1]) + _mv(p.Huu, p.du)
  rhs_u += _mv(p.B.swapaxes(-1, -2), p.dlmbda[1:]) + _mv(p.Eu.swapaxes(-1, -2), p.dbeta)
  rhs_u += _mv(p.Gu.swapaxes(-1, -2), p.dy)

  # -δyₖ - δzˡₖ + δzᵘₖ + rˢₖ
  rhs_s = p.r_s - p.dy - p.dz_l + p.dz_u

  # -δyₜ - δzˡₜ + δzᵘₜ + rˢₜ
  rhs_sf = p.r_sf - p.dyf - p.dzf_l + p.dzf_u

  # Aₖδxₖ + Bₖδuₖ - δxₖ₊₁ = -rᶠₖ
  rhs_f = p.r_f + _mv(p.A, p.dx[:-1]) + _mv(p.B, p.du) - p.dx[1:]

  # Eˣₖδxₖ + Eᵘₖδuₖ + rᵉₖ
  rhs_e = p.r_e + _mv(p.Ex, p.dx[:-1]) + _mv(p.Eu, p.du) - p.delta_c * p.dbeta

  # Gˣₖδxₖ + Gᵘₖδuₖ - δsₖ + rᵍₖ
  rhs_g = p.r_g + _mv(p.Gx, p.dx[:-1]) + _mv(p.Gu, p.du) - p.ds

  # Gf δxₜ - δsₜ + rᵍₜ
  rhs_gf = p.r_gf + p.Gf @ p.dx[-1] - p.dsf

  s_l, s_u = distances(p.s, p.g_l, p.g_u)
  sf_l, sf_u = distances(p.sf, p.gf_l, p.gf_u)

  # Zˡₖδsₖ + Sˡₖδzˡₖ + rᶜ˒ˡₖ
  rhs_c_l = jp.where(jp.isfinite(p.g_l), p.r_c_l + p.z_l * p.ds + s_l * p.dz_l, 0.0)

  # -Zᵘₖδsₖ + Sᵘₖδzᵘₖ + rᶜ˒ᵘₖ.
  rhs_c_u = jp.where(jp.isfinite(p.g_u), p.r_c_u - p.z_u * p.ds + s_u * p.dz_u, 0.0)

  # Zˡₜ δsₜ + Sˡₜ δzˡₜ + rᶜ˒ˡₜ
  rhs_cf_l = jp.where(jp.isfinite(p.gf_l), p.r_cf_l + p.zf_l * p.dsf + sf_l * p.dzf_l, 0.0)

  # -Zᵘₜ δsₜ + Sᵘₜ δzᵘₜ + rᶜ˒ᵘₜ.
  rhs_cf_u = jp.where(jp.isfinite(p.gf_u), p.r_cf_u - p.zf_u * p.dsf + sf_u * p.dzf_u, 0.0)

  # E0 δx₀ + r₀
  rhs0 = p.r0 + p.E0 @ p.dx[0]

  return {
      'r_x': rhs_x,
      'r_u': rhs_u,
      'r_s': rhs_s,
      'r_sf': rhs_sf,
      'r_f': rhs_f,
      'r_e': rhs_e,
      'r_g': rhs_g,
      'r_gf': rhs_gf,
      'r_c_l': rhs_c_l,
      'r_c_u': rhs_c_u,
      'r_cf_l': rhs_cf_l,
      'r_cf_u': rhs_cf_u,
      'r0': rhs0,
  }


def iterative_refinement_solve(
    system: IpPrimalDualSystem,
    linear_tol: float,
    max_refinement: int,
) -> tuple[IpPrimalDualSystem, jax.Array]:
  """Solve and refine a Newton system, returning (system, accurate).

  Solve once if increments or cached factors are missing, otherwise refine
  the supplied step without overwriting it.
  Solve correction equations using the full Newton defects as RHS, reusing
  those factors and preserving the original RHS and regularization.
  Stop after max_refinement corrections or when the largest absolute defect
  is ≤ linear_tol × (1 + ‖RHS‖∞). The returned flag reports this criterion.
  """
  if system._cache is None or any(getattr(system, name) is None for name in _STEPS):
    system = solve(system)

  defect = system_defect(system)
  threshold = linear_tol * (1 + max_abs(*(getattr(system, name) for name in defect)))

  def cond(carry) -> jax.Array:
    _, residual, count = carry
    return (count < max_refinement) & (max_abs(*residual.values()) > threshold)

  def body(carry):
    original, defect, count = carry
    correction = solve(original.replace_rhs(**defect))
    improved = replace(
        original,
        **{
            name: getattr(original, name) + getattr(correction, name)
            for name in _STEPS
        },
    )
    return improved, system_defect(improved), count + 1

  improved, defect, _ = jax.lax.while_loop(
      cond, body, (system, defect, 0)
  )
  return improved, max_abs(*defect.values()) <= threshold
