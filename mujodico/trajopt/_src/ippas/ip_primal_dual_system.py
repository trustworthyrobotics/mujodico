"""
Compute a primal-dual interior-point Newton step for the bounded OCP

  minimizeₓ,ᵤ ∑ₖ₌₀ᵀ⁻¹ ℓₖ(xₖ, uₖ) + ℓₜ(xₜ)          (t=T denotes terminal)
  subject to  h⁰(x₀) = 0
              xₖ₊₁ = fₖ(xₖ, uₖ)                   for k = 0,…,T−1
              hₖ(xₖ, uₖ) = 0                      for k = 0,…,T−1
              gˡₖ ≤ gₖ(xₖ, uₖ) ≤ gᵘₖ               for k = 0,…,T−1
              gˡₜ ≤ gₜ(xₜ) ≤ gᵘₜ.

Introduce a single slack sₖ = gₖ for both bounds of each inequality function,
including the terminal slack sₜ (the field sf). The Lagrangian is

  L = ∑ₖ₌₀ᵀ⁻¹ [ℓₖ + λₖ₊₁ᵀ(fₖ − xₖ₊₁) + βₖᵀhₖ
              − (zˡₖ)ᵀ(gₖ − gˡₖ) − (zᵘₖ)ᵀ(gᵘₖ − gₖ)]
      + ℓₜ − (zˡₜ)ᵀ(gₜ − gˡₜ) − (zᵘₜ)ᵀ(gᵘₜ − gₜ)
      + λ₀ᵀ h⁰(x₀).

The inputs Hxx, Hxu, Huu are blocks of ∇²L, with Hxx including the terminal
block. A, B are the Jacobians of f; Ex, Eu those of h; and Gx, Gu those of g.
Gf is the terminal inequality Jacobian, and E0 = ∇h⁰(x₀) is the square,
nonsingular initial equality Jacobian. For barrier parameter κ, residuals are

  rˣₜ = ∇ℓₜ − Gfᵀ(zˡₜ − zᵘₜ) − λₜ
  rˣₖ = ∇ₓℓₖ + Aₖᵀλₖ₊₁ + (Eˣₖ)ᵀβₖ − (Gˣₖ)ᵀ(zˡₖ − zᵘₖ) − λₖ
  rᵘₖ = ∇ᵤℓₖ + Bₖᵀλₖ₊₁ + (Eᵘₖ)ᵀβₖ − (Gᵘₖ)ᵀ(zˡₖ − zᵘₖ)
  rᶠₖ = fₖ − xₖ₊₁,    rᵉₖ = hₖ,    rᵍₖ = gₖ − sₖ,    rᵍₜ = gₜ − sₜ
  rᶜ˒ˡₖ = Zˡₖ(sₖ − gˡₖ) − κ1,    rᶜ˒ᵘₖ = Zᵘₖ(gᵘₖ − sₖ) − κ1
  r₀ = h⁰(x₀).

Write Zˡₖ = diag(zˡₖ), Zᵘₖ = diag(zᵘₖ), Sˡₖ = diag(sₖ − gˡₖ), and Sᵘₖ = diag(gᵘₖ − sₖ).
The Newton KKT equations for the terminal stage are

  Hxxₜ δxₜ − Gfᵀ(δzˡₜ − δzᵘₜ) − δλₜ = −rˣₜ
  Gf δxₜ − δsₜ = −rᵍₜ
  Zˡₜ δsₜ + Sˡₜ δzˡₜ = −rᶜ˒ˡₜ
  −Zᵘₜ δsₜ + Sᵘₜ δzᵘₜ = −rᶜ˒ᵘₜ.

KKT equations for the stages at k=0,…,T−1 are

  Hxxₖ δxₖ + Hxuₖ δuₖ + Aₖᵀδλₖ₊₁ + (Eˣₖ)ᵀδβₖ − (Gˣₖ)ᵀ(δzˡₖ − δzᵘₖ) − δλₖ = −rˣₖ
  Hxuₖᵀ δxₖ + Huuₖ δuₖ + Bₖᵀδλₖ₊₁ + (Eᵘₖ)ᵀδβₖ − (Gᵘₖ)ᵀ(δzˡₖ − δzᵘₖ)     = −rᵘₖ
  Aₖ δxₖ + Bₖ δuₖ − δxₖ₊₁                                            = −rᶠₖ
  Eˣₖ δxₖ + Eᵘₖ δuₖ                                                 = −rᵉₖ
  Gˣₖ δxₖ + Gᵘₖ δuₖ − δsₖ                                            = −rᵍₖ
  Zˡₖ δsₖ + Sˡₖ δzˡₖ                                                = −rᶜ˒ˡₖ
  −Zᵘₖ δsₖ + Sᵘₖ δzᵘₖ                                               = −rᶜ˒ᵘₖ
  E₀ δx₀ = −r₀.

Except for at k=0, the state-stationarity equation instead reads

  Hxx₀ δx₀ + Hxu₀ δu₀ + A₀ᵀδλ₁ + (Eˣ₀)ᵀδβ₀ − (Gˣ₀)ᵀ(δzˡ₀ − δzᵘ₀) + E₀ᵀδλ₀ = −rˣ₀.

Eliminate slacks and bound multipliers using

  Dₖ = (Sˡₖ)⁻¹Zˡₖ + (Sᵘₖ)⁻¹Zᵘₖ
  ξₖ = Dₖ rᵍₖ + (Sˡₖ)⁻¹rᶜ˒ˡₖ − (Sᵘₖ)⁻¹rᶜ˒ᵘₖ
  Qₖ = Hxxₖ + (Gˣₖ)ᵀDₖGˣₖ,     Qₜ = Hxxₜ + GfᵀDₜGf
  Nₖ = Hxuₖ + (Gˣₖ)ᵀDₖGᵘₖ,     Rₖ = Huuₖ + (Gᵘₖ)ᵀDₖGᵘₖ
  qₖ = rˣₖ + (Gˣₖ)ᵀξₖ,         qₜ = rˣₜ + Gfᵀξₜ
  rₖ = rᵘₖ + (Gᵘₖ)ᵀξₖ.

EqualityConstrainedLQR solves this reduced system with c=r_f, e=r_e, and x0=−E₀⁻¹r₀.
Its initial multiplier δλ̄₀ is transformed back using δλ₀ = −E₀⁻ᵀδλ̄₀;
all later dynamics multipliers transfer unchanged.
Recover δsₖ = Gˣₖδxₖ + Gᵘₖδuₖ + rᵍₖ and δsₜ = Gf δxₜ + rᵍₜ, then

  δzˡₖ = −(Sˡₖ)⁻¹(Zˡₖδsₖ + rᶜ˒ˡₖ)
  δzᵘₖ = (Sᵘₖ)⁻¹(Zᵘₖδsₖ − rᶜ˒ᵘₖ),               for k=0,…,T.

Finite bounds require strictly interior slacks and nonnegative multipliers.
Infinite-bound sides omit the corresponding multiplier and complementarity
equation: their inverse distances, contributions, and returned δz are zero.
"""

from dataclasses import dataclass
from dataclasses import replace

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import equality_constrained_lqr as lqr


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
      r0: jax.Array,
  ) -> 'IpPrimalDualSystem':
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
        r0=r0,
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

  xi = _bound_rhs(stage, prob.r_g, prob.r_c_l, prob.r_c_u)
  xif = _bound_rhs(terminal, prob.r_gf, prob.r_cf_l, prob.r_cf_u)
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
      _cache=_ReductionCache(
          stage=stage,
          terminal=terminal,
          initial_lu=initial_lu,
          reduced=reduced,
      ),
  )
