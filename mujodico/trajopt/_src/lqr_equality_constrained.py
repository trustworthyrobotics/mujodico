"""
Solve the following equality-constrained linear quadratic optimal control problem.

  minimizeₓ,ᵤ ∑ₖ₌₀ᵀ⁻¹ [½ xₖᵀQₖxₖ + xₖᵀNₖuₖ + ½ uₖᵀRₖuₖ + qₖᵀxₖ + rₖᵀuₖ]
              + ½ xₜᵀQₜxₜ + qₜᵀxₜ           (t=T denotes terminal)
  subject to  x₀ = x̂₀
              xₖ₊₁ = Aₖ xₖ + Bₖ uₖ + cₖ     for k = 0,…,T−1
              Eˣₖ xₖ + Eᵘₖ uₖ = −eₖ         for k = 0,…,T−1

Introduce multipliers λ₀ for the initial-state constraint, λₖ₊₁ for dynamics,
and βₖ for stage equalities. The Lagrangian is

  L = ∑ₖ₌₀ᵀ⁻¹ [
        ½ xₖᵀQₖxₖ + xₖᵀNₖuₖ + ½ uₖᵀRₖuₖ + qₖᵀxₖ + rₖᵀuₖ
        + λₖ₊₁ᵀ (Aₖ xₖ + Bₖ uₖ + cₖ − xₖ₊₁)
        + βₖᵀ (Eˣₖ xₖ + Eᵘₖ uₖ + eₖ)
      ]
      + ½ xₜᵀQₜxₜ + qₜᵀxₜ − λ₀ᵀ (x₀ − x̂₀)

The KKT conditions are

  Qₜ xₜ − λₜ = -qₜ
  Qₖ xₖ + Nₖ uₖ + Aₖᵀ λₖ₊₁ + (Eˣₖ)ᵀ βₖ − λₖ = -qₖ    for k = 0,…,T−1
  Nₖᵀ xₖ + Rₖ uₖ + Bₖᵀ λₖ₊₁ + (Eᵘₖ)ᵀ βₖ = -rₖ        for k = 0,…,T−1
  Aₖxₖ + Bₖuₖ − xₖ₊₁ = -cₖ                         for k = 0,…,T−1
  Eˣₖ xₖ + Eᵘₖ uₖ = -eₖ                            for k = 0,…,T−1
  x₀ = x̂₀
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jp


def _matvec(matrix, vector):
  return jax.vmap(jp.matmul)(matrix, vector)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _AffineScanCache:
  """
  The affine scan starting from x₋₁ = 0, and recursion
    xₖ = Dₖ xₖ₋₁ + dₖ    k=0,...,N,
  can be reduced by one level to
    x₂ⱼ₊₁ = (D₂ⱼ₊₁ D₂ⱼ) x₂ⱼ₋₁ + (D₂ⱼ₊₁ d₂ⱼ + d₂ⱼ₊₁)  j=0,...,⌊N/2⌋.

  `reduction` stores D₂ⱼ₊₁ for constructing the reduced RHS.
  `expansion` stores D₂ⱼ, used to reconstruct x₂ⱼ = D₂ⱼ x₂ⱼ₋₁ + d₂ⱼ.
  """

  reduction: jax.Array
  expansion: jax.Array
  child: '_AffineScanCache | None'


def _build_affine_scan(matrices):
  n = matrices.shape[0]
  if n < 2:
    return None
  reduction = matrices[1::2]
  pairs = reduction @ matrices[: 2*(n//2) : 2]  # D₂ⱼ₊₁ D₂ⱼ
  return _AffineScanCache(
      reduction=reduction,
      expansion=matrices[::2],
      child=_build_affine_scan(pairs),
  )


def _apply_affine_scan(cache, rhs):
  if cache is None:
    return rhs
  n = rhs.shape[0]
  pairs = _matvec(cache.reduction, rhs[: 2*(n//2) : 2]) + rhs[1::2]  # D₂ⱼ₊₁ d₂ⱼ + d₂ⱼ₊₁
  odd = _apply_affine_scan(cache.child, pairs)
  odd_padded = jp.concatenate([jp.zeros_like(rhs[:1]), odd])  # prepend x₋₁ = 0
  even = _matvec(cache.expansion, odd_padded[:cache.expansion.shape[0]]) + rhs[::2]
  return jp.empty_like(rhs).at[1::2].set(odd).at[::2].set(even)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _MatrixCache:
  backward_lu: tuple[jax.Array, jax.Array]
  Minv_K: jax.Array
  Minv_L: jax.Array
  C: jax.Array
  V: jax.Array
  forward_lu: tuple[jax.Array, jax.Array]
  D: jax.Array
  DT_Vnext: jax.Array
  backward_scan: _AffineScanCache | None
  forward_scan: _AffineScanCache | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EqualityConstrainedLQR:
  Q: jax.Array
  N: jax.Array
  R: jax.Array
  q: jax.Array
  r: jax.Array
  A: jax.Array
  B: jax.Array
  c: jax.Array
  Ex: jax.Array
  Eu: jax.Array
  e: jax.Array
  x0: jax.Array
  x: jax.Array | None = None
  u: jax.Array | None = None
  lmbda: jax.Array | None = None
  beta: jax.Array | None = None
  _cache: _MatrixCache | None = None

  @staticmethod
  def create(
      Q: jax.Array,
      N: jax.Array,
      R: jax.Array,
      q: jax.Array,
      r: jax.Array,
      A: jax.Array,
      B: jax.Array,
      c: jax.Array,
      Ex: jax.Array,
      Eu: jax.Array,
      e: jax.Array,
      x0: jax.Array,
  ) -> 'EqualityConstrainedLQR':
    T, nx, nu = N.shape
    assert Q.shape == (T + 1, nx, nx)
    assert N.shape == (T, nx, nu)
    assert R.shape == (T, nu, nu)
    assert q.shape == (T + 1, nx)
    assert r.shape == (T, nu)
    assert A.shape == (T, nx, nx)
    assert B.shape == (T, nx, nu)
    assert c.shape == (T, nx)
    nc = e.shape[1]
    assert Ex.shape == (T, nc, nx)
    assert Eu.shape == (T, nc, nu)
    assert e.shape == (T, nc)
    assert x0.shape == (nx,)

    return EqualityConstrainedLQR(
        Q=Q,
        N=N,
        R=R,
        q=q,
        r=r,
        A=A,
        B=B,
        c=c,
        Ex=Ex,
        Eu=Eu,
        e=e,
        x0=x0,
    )

  def replace_rhs(
      self,
      *,
      q=None,
      r=None,
      c=None,
      e=None,
      x0=None,
  ) -> 'EqualityConstrainedLQR':
    updates = {}
    for name, value in dict(q=q, r=r, c=c, e=e, x0=x0).items():
      if value is None:
        continue
      value = jp.asarray(value)
      original = getattr(self, name)
      assert value.shape == original.shape
      assert value.dtype == original.dtype
      updates[name] = value
    return replace(self, **updates, x=None, u=None, lmbda=None, beta=None)


def _build_matrix_cache(prob: EqualityConstrainedLQR) -> _MatrixCache:
  assert isinstance(prob, EqualityConstrainedLQR)

  nx = prob.Q.shape[1]
  nc = prob.e.shape[1]

  def boundary_map(Qk, Nk, Rk, Ak, Bk, Exk, Euk):
    Mk = jp.block([
        [Rk, Euk.T],
        [Euk, jp.zeros((nc, nc), dtype=Rk.dtype)],
    ])
    Kk = jp.vstack([
        Nk.T,
        Exk,
    ])
    Lk = jp.vstack([
        Bk.T,
        jp.zeros((nc, nx), dtype=Bk.dtype),
    ])
    lu_Mk = jax.scipy.linalg.lu_factor(Mk)
    Mkinv_Kk = jax.scipy.linalg.lu_solve(lu_Mk, Kk)
    Mkinv_Lk = jax.scipy.linalg.lu_solve(lu_Mk, Lk)

    # Pₖ = Qₖ - [Nₖ (Eˣₖ)ᵀ] Mₖ⁻¹ [Nₖ (Eˣₖ)ᵀ]ᵀ
    Pk = Qk - Kk.T @ Mkinv_Kk
    # A̅ₖ = Aₖ - [Bₖ 0] Mₖ⁻¹ [Nₖ (Eˣₖ)ᵀ]ᵀ
    Abark = Ak - Lk.T @ Mkinv_Kk
    # Cₖ = [Bₖ 0] Mₖ⁻¹ [Bₖ 0]ᵀ
    Ck = Lk.T @ Mkinv_Lk

    return (Pk, Abark, Ck), (lu_Mk, Mkinv_Kk, Mkinv_Lk)

  (P, Abar, C), (backward_lu, Minv_K, Minv_L) = jax.vmap(boundary_map)(
      Qk=prob.Q[:-1],
      Nk=prob.N,
      Rk=prob.R,
      Ak=prob.A,
      Bk=prob.B,
      Exk=prob.Ex,
      Euk=prob.Eu,
  )

  def compose_boundary(later, earlier):
    # reverse=True supplies the later interval first. Compose in time order.
    P1, A1, C1 = earlier
    P2, A2, C2 = later
    eye = jp.eye(nx, dtype=P1.dtype)
    lu_P2C1 = jax.scipy.linalg.lu_factor(eye + P2 @ C1)

    Pcomp = P1 + A1.T @ jax.scipy.linalg.lu_solve(lu_P2C1, P2 @ A1)
    Acomp = A2 @ jax.scipy.linalg.lu_solve(lu_P2C1, A1, trans=1)
    Ccomp = C2 + A2 @ jax.scipy.linalg.lu_solve(lu_P2C1, C1 @ A2.T, trans=1)
    return Pcomp, Acomp, Ccomp

  V, _, _ = jax.lax.associative_scan(
      jax.vmap(compose_boundary),
      (
          jp.concatenate((P, prob.Q[-1:])),
          jp.concatenate((Abar, jp.zeros((1, nx, nx), dtype=Abar.dtype))),
          jp.concatenate((C, jp.zeros((1, nx, nx), dtype=C.dtype))),
      ),
      reverse=True,
  )

  forward_lu = jax.vmap(jax.scipy.linalg.lu_factor)(
      jp.eye(nx, dtype=C.dtype) + C @ V[1:],
  )
  # Dₖ = (I + Cₖ Vₖ₊₁)⁻¹ A̅ₖ
  D = jax.vmap(jax.scipy.linalg.lu_solve)(forward_lu, Abar)
  # Dₖᵀ = A̅ₖᵀ (I + Vₖ₊₁ Cₖ)⁻¹
  DT = D.swapaxes(-1, -2)
  zero = jp.zeros((1, nx, nx), dtype=D.dtype)

  return _MatrixCache(
      backward_lu=backward_lu,
      Minv_K=Minv_K,
      Minv_L=Minv_L,
      C=C,
      V=V,
      forward_lu=forward_lu,
      D=D,
      DT_Vnext=DT @ V[1:],
      # Constant endpoint maps encode v[T] = q[T] and x[0] = x0.
      backward_scan=_build_affine_scan(jp.concatenate((DT, zero))[::-1]),
      forward_scan=_build_affine_scan(jp.concatenate((zero, D))),
  )


def _solve_rhs(prob: EqualityConstrainedLQR) -> EqualityConstrainedLQR:
  assert isinstance(prob, EqualityConstrainedLQR)
  assert prob._cache is not None
  cache = prob._cache

  nu = prob.R.shape[1]

  # fₖ = [rₖᵀ eₖᵀ]ᵀ
  f = jp.concatenate((prob.r, prob.e), axis=-1)
  Minv_f = jax.vmap(jax.scipy.linalg.lu_solve)(cache.backward_lu, f)
  # pₖ = qₖ - [Nₖ (Eˣₖ)ᵀ] Mₖ⁻¹ fₖ
  p = (
      prob.q[:-1]
      - _matvec(prob.N, Minv_f[:, :nu])
      - _matvec(prob.Ex.swapaxes(-1, -2), Minv_f[:, nu:])
  )
  # bₖ = cₖ - [Bₖ 0] Mₖ⁻¹ fₖ
  b = prob.c - _matvec(prob.B, Minv_f[:, :nu])
  # vₖ = Dₖᵀ vₖ₊₁ + pₖ + Dₖᵀ Vₖ₊₁ bₖ
  offsets = jp.concatenate((p + _matvec(cache.DT_Vnext, b), prob.q[-1:]))
  v = _apply_affine_scan(cache.backward_scan, offsets[::-1])[::-1]

  # dₖ = (I + Cₖ Vₖ₊₁)⁻¹ (bₖ - Cₖ vₖ₊₁)
  d = jax.vmap(jax.scipy.linalg.lu_solve)(
      cache.forward_lu,
      b - _matvec(cache.C, v[1:]),
  )
  # xₖ₊₁ = Dₖ xₖ + dₖ
  x = _apply_affine_scan(cache.forward_scan, jp.concatenate((prob.x0[None], d)))

  # λₖ = Vₖ xₖ + vₖ
  lmbda = _matvec(cache.V, x) + v
  # [uₖᵀ βₖᵀ] = −Mₖ⁻¹ (Kₖ xₖ + Lₖ λₖ₊₁ + fₖ)
  u_beta = -(_matvec(cache.Minv_K, x[:-1]) + _matvec(cache.Minv_L, lmbda[1:]) + Minv_f)
  u = u_beta[:, :nu]
  beta = u_beta[:, nu:]

  return replace(
      prob,
      x=x,
      u=u,
      lmbda=lmbda,
      beta=beta,
  )

def solve(prob: EqualityConstrainedLQR) -> EqualityConstrainedLQR:
  assert isinstance(prob, EqualityConstrainedLQR)
  if prob._cache is None:
    return _solve_rhs(replace(prob, _cache=_build_matrix_cache(prob)))
  else:
    return _solve_rhs(prob)
