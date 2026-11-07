"""
Solve the following equality-constrained linear quadratic optimal control problem.

  minimizeₓ,ᵤ ∑ₖ₌₀ᵀ⁻¹ [½ xₖᵀQₖxₖ + xₖᵀNₖuₖ + ½ uₖᵀRₖuₖ + qₖᵀxₖ + rₖᵀuₖ]
              + ½ xₜᵀQₜxₜ + qₜᵀxₜ           (t=T denotes terminal)
  subject to  x₀ = x̂₀
              xₖ₊₁ = Aₖ xₖ + Bₖ uₖ + cₖ     for k = 0,…,T−1
              Hˣₖ xₖ + Hᵘₖ uₖ = −hₖ         for k = 0,…,T−1

Introduce multipliers λ₀ for the initial-state constraint, λₖ₊₁ for dynamics,
and νₖ for stage equalities. The Lagrangian is

  L = ∑ₖ₌₀ᵀ⁻¹ [
        ½ xₖᵀQₖxₖ + xₖᵀNₖuₖ + ½ uₖᵀRₖuₖ + qₖᵀxₖ + rₖᵀuₖ
        + λₖ₊₁ᵀ (Aₖ xₖ + Bₖ uₖ + cₖ − xₖ₊₁)
        + νₖᵀ (Hˣₖ xₖ + Hᵘₖ uₖ + hₖ)
      ]
      + ½ xₜᵀQₜxₜ + qₜᵀxₜ − λ₀ᵀ (x₀ − x̂₀)

The KKT conditions are

  Qₜ xₜ − λₜ = -qₜ
  Qₖ xₖ + Nₖ uₖ + Aₖᵀ λₖ₊₁ + (Hˣₖ)ᵀ νₖ − λₖ = -qₖ    for k = 0,…,T−1
  Nₖᵀ xₖ + Rₖ uₖ + Bₖᵀ λₖ₊₁ + (Hᵘₖ)ᵀ νₖ = -rₖ        for k = 0,…,T−1
  Aₖxₖ + Bₖuₖ − xₖ₊₁ = -cₖ                         for k = 0,…,T−1
  Hˣₖ xₖ + Hᵘₖ uₖ = -hₖ                            for k = 0,…,T−1
  x₀ = x̂₀
"""

import functools
from dataclasses import dataclass, replace

import jax
import jax.numpy as jp


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        'Q',
        'N',
        'R',
        'q',
        'r',
        'A',
        'B',
        'c',
        'Hx',
        'Hu',
        'h',
        'x0',
        'x',
        'u',
        'lmbda',
        'nu',
    ],
    meta_fields=[],
)
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
  Hx: jax.Array
  Hu: jax.Array
  h: jax.Array
  x0: jax.Array
  x: jax.Array | None = None
  u: jax.Array | None = None
  lmbda: jax.Array | None = None
  nu: jax.Array | None = None

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
      Hx: jax.Array,
      Hu: jax.Array,
      h: jax.Array,
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
    nc = h.shape[1]
    assert Hx.shape == (T, nc, nx)
    assert Hu.shape == (T, nc, nu)
    assert h.shape == (T, nc)
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
        Hx=Hx,
        Hu=Hu,
        h=h,
        x0=x0,
    )


def solve(prob: EqualityConstrainedLQR) -> EqualityConstrainedLQR:
  assert isinstance(prob, EqualityConstrainedLQR)
  _, nx, nu = prob.N.shape
  nc = prob.h.shape[1]

  def boundary_map(Qk, Nk, Rk, qk, rk, Ak, Bk, ck, Hxk, Huk, hk):
    Mk = jp.block([
        [Rk, Huk.T],
        [Huk, jp.zeros((nc, nc), dtype=Rk.dtype)],
    ])
    Kk = jp.vstack([
        Nk.T,
        Hxk,
    ])
    Lk = jp.vstack([
        Bk.T,
        jp.zeros((nc, nx), dtype=Bk.dtype),
    ])
    ek = jp.concatenate([
        rk,
        hk,
    ])

    lu_Mk = jax.scipy.linalg.lu_factor(Mk)
    Mkinv_Kk = jax.scipy.linalg.lu_solve(lu_Mk, Kk)
    Mkinv_ek = jax.scipy.linalg.lu_solve(lu_Mk, ek)
    Mkinv_Lk = jax.scipy.linalg.lu_solve(lu_Mk, Lk)

    Pk = Qk - Kk.T @ Mkinv_Kk
    pk = qk - Kk.T @ Mkinv_ek
    Abark = Ak - Lk.T @ Mkinv_Kk
    bk = ck - Lk.T @ Mkinv_ek
    Ck = Lk.T @ Mkinv_Lk

    return (Pk, pk, Abark, bk, Ck), (Mkinv_Kk, Mkinv_ek, Mkinv_Lk)

  (P, p, Abar, b, C), (Minv_K, Minv_e, Minv_L) = jax.vmap(boundary_map)(
      Qk=prob.Q[:-1],
      Nk=prob.N,
      Rk=prob.R,
      qk=prob.q[:-1],
      rk=prob.r,
      Ak=prob.A,
      Bk=prob.B,
      ck=prob.c,
      Hxk=prob.Hx,
      Huk=prob.Hu,
      hk=prob.h,
  )
  P = jp.concatenate((P, prob.Q[-1:]))
  p = jp.concatenate((p, prob.q[-1:]))
  Abar = jp.concatenate((Abar, jp.zeros((1, nx, nx), dtype=Abar.dtype)))
  b = jp.concatenate((b, jp.zeros((1, nx), dtype=b.dtype)))
  C = jp.concatenate((C, jp.zeros((1, nx, nx), dtype=C.dtype)))

  def compose_boundary(later, earlier):
    # reverse=True supplies the later interval first. Compose in time order.
    P1, p1, A1, b1, C1 = earlier
    P2, p2, A2, b2, C2 = later
    eye = jp.eye(nx, dtype=P1.dtype)
    lu_F = jax.scipy.linalg.lu_factor(eye + P2 @ C1)
    lu_G = jax.scipy.linalg.lu_factor(eye + C1 @ P2)

    Pcomp = P1 + A1.T @ jax.scipy.linalg.lu_solve(lu_F, P2 @ A1)
    pcomp = p1 + A1.T @ jax.scipy.linalg.lu_solve(lu_F, p2 + P2 @ b1)
    Acomp = A2 @ jax.scipy.linalg.lu_solve(lu_G, A1)
    bcomp = b2 + A2 @ jax.scipy.linalg.lu_solve(lu_G, b1 - C1 @ p2)
    Ccomp = C2 + A2 @ jax.scipy.linalg.lu_solve(lu_G, C1 @ A2.T)
    return Pcomp, pcomp, Acomp, bcomp, Ccomp

  V, v, _, _, _ = jax.lax.associative_scan(
      jax.vmap(compose_boundary),
      (P, p, Abar, b, C),
      reverse=True,
  )

  def forward_map(Abark, bk, Ck, Vnextk, vnextk):
    affine = jax.scipy.linalg.solve(
        jp.eye(nx, dtype=Ck.dtype) + Ck @ Vnextk,
        jp.column_stack((Abark, bk - Ck @ vnextk))
    )
    return affine[:, :nx], affine[:, nx]

  D, d = jax.vmap(forward_map)(
      Abark=Abar[:-1],
      bk=b[:-1],
      Ck=C[:-1],
      Vnextk=V[1:],
      vnextk=v[1:],
  )

  def compose_affine(earlier, later):
    D1, d1 = earlier
    D2, d2 = later
    return D2 @ D1, D2 @ d1 + d2

  D, d = jax.lax.associative_scan(
    jax.vmap(compose_affine),
    (D, d)
  )

  matvec = jax.vmap(jp.matmul)
  x = jp.concatenate((prob.x0[None], D @ prob.x0 + d))
  lmbda = matvec(V, x) + v
  u_nu = -(matvec(Minv_K, x[:-1]) + matvec(Minv_L, lmbda[1:]) + Minv_e)
  u = u_nu[:, :nu]
  nu = u_nu[:, nu:]

  return replace(
      prob,
      x=x,
      u=u,
      lmbda=lmbda,
      nu=nu,
  )
