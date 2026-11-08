"""
Solve the following generalized equality-constrained linear quadratic problem.
Let wₖ = [xₖᵀ uₖᵀ xₖ₊₁ᵀ]ᵀ collect the current state, control, and next state.

  minimizeₓ,ᵤ ∑ₖ₌₀ᵀ⁻¹ [½ wₖᵀHₖwₖ + gₖᵀwₖ]
              + ½ xₜᵀQₜxₜ + qₜᵀxₜ           (t=T denotes terminal)
  subject to  x₀ = x̂₀
              xₖ₊₁ = Aₖ xₖ + Bₖ uₖ + cₖ     for k = 0,…,T−1
              Eₖ wₖ = −eₖ                  for k = 0,…,T−1

Introduce multipliers λ₀ for the initial-state constraint, λₖ₊₁ for dynamics,
and βₖ for stage equalities. The Lagrangian is

  L = ∑ₖ₌₀ᵀ⁻¹ [
        ½ wₖᵀHₖwₖ + gₖᵀwₖ
        + λₖ₊₁ᵀ (Aₖxₖ + Bₖuₖ + cₖ − xₖ₊₁)
        + βₖᵀ (Eₖwₖ + eₖ)
      ]
      + ½ xₜᵀQₜxₜ + qₜᵀxₜ + λ₀ᵀ (x̂₀ - x₀).

Define the stage cost and equality gradient, split into the blocks of wₖ:

  σₖ = Hₖwₖ + gₖ + Eₖᵀβₖ = [(σˣₖ)ᵀ (σᵘₖ)ᵀ (σ⁺ₖ)ᵀ]ᵀ.

Here σˣₖ, σᵘₖ, and σ⁺ₖ are the derivatives with respect to xₖ, uₖ, and xₖ₊₁.
Set σ⁺₋₁ = 0. The KKT conditions are

  Qₜxₜ + qₜ + σ⁺ₜ₋₁ − λₜ = 0                (t=T denotes terminal)
  σˣₖ + σ⁺ₖ₋₁ + Aₖᵀλₖ₊₁ − λₖ = 0           for k = 0,…,T−1
  σᵘₖ + Bₖᵀλₖ₊₁ = 0                       for k = 0,…,T−1
  Aₖxₖ + Bₖuₖ − xₖ₊₁ = −cₖ                 for k = 0,…,T−1
  Eₖwₖ = −eₖ                              for k = 0,…,T−1
  x₀ = x̂₀

Substitute wₖ = Ψₖ [xₖᵀ uₖᵀ]ᵀ + ψₖ, where

  Ψₖ = [ I   0 ],    ψₖ = [ 0  ]
      [ 0   I ]          [ 0  ]
      [ Aₖ  Bₖ ]          [ cₖ ]

The reduced stage coefficients are

  H̄ₖ = ΨₖᵀHₖΨₖ,    ḡₖ = Ψₖᵀ(Hₖψₖ + gₖ)
  Ēₖ = EₖΨₖ,       ēₖ = eₖ + Eₖψₖ.

The reduced problem is solved by lqr_equality_constrained without changing
its parallel algorithm or its nonsingularity assumptions.

Recover the original dynamics multipliers from the reduced multipliers λ̄ₖ:

  σ⁺ₖ = (Hₖwₖ + gₖ)⁺ + (E⁺ₖ)ᵀβₖ
  λₖ₊₁ = λ̄ₖ₊₁ + σ⁺ₖ                       for k = 0,…,T−1
  λ₀ = λ̄₀.

Here ⁺ selects the xₖ₊₁ block of a vector or the corresponding matrix columns.
The stage-equality multipliers βₖ transfer unchanged.
"""

from dataclasses import dataclass
from dataclasses import replace

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import equality_constrained_lqr as lqr


def _matvec(matrix, vector):
  return jax.vmap(jp.matmul)(matrix, vector)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _ReductionCache:
  Psi: jax.Array
  reduced: lqr.EqualityConstrainedLQR


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GeneralizedEqualityConstrainedLQR:
  H: jax.Array
  g: jax.Array
  A: jax.Array
  B: jax.Array
  c: jax.Array
  E: jax.Array
  e: jax.Array
  x0: jax.Array
  Qf: jax.Array
  qf: jax.Array
  x: jax.Array | None = None
  u: jax.Array | None = None
  lmbda: jax.Array | None = None
  beta: jax.Array | None = None
  _cache: _ReductionCache | None = None

  @staticmethod
  def create(
      H: jax.Array,
      g: jax.Array,
      A: jax.Array,
      B: jax.Array,
      c: jax.Array,
      E: jax.Array,
      e: jax.Array,
      Qf: jax.Array,
      qf: jax.Array,
      x0: jax.Array,
  ) -> 'GeneralizedEqualityConstrainedLQR':
    T, nx, nu = B.shape
    d = 2 * nx + nu
    nc = e.shape[1]
    assert H.shape == (T, d, d)
    assert g.shape == (T, d)
    assert A.shape == (T, nx, nx)
    assert B.shape == (T, nx, nu)
    assert c.shape == (T, nx)
    assert E.shape == (T, nc, d)
    assert e.shape == (T, nc)
    assert Qf.shape == (nx, nx)
    assert qf.shape == (nx,)
    assert x0.shape == (nx,)
    return GeneralizedEqualityConstrainedLQR(
        H=H,
        g=g,
        A=A,
        B=B,
        c=c,
        E=E,
        e=e,
        Qf=Qf,
        qf=qf,
        x0=x0,
    )

  def replace_rhs(
      self,
      *,
      g=None,
      c=None,
      e=None,
      x0=None,
      qf=None,
  ) -> 'GeneralizedEqualityConstrainedLQR':
    updates = {}
    for name, value in dict(g=g, c=c, e=e, qf=qf, x0=x0).items():
      if value is None:
        continue
      value = jp.asarray(value)
      original = getattr(self, name)
      assert value.shape == original.shape
      assert value.dtype == original.dtype
      updates[name] = value
    return replace(self, **updates, x=None, u=None, lmbda=None, beta=None)


def solve(
    prob: GeneralizedEqualityConstrainedLQR
) -> GeneralizedEqualityConstrainedLQR:
  assert isinstance(prob, GeneralizedEqualityConstrainedLQR)

  T, nx, nu = prob.B.shape

  if prob._cache is None:
    Psi = jp.concatenate(
        (
            jp.broadcast_to(
                jp.eye(nx + nu, dtype=prob.H.dtype), (T, nx + nu, nx + nu)
            ),
            jp.concatenate((prob.A, prob.B), axis=-1),
        ),
        axis=1,
    )
  else:
    Psi = prob._cache.Psi

  psi = jp.concatenate(
      (jp.zeros((T, nx + nu), dtype=prob.c.dtype), prob.c), axis=-1
  )
  gbar = _matvec(Psi.swapaxes(-1, -2), _matvec(prob.H, psi) + prob.g)
  ebar = prob.e + _matvec(prob.E, psi)
  rhs = {
      'q': jp.concatenate((gbar[:, :nx], prob.qf[None])),
      'r': gbar[:, nx:],
      'c': prob.c,
      'e': ebar,
      'x0': prob.x0,
  }

  if prob._cache is None:
    Hbar = Psi.swapaxes(-1, -2) @ prob.H @ Psi
    Ebar = prob.E @ Psi
    reduced = lqr.EqualityConstrainedLQR.create(
        Q=jp.concatenate((Hbar[:, :nx, :nx], prob.Qf[None])),
        N=Hbar[:, :nx, nx:],
        R=Hbar[:, nx:, nx:],
        A=prob.A,
        B=prob.B,
        Ex=Ebar[:, :, :nx],
        Eu=Ebar[:, :, nx:],
        **rhs,
    )
  else:
    reduced = prob._cache.reduced.replace_rhs(**rhs)

  reduced = lqr.solve(reduced)

  w = jp.concatenate((reduced.x[:-1], reduced.u, reduced.x[1:]), axis=-1)
  sigma_xnext = (
      _matvec(prob.H[:, nx+nu:, :], w)
      + prob.g[:, nx+nu:]
      + _matvec(prob.E[:, :, nx+nu:].swapaxes(-1, -2), reduced.beta)
  )
  lmbda = jp.concatenate((reduced.lmbda[:1], reduced.lmbda[1:] + sigma_xnext))

  return replace(
      prob,
      x=reduced.x,
      u=reduced.u,
      lmbda=lmbda,
      beta=reduced.beta,
      _cache=_ReductionCache(Psi=Psi, reduced=reduced),
  )
