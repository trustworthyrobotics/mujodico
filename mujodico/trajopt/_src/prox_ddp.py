"""
ProxDDP solves the following problem
    minimizeₓ,ᵤ  ∑ₜ₌₀ᴺ⁻¹ ℓ(xₜ,uₜ) + ℓf(x_N)
    subject to   g(xₜ, uₜ, xₜ₊₁) = 0   t=0,...N−1
                 x₀ = x̂₀
                 h(xₜ, uₜ) ≤ 0        t=0,...N−1
                 hf(x_N) ≤ 0
"""

from collections.abc import Callable
from dataclasses import dataclass
import jax
import jax.numpy as jp


class ProxDDP:

  def __init__(
      self,
      stage_cost: Callable[[jax.Array, jax.Array], jax.Array],
      terminal_cost: Callable[[jax.Array], jax.Array],
      implicit_dynamics: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
      stage_constraint: Callable[[jax.Array, jax.Array], jax.Array] | None,
      terminal_constraint: Callable[[jax.Array], jax.Array] | None,
      horizon: int,
      num_states: int,
      num_controls: int,
  ):
    if stage_constraint is None:
      stage_constraint = lambda x, u: jp.empty(0)
    if terminal_constraint is None:
      terminal_constraint = lambda x: jp.empty(0)

    self._ell = stage_cost
    self._ellf = terminal_cost
    self._g = implicit_dynamics
    self._h = stage_constraint
    self._hf = terminal_constraint
    self._N = horizon
    self._nx = num_states
    self._nu = num_controls

    x = jp.zeros(num_states)
    u = jp.zeros(num_controls)
    self._ng = self._g(x, u, x).size
    self._nh = self._h(x, u).size
    self._nhf = self._hf(x).size


  def solve(
      self,
      x0: jax.Array,
      x_init: jax.Array | None = None,
      u_init: jax.Array | None = None,
      lam_init: tuple[jax.Array, jax.Array] | None = None,
      nu_init: tuple[jax.Array, jax.Array] | None = None,
  ):
    assert x0.shape == (self._nx,)

    if x_init is None:
      # x_init = jp.repeat(x0[None, :], self._N, axis=0)
      x_init = jp.zeros((self._N, self._nx))
    if u_init is None:
      u_init = jp.zeros((self._N - 1, self._nu))
    if lam_init is None:
      lam_init = jp.zeros((self._N - 1, self._ng)), jp.zeros(self._nx)
    if nu_init is None:
      nu_init = jp.zeros((self._N - 1, self._nh)), jp.zeros(self._nhf)

    def cond_fn(carry):
      _, _, i = carry
      return i < 200

    def body_fn(carry):
      (x, u, lam, nu), penalty, i = carry

      (x, u, lam_new, nu_new), iters = self._minimize_augmented_lagrangian(
          x0_hat=x0,
          lam_fixed=lam,
          nu_fixed=nu,
          penalty=penalty,
          x_init=x,
          u_init=u,
      )

      lam = lam_new
      nu = nu_new
      penalty = jp.maximum(penalty * 0.1, 1e-6)

      return (x, u, lam, nu), penalty, i + iters

    penalty = 1.0
    (x, u, lam, nu), penalty, iters = jax.lax.while_loop(
        cond_fn,
        body_fn,
        ((x_init, u_init, lam_init, nu_init), penalty, 0.0),
    )

    return x, u


  def _minimize_augmented_lagrangian(
      self,
      x0_hat: jax.Array,
      lam_fixed: tuple[jax.Array, jax.Array],
      nu_fixed: tuple[jax.Array, jax.Array],
      penalty: float,
      x_init: jax.Array,
      u_init: jax.Array,
  ):
    """
    Mimimizes
        ℒ_μ(x, u; λ, ν) = ∑ₜ₌₀ᴺ⁻¹ ℓ(xₜ,uₜ) + ℓf(x_N)
            + ∑ₜ₌₀ᴺ⁻¹ 1/(2μ) ‖g(xₜ, uₜ, xₜ₊₁) + μ λₜ‖²
            + 1/(2μ) ‖x₀ − x̂₀ + μ λ₀‖²
            + ∑ₜ₌₀ᴺ⁻¹ 1/(2μ) ‖[h(xₜ, uₜ) + μ νₜ]₊‖²
            + 1/(2μ) ‖[hf(x_N) + μ ν_N]₊‖²
    where μ is the penalty parameter.
    """
    assert x0_hat.shape == (self._nx,)
    assert lam_fixed[0].shape == (self._N - 1, self._ng)
    assert lam_fixed[1].shape == (self._nx,)
    assert nu_fixed[0].shape == (self._N - 1, self._nh)
    assert nu_fixed[1].shape == (self._nhf,)
    assert jp.isscalar(penalty)
    assert x_init.shape == (self._N, self._nx)
    assert u_init.shape == (self._N - 1, self._nu)

    x = x_init
    u = u_init
    lam = lam_fixed
    nu = nu_fixed

    def cond_fn(carry):
        _, alpha, i = carry
        return (alpha > 1e-6) & (i < 10)

    def body_fn(carry):
        (x, u, lam, nu), _, i = carry

        delta_x, delta_u, delta_lam, delta_nu = self._newton_direction(
            x0_hat=x0_hat,
            lam_fixed=lam_fixed,
            nu_fixed=nu_fixed,
            penalty=penalty,
            x_nominal=x,
            u_nominal=u,
            lam_nominal=lam,
            nu_nominal=nu,
        )

        (x, u, lam, nu), alpha = self._line_search(
            x0_hat=x0_hat,
            lam_fixed=lam_fixed,
            nu_fixed=nu_fixed,
            penalty=penalty,
            x_nominal=x,
            u_nominal=u,
            lam_nominal=lam,
            nu_nominal=nu,
            delta_x=delta_x,
            delta_u=delta_u,
            delta_lam=delta_lam,
            delta_nu=delta_nu,
        )

        return (x, u, lam, nu), alpha, i + 1

    (x, u, lam, nu), _, iters = jax.lax.while_loop(
        cond_fn,
        body_fn,
        ((x, u, lam, nu), jp.inf, 0),
    )

    return (x, u, lam, nu), iters


  def _newton_direction(
      self,
      x0_hat: jax.Array,
      lam_fixed: tuple[jax.Array, jax.Array],
      nu_fixed: tuple[jax.Array, jax.Array],
      penalty: float,
      x_nominal: jax.Array,
      u_nominal: jax.Array,
      lam_nominal: tuple[jax.Array, jax.Array],
      nu_nominal: tuple[jax.Array, jax.Array],
  ):
    _nx, _nu = self._nx, self._nu
    _ng = self._ng
    _nh = self._nh

    xN = x_nominal[-1]
    ellf_x = jax.grad(self._ellf)(xN)
    ellf_xx = jax.hessian(self._ellf)(xN)
    hf = self._hf(xN)
    hf_x = jax.jacfwd(self._hf)(xN)

    nuN_fixed = nu_fixed[-1]
    nuN_plus = nuN_fixed + hf / penalty
    active = nuN_plus > 0
    nuN_plus = jp.where(active, nuN_plus, 0.0)

    nuN = nu_nominal[-1]
    c = nuN_plus - nuN
    C = (hf_x * active[:, None]) / penalty

    V_x = ellf_x + (hf_x.T * active) @ nuN_plus
    V_xx = ellf_xx + (hf_x.T * active) @ hf_x / penalty

    def backward_step(carry, input):
      V_x, V_xx = carry
      xt, ut, yt, lamt, nut, lamt_fixed, nut_fixed = input

      ell_x, ell_u = jax.grad(self._ell, argnums=(0,1))(xt, ut)

      (ell_xx, ell_xu), (ell_ux, ell_uu) = jax.hessian(self._ell, argnums=(0,1))(xt, ut)

      g = self._g(xt, ut, yt)
      g_x, g_u, g_y = jax.jacfwd(self._g, argnums=(0,1,2))(xt, ut, yt)

      h = self._h(xt, ut)
      h_x, h_u = jax.jacfwd(self._h, argnums=(0,1))(xt, ut)

      lamt_plus = lamt_fixed + g / penalty

      nut_plus = nut_fixed + h / penalty
      active = nut_plus > 0
      nut_plus = jp.where(active, nut_plus, 0.0)

      Q_x = ell_x + g_x.T @ lamt + (h_x.T * active) @ nut
      Q_xx = ell_xx
      Q_xu = ell_xu
      Q_xy = jp.zeros((_nx, _nx))

      Q_u = ell_u + g_u.T @ lamt + (h_u.T * active) @ nut
      Q_ux = ell_ux
      Q_uu = ell_uu
      Q_uy = jp.zeros((_nu, _nx))

      Q_y = V_x + g_y.T @ lamt
      Q_yy = V_xx
      Q_yx = Q_xy.T
      Q_yu = Q_uy.T

      M = jp.block([
          [Q_uu, Q_uy, g_u.T, h_u.T * active],
          [Q_yu, Q_yy, g_y.T, jp.zeros((_nx, _nh))],
          [g_u, g_y, -penalty * jp.eye(_ng), jp.zeros((_ng, _nh))],
          [h_u * active[:, None], jp.zeros((_nh, _nx)), jp.zeros((_nh, _ng)), -penalty * jp.eye(_nh)],
      ])

      decompM = jax.scipy.linalg.lu_factor(M)

      kt = jax.scipy.linalg.lu_solve(
          decompM,
          -jp.concatenate([
              Q_u,
              Q_y,
              penalty * (lamt_plus - lamt),
              penalty * (nut_plus - nut),
          ])
      )

      Kt = jax.scipy.linalg.lu_solve(
          decompM,
          -jp.vstack([
              Q_ux,
              Q_yx,
              g_x,
              h_x * active[:, None],
          ])
      )

      Q_xw = jp.hstack([Q_xu, Q_xy, g_x.T, h_x.T * active])
      V_x = Q_x + Q_xw @ kt
      V_xx = Q_xx + Q_xw @ Kt

      return (V_x, V_xx), (kt, Kt)

    (V_x, V_xx), (k, K) = jax.lax.scan(
        backward_step,
        (V_x, V_xx),
        (
            x_nominal[:-1],
            u_nominal,
            x_nominal[1:],
            lam_nominal[0],
            nu_nominal[0],
            lam_fixed[0],
            nu_fixed[0],
        ),
        reverse=True,
    )

    delta_x0, delta_lam0 = jp.split(
        jax.scipy.linalg.solve(
            jp.block([
                [V_xx, jp.eye(_nx)],
                [jp.eye(_nx), -penalty * jp.eye(_nx)],
            ]),
            -jp.concatenate([
                V_x + lam_nominal[-1],
                penalty * (lam_fixed[-1] - lam_nominal[-1]) + (x_nominal[0] - x0_hat),
            ]),
        ),
        [_nx],
    )

    def forward_step(carry, input):
      delta_xt = carry
      kt, Kt = input

      delta_ut, delta_yt, delta_lamt, delta_nut = jp.split(
          kt + Kt @ delta_xt,
          [_nu, _nu + _nx, _nu + _nx + _ng],
      )

      return delta_yt, (delta_ut, delta_yt, delta_lamt, delta_nut)

    _, (delta_u, delta_x, delta_lam, delta_nu) = jax.lax.scan(
        forward_step,
        delta_x0,
        (k, K),
    )

    delta_x = jp.vstack([delta_x0[None, :], delta_x])

    delta_lam = (delta_lam, delta_lam0)

    delta_nuN = c + C @ delta_x[-1]
    delta_nu = (delta_nu, delta_nuN)

    return delta_x, delta_u, delta_lam, delta_nu


  def _line_search(
      self,
      x0_hat: jax.Array,
      lam_fixed: tuple[jax.Array, jax.Array],
      nu_fixed: tuple[jax.Array, jax.Array],
      penalty: float,
      x_nominal: jax.Array,
      u_nominal: jax.Array,
      lam_nominal: tuple[jax.Array, jax.Array],
      nu_nominal: tuple[jax.Array, jax.Array],
      delta_x: jax.Array,
      delta_u: jax.Array,
      delta_lam: tuple[jax.Array, jax.Array],
      delta_nu: tuple[jax.Array, jax.Array],
  ):

    def squared(vec):
      return vec.dot(vec)

    def merit(
        x: jax.Array,
        u: jax.Array,
        lam: tuple[jax.Array, jax.Array],
        nu: tuple[jax.Array, jax.Array],
    ):
      val = jp.sum(jax.vmap(self._ell)(x[:-1], u))

      val += self._ellf(x[-1])

      val += jp.sum(
          jax.vmap(
              lambda xt, ut, xtp1, lamt, lamt_fixed: (
                  squared(self._g(xt, ut, xtp1) + penalty * (lamt - lamt_fixed / 2)) / penalty
                  + squared(lamt) * penalty / 4
              )
          )(x[:-1], u, x[1:], lam[0], lam_fixed[0])
      )
      val += (
          squared(x[0] - x0_hat + penalty * (lam[-1] - lam_fixed[-1] / 2)) / penalty
          + squared(lam[-1]) * penalty / 4
      )
      val += jp.sum(
          jax.vmap(
              lambda xt, ut, nut, nut_fixed: (
                  squared(jp.maximum(self._h(xt, ut) + penalty * (nut - nut_fixed / 2), 0.0)) / penalty
                  + squared(nut) * penalty / 4
              )
          )(x[:-1], u, nu[0], nu_fixed[0])
      )
      val += (
          squared(self._hf(x[-1]) + penalty * (nu[-1] - nu_fixed[-1] / 2)) / penalty
          + squared(nu[-1]) * penalty / 4
      )

      return val

    def step(alpha):
      return (
          x_nominal + alpha * delta_x,
          u_nominal + alpha * delta_u,
          (
              lam_nominal[0] + alpha * delta_lam[0],
              lam_nominal[1] + alpha * delta_lam[1],
          ),
          (
              nu_nominal[0] + alpha * delta_nu[0],
              nu_nominal[1] + alpha * delta_nu[1],
          ),
      )

    merit_nominal, merit_grad = jax.value_and_grad(
        merit,
        argnums=(0, 1, 2, 3),
    )(*step(0.0))

    merit_slope = sum(
        jp.vdot(g, d)
        for g, d in zip(
            jax.tree_util.tree_leaves(merit_grad),
            jax.tree_util.tree_leaves((delta_x, delta_u, delta_lam, delta_nu)),
        )
    )

    # Armijo parameters
    c1 = 1e-4
    rho = 0.5
    max_backtracks = 20

    def cond_fn(carry):
      alpha, merit_trial, i = carry
      return (
          (merit_trial > merit_nominal + c1 * alpha * merit_slope)
          & (i < max_backtracks)
      )

    def body_fn(carry):
      alpha, _, i = carry
      alpha = rho * alpha
      merit_trial = merit(*step(alpha))
      return alpha, merit_trial, i + 1

    alpha = 1.0
    merit_trial = merit(*step(alpha))

    alpha, merit_trial, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (alpha, merit_trial, 0.0),
    )

    return step(alpha), alpha
