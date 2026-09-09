"""
ProxDDP solves the following problem
    minimizeₓ,ᵤ  ∑ₜ₌₀ᴺ⁻¹ ℓ(xₜ,uₜ) + ℓf(x_N)
    subject to   g(xₜ, uₜ, xₜ₊₁) = 0   t=0,...N−1
                 gi(x₀) = 0
                 h(xₜ, uₜ) ≤ 0        t=0,...N−1
                 hf(x_N) ≤ 0
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import jax
import jax.numpy as jp
import numpy as np


class ProxDDP:

  def __init__(
      self,
      stage_cost: Callable[[jax.Array, jax.Array, Any], jax.Array],
      terminal_cost: Callable[[jax.Array, Any], jax.Array],
      implicit_dynamics: Callable[[jax.Array, jax.Array, jax.Array, Any], jax.Array],
      initial_state_constraint: Callable[[jax.Array, Any], jax.Array],
      stage_constraint: Callable[[jax.Array, jax.Array, Any], jax.Array] | None,
      terminal_constraint: Callable[[jax.Array, Any], jax.Array] | None,
      horizon: int,
      num_states: int,
      num_controls: int,
      params_example: Any = None,
  ):
    if stage_constraint is None:
      stage_constraint = lambda x, u, params: jp.empty(0)
    if terminal_constraint is None:
      terminal_constraint = lambda x, params: jp.empty(0)

    self._ell = stage_cost
    self._ellf = terminal_cost
    self._g = implicit_dynamics
    self._gi = initial_state_constraint
    self._h = stage_constraint
    self._hf = terminal_constraint
    self._N = horizon
    self._nx = num_states
    self._nu = num_controls

    x = jp.zeros(num_states)
    u = jp.zeros(num_controls)
    params = params_example
    self._ng = self._g(x, u, x, params).size
    self._ngi = self._gi(x, params).size
    self._nh = self._h(x, u, params).size
    self._nhf = self._hf(x, params).size


  @dataclass(frozen=True)
  class Options:
    max_iters: int = 500
    parallel_associative_scan: bool = True
    verbose: bool = True


  def solve(
      self,
      params: Any,
      x_init: jax.Array | None = None,
      u_init: jax.Array | None = None,
      lam_init: tuple[jax.Array, jax.Array] | None = None,
      nu_init: tuple[jax.Array, jax.Array] | None = None,
      options: Options = Options(),
  ):
    if x_init is None:
      x_init = jp.zeros((self._N + 1, self._nx))
    if u_init is None:
      u_init = jp.zeros((self._N, self._nu))
    if lam_init is None:
      lam_init = jp.zeros((self._N, self._ng)), jp.zeros(self._ngi)
    if nu_init is None:
      nu_init = jp.zeros((self._N, self._nh)), jp.zeros(self._nhf)

    def cond_fn(carry):
      _, _, log = carry
      return log.iters < options.max_iters

    def body_fn(carry):
      (x, u, lam, nu), penalty, log = carry

      (x, u, lam_new, nu_new), log = self._minimize_augmented_lagrangian(
          params=params,
          lam_proximal=lam,
          nu_proximal=nu,
          penalty=penalty,
          x_init=x,
          u_init=u,
          log=log,
          parallel=options.parallel_associative_scan,
      )

      lam = lam_new
      nu = nu_new
      penalty = jp.maximum(penalty * 0.1, 1e-6)

      return (x, u, lam, nu), penalty, log

    penalty = 1.0
    (x, u, lam, nu), penalty, log = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (
            (x_init, u_init, lam_init, nu_init),
            penalty,
            Log.create(options.max_iters * 2),
        ),
    )

    if options.verbose:
      jax.debug.callback(lambda log: print(' ' * 33 + 'ProxDDP\n' + str(log)), log)

    return x, u


  def _minimize_augmented_lagrangian(
      self,
      params: Any,
      lam_proximal: tuple[jax.Array, jax.Array],
      nu_proximal: tuple[jax.Array, jax.Array],
      penalty: float,
      x_init: jax.Array,
      u_init: jax.Array,
      log: 'Log',
      *,
      max_iters: int = 10,
      parallel: bool = True,
  ):
    """
    Mimimizes
        ℒ_μ(x, u; λ, ν) = ∑ₜ₌₀ᴺ⁻¹ ℓ(xₜ,uₜ) + ℓf(x_N)
            + ∑ₜ₌₀ᴺ⁻¹ 1/(2μ) ‖g(xₜ, uₜ, xₜ₊₁) + μ λₜ‖²
            + 1/(2μ) ‖gi(x₀) + μ λ₀‖²
            + ∑ₜ₌₀ᴺ⁻¹ 1/(2μ) ‖[h(xₜ, uₜ) + μ νₜ]₊‖²
            + 1/(2μ) ‖[hf(x_N) + μ ν_N]₊‖²
    where μ is the penalty parameter.
    """
    assert lam_proximal[0].shape == (self._N, self._ng)
    assert lam_proximal[1].shape == (self._ngi,)
    assert nu_proximal[0].shape == (self._N, self._nh)
    assert nu_proximal[1].shape == (self._nhf,)
    assert jp.isscalar(penalty)
    assert x_init.shape == (self._N + 1, self._nx)
    assert u_init.shape == (self._N, self._nu)

    x = x_init
    u = u_init
    lam = lam_proximal
    nu = nu_proximal

    def cond_fn(carry: tuple[tuple, Log, int]):
      _, log, i = carry
      return ((i == 0) | (log.step_size > 1e-6)) & (i < max_iters)

    def body_fn(carry: tuple[tuple, Log, int]):
        (x, u, lam, nu), log, i = carry

        newton_direction_fun = (
            self._newton_direction_parallel
            if parallel
            else self._newton_direction_serial
        )

        delta_x, delta_u, delta_lam, delta_nu = newton_direction_fun(
            params=params,
            lam_proximal=lam_proximal,
            nu_proximal=nu_proximal,
            penalty=penalty,
            x_nominal=x,
            u_nominal=u,
            lam_nominal=lam,
            nu_nominal=nu,
        )

        (x, u, lam, nu), alpha, ls_iters = self._line_search(
            params=params,
            lam_proximal=lam_proximal,
            nu_proximal=nu_proximal,
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

        objective = (
            jp.sum(jax.vmap(lambda xt, ut: self._ell(xt, ut, params))(x[:-1], u))
            + self._ellf(x[-1], params)
        )

        objective, primal_infeasibility, dual_infeasibility \
        = self._calc_objective_primal_dual_infeasibility(
            params=params, penalty=penalty, x=x, u=u, lam=lam, nu=nu,
        )

        step_inf_norm = jp.maximum(
            jp.linalg.norm(delta_x, ord=jp.inf),
            jp.linalg.norm(delta_u, ord=jp.inf),
        )

        log = log.append(
            objective=objective,
            primal_infeasibility=primal_infeasibility,
            dual_infeasibility=dual_infeasibility,
            penalty_parameter=penalty,
            step_inf_norm=step_inf_norm,
            step_size=alpha,
            line_search_trials=ls_iters,
        )

        return (x, u, lam, nu), log, i + 1

    (x, u, lam, nu), log, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (
            (x, u, lam, nu),
            log,
            0
        ),
    )

    return (x, u, lam, nu), log


  def _newton_direction_serial(
      self,
      params: Any,
      lam_proximal: tuple[jax.Array, jax.Array],
      nu_proximal: tuple[jax.Array, jax.Array],
      penalty: float,
      x_nominal: jax.Array,
      u_nominal: jax.Array,
      lam_nominal: tuple[jax.Array, jax.Array],
      nu_nominal: tuple[jax.Array, jax.Array],
  ):
    _nx, _nu = self._nx, self._nu
    _ng = self._ng
    _nh = self._nh

    def terminal_stage(xN, nuf_proximal, nuf):
      ellf_x = jax.grad(self._ellf)(xN, params)
      ellf_xx = jax.hessian(self._ellf)(xN, params)
      hf_x = jax.jacfwd(self._hf)(xN, params)

      hf = self._hf(xN, params)
      nuf_plus = nuf_proximal + hf / penalty
      active = nuf_plus > 0
      nuf_plus = jp.where(active, nuf_plus, 0.0)

      hf_x = hf_x * active[:, None]

      V_x = ellf_x + hf_x.T @ nuf_plus
      V_xx = ellf_xx + hf_x.T @ hf_x / penalty

      knuf = nuf_plus - nuf
      Knuf = (hf_x * active[:, None]) / penalty

      return V_x, V_xx, knuf, Knuf

    V_x, V_xx, knuf, Knuf = terminal_stage(
        xN=x_nominal[-1],
        nuf_proximal=nu_proximal[-1],
        nuf=nu_nominal[-1],
    )

    def backward_step(carry, input):
      V_x, V_xx = carry
      xt, ut, yt, lamt, nut, lamt_proximal, nut_proximal = input

      ell_x, ell_u = jax.grad(self._ell, argnums=(0,1))(xt, ut, params)

      (ell_xx, ell_xu), (ell_ux, ell_uu) = jax.hessian(self._ell, argnums=(0,1))(xt, ut, params)

      g = self._g(xt, ut, yt, params)
      g_x, g_u, g_y = jax.jacfwd(self._g, argnums=(0,1,2))(xt, ut, yt, params)

      h = self._h(xt, ut, params)
      h_x, h_u = jax.jacfwd(self._h, argnums=(0,1))(xt, ut, params)

      lamt_plus = lamt_proximal + g / penalty

      nut_plus = nut_proximal + h / penalty
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
            lam_proximal[0],
            nu_proximal[0],
        ),
        reverse=True,
    )

    def initial_stage(x0, lami_proximal, lami, V0_x, V0_xx):
      gi = self._gi(x0, params)
      gi_x = jax.jacfwd(self._gi)(x0, params)

      delta_x0, delta_lami = jp.split(
          jax.scipy.linalg.solve(
              jp.block([
                  [V0_xx, gi_x.T],
                  [gi_x, -penalty * jp.eye(self._ngi)],
              ]),
              -jp.concatenate([
                  V0_x + gi_x.T @ lami,
                  penalty * (lami_proximal - lami) + gi,
              ]),
          ),
          [_nx],
      )
      return delta_x0, delta_lami

    delta_x0, delta_lami = initial_stage(
        x0=x_nominal[0],
        lami_proximal=lam_proximal[-1],
        lami=lam_nominal[-1],
        V0_x=V_x,
        V0_xx=V_xx,
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

    delta_lam = (delta_lam, delta_lami)

    delta_nuf = knuf + Knuf @ delta_x[-1]
    delta_nu = (delta_nu, delta_nuf)

    return delta_x, delta_u, delta_lam, delta_nu


  def _newton_direction_parallel(
      self,
      params: Any,
      lam_proximal: tuple[jax.Array, jax.Array],
      nu_proximal: tuple[jax.Array, jax.Array],
      penalty: float,
      x_nominal: jax.Array,
      u_nominal: jax.Array,
      lam_nominal: tuple[jax.Array, jax.Array],
      nu_nominal: tuple[jax.Array, jax.Array],
  ):
    _nx, _nu = self._nx, self._nu
    _ng = self._ng
    _nh = self._nh

    def terminal_stage(xN, nuf_proximal, nuf):
      ellf_x = jax.grad(self._ellf)(xN, params)
      ellf_xx = jax.hessian(self._ellf)(xN, params)
      hf_x = jax.jacfwd(self._hf)(xN, params)

      hf = self._hf(xN, params)
      nuf_plus = nuf_proximal + hf / penalty
      active = nuf_plus > 0
      nuf_plus = jp.where(active, nuf_plus, 0.0)

      hf_x = hf_x * active[:, None]

      pN = ellf_x + hf_x.T @ nuf_plus
      PN = ellf_xx + hf_x.T @ hf_x / penalty

      knuf = nuf_plus - nuf
      Knuf = (hf_x * active[:, None]) / penalty

      return pN, PN, knuf, Knuf

    pN, PN, knuf, Knuf = terminal_stage(
        xN=x_nominal[-1],
        nuf_proximal=nu_proximal[-1],
        nuf=nu_nominal[-1],
    )

    def eval_cost_dynamics(xt, ut, yt, lamt_proximal, nut_proximal):
      ellt_x, ellt_u = jax.grad(self._ell, argnums=(0,1))(xt, ut, params)

      (ellt_xx, ellt_xu), (ellt_ux, ellt_uu) = jax.hessian(self._ell, argnums=(0,1))(xt, ut, params)

      gt_x, gt_u, gt_y = jax.jacfwd(self._g, argnums=(0,1,2))(xt, ut, yt, params)

      ht_x, ht_u = jax.jacfwd(self._h, argnums=(0,1))(xt, ut, params)

      gt = self._g(xt, ut, yt, params)
      lamt_plus = lamt_proximal + gt / penalty

      ht = self._h(xt, ut, params)
      nut_plus = nut_proximal + ht / penalty
      active = nut_plus > 0
      nut_plus = jp.where(active, nut_plus, 0.0)

      ht_x = ht_x * active[:, None]
      ht_u = ht_u * active[:, None]

      return ellt_x, ellt_u, ellt_xx, ellt_xu, ellt_ux, ellt_uu, gt_x, gt_u, gt_y, ht_x, ht_u, lamt_plus, nut_plus

    (
        ell_x, ell_u,
        ell_xx, ell_xu, ell_ux, ell_uu,
        g_x, g_u, g_y,
        h_x, h_u,
        lam_plus, nu_plus
    ) = jax.vmap(eval_cost_dynamics)(
        x_nominal[:-1],
        u_nominal,
        x_nominal[1:],
        lam_proximal[0],
        nu_proximal[0]
    )

    def stage_conditional_value_funcion(
        ellt_x, ellt_u,
        ellt_xx, ellt_xu, ellt_ux, ellt_uu,
        gt_x, gt_u, gt_y,
        ht_x, ht_u,
        lamt_plus, nut_plus,
        lamt, nut,
    ):
      M = jp.block([
          [ellt_uu, gt_u.T, ht_u.T],
          [gt_u, -penalty * jp.eye(_ng), jp.zeros((_ng, _nh))],
          [ht_u, jp.zeros((_nh, _ng)), -penalty * jp.eye(_nh)],
      ])

      Ex = jp.vstack([
          ellt_ux,
          gt_x,
          ht_x,
      ])

      Ey = jp.vstack([
          jp.zeros((_nu, _nx)),
          gt_y,
          jp.zeros((_nh, _nx)),
      ])

      r = jp.concatenate([
          ellt_u + gt_u.T @ lamt + ht_u.T @ nut,
          penalty * (lamt_plus - lamt),
          penalty * (nut_plus - nut),
      ])

      decompM = jax.scipy.linalg.lu_factor(M)
      Minv_Ex = jax.scipy.linalg.lu_solve(decompM, Ex)
      Minv_Ey = jax.scipy.linalg.lu_solve(decompM, Ey)
      Minv_r = jax.scipy.linalg.lu_solve(decompM, r)

      Dxx = -Ex.T @ Minv_Ex + ellt_xx
      Dxy = -Ex.T @ Minv_Ey
      Dyx = -Ey.T @ Minv_Ex
      Dyy = -Ey.T @ Minv_Ey
      dx = -Ex.T @ Minv_r + ellt_x
      dy = -Ey.T @ Minv_r

      decompDyy = jax.scipy.linalg.lu_factor(Dyy)
      Dyyinv = jax.scipy.linalg.lu_solve(decompDyy, jp.eye(_nx))
      Dyyinv_Dyx = jax.scipy.linalg.lu_solve(decompDyy, Dyx)
      Dyyinv_dy = jax.scipy.linalg.lu_solve(decompDyy, dy)

      Pt = Dxx - Dxy @ Dyyinv_Dyx
      pt = dx - Dxy @ Dyyinv_dy
      Ct = Dyyinv
      At = -Dyyinv_Dyx
      bt = -Dyyinv_dy

      return Pt, pt, Ct, At, bt

    P, p, C, A, b = jax.vmap(stage_conditional_value_funcion)(
        ell_x, ell_u,
        ell_xx, ell_xu, ell_ux, ell_uu,
        g_x, g_u, g_y,
        h_x, h_u,
        lam_plus, nu_plus,
        lam_nominal[0], nu_nominal[0],
    )

    P = jp.concatenate((P, PN[None, :, :]))
    p = jp.concatenate((p, pN[None, :]))
    C = jp.concatenate((C, jp.zeros((1, _nx, _nx))))
    A = jp.concatenate((A, jp.zeros((1, _nx, _nx))))
    b = jp.concatenate((b, jp.zeros((1, _nx))))

    def conditional_value_funcion_composition(elem2, elem1):
      P1, p1, C1, A1, b1 = elem1
      P2, p2, C2, A2, b2 = elem2

      Slinv = jax.scipy.linalg.lu_factor(jp.eye(_nx) + P2 @ C1)
      Srinv = jax.scipy.linalg.lu_factor(jp.eye(_nx) + C1 @ P2)

      P = A1.T @ jax.scipy.linalg.lu_solve(Slinv, P2 @ A1) + P1
      p = A1.T @ jax.scipy.linalg.lu_solve(Slinv, p2 + P2 @ b1) + p1
      C = A2 @ jax.scipy.linalg.lu_solve(Srinv, C1 @ A2.T) + C2
      A = A2 @ jax.scipy.linalg.lu_solve(Srinv, A1)
      b = A2 @ jax.scipy.linalg.lu_solve(Srinv, b1 - C1 @ p2) + b2

      return P, p, C, A, b

    V_xx, V_x, _, _, _ = jax.lax.associative_scan(
        jax.vmap(conditional_value_funcion_composition),
        (P, p, C, A, b),
        reverse=True,
    )

    def calc_gain(
        ellt_x, ellt_u,
        ellt_xx, ellt_xu, ellt_ux, ellt_uu,
        gt_x, gt_u, gt_y,
        ht_x, ht_u,
        lamt_plus, nut_plus,
        lamt, nut,
        Vtp1_x, Vtp1_xx,
    ):
      M = jp.block([
          [ellt_uu, jp.zeros((_nu, _nx)), gt_u.T, ht_u.T],
          [jp.zeros((_nx, _nu)), Vtp1_xx, gt_y.T, jp.zeros((_nx, _nh))],
          [gt_u, gt_y, -penalty * jp.eye(_ng), jp.zeros((_ng, _nh))],
          [ht_u, jp.zeros((_nh, _nx)), jp.zeros((_nh, _ng)), -penalty * jp.eye(_nh)],
      ])

      decompM = jax.scipy.linalg.lu_factor(M)

      kt = jax.scipy.linalg.lu_solve(
          decompM,
          -jp.concatenate([
              ellt_u + gt_u.T @ lamt + ht_u.T @ nut,
              Vtp1_x + gt_y.T @ lamt,
              penalty * (lamt_plus - lamt),
              penalty * (nut_plus - nut),
          ])
      )

      Kt = jax.scipy.linalg.lu_solve(
          decompM,
          -jp.vstack([
              ellt_ux,
              jp.zeros((_nx, _nx)),
              gt_x,
              ht_x,
          ])
      )

      return kt, Kt

    k, K = jax.vmap(calc_gain)(
        ell_x, ell_u,
        ell_xx, ell_xu, ell_ux, ell_uu,
        g_x, g_u, g_y,
        h_x, h_u,
        lam_plus, nu_plus,
        lam_nominal[0], nu_nominal[0],
        V_x[1:], V_xx[1:],
    )

    def initial_stage(x0, lami_proximal, lami, V0_x, V0_xx):
      gi = self._gi(x0, params)
      gi_x = jax.jacfwd(self._gi)(x0, params)

      delta_x0, delta_lami = jp.split(
          jax.scipy.linalg.solve(
              jp.block([
                  [V0_xx, gi_x.T],
                  [gi_x, -penalty * jp.eye(self._ngi)],
              ]),
              -jp.concatenate([
                  V0_x + gi_x.T @ lami,
                  penalty * (lami_proximal - lami) + gi,
              ]),
          ),
          [_nx],
      )
      return delta_x0, delta_lami

    delta_x0, delta_lami = initial_stage(
        x0=x_nominal[0],
        lami_proximal=lam_proximal[-1],
        lami=lam_nominal[-1],
        V0_x=V_x[0],
        V0_xx=V_xx[0],
    )

    kx = k[:, _nu:_nu+_nx]
    Kx = K[:, _nu:_nu+_nx]
    kx = jp.concatenate((delta_x0[None, :], kx))
    Kx = jp.concatenate((jp.eye(_nx)[None, :, :], Kx))

    def gain_composition(elem1, elem2):
      k1, K1 = elem1
      k2, K2 = elem2
      k = k2 + K2 @ k1
      K = K2 @ K1
      return k, K

    delta_x, _ = jax.lax.associative_scan(
        jax.vmap(gain_composition),
        (kx, Kx),
    )

    def calc_delta(delta_xt, kt, Kt):
      delta_ut, delta_yt, delta_lamt, delta_nut = jp.split(
          kt + Kt @ delta_xt,
          [_nu, _nu + _nx, _nu + _nx + _ng],
      )
      return delta_ut, delta_lamt, delta_nut

    delta_u, delta_lam, delta_nu = jax.vmap(calc_delta)(delta_x[:-1], k, K)

    delta_lam = (delta_lam, delta_lami)

    delta_nuf = knuf + Knuf @ delta_x[-1]
    delta_nu = (delta_nu, delta_nuf)

    return delta_x, delta_u, delta_lam, delta_nu


  def _calc_objective_primal_dual_infeasibility(
      self,
      params: Any,
      penalty: float,
      x: jax.Array,
      u: jax.Array,
      lam: tuple[jax.Array, jax.Array],
      nu: tuple[jax.Array, jax.Array],
  ) -> tuple[float, float, float]:

    def lagrangian_fun(x, u, lam, nu):
      g = jax.vmap(
          lambda xt, ut, xtp1: self._g(xt, ut, xtp1, params)
      )(x[:-1], u, x[1:])
      gi = self._gi(x[0], params)

      h = jax.vmap(
          lambda xt, ut: self._h(xt, ut, params)
      )(x[:-1], u)
      hf = self._hf(x[-1], params)

      objective = (
          jp.sum(jax.vmap(lambda xt, ut: self._ell(xt, ut, params))(x[:-1], u))
          + self._ellf(x[-1], params)
      )

      lagrangian = (
          objective
          + jp.sum(g * lam[0])
          + jp.sum(gi * lam[1])
          + jp.sum(h * nu[0])
          + jp.sum(hf * nu[1])
      )

      return lagrangian, (objective, g, gi, h, hf)

    def inf_norm(*vecs):
      return jp.max(jp.array([
          jp.linalg.norm(vec.ravel(), ord=jp.inf)
          for vec in jax.tree_util.tree_leaves(vecs)
      ]))

    (
        (lagrangian, (objective, g, gi, h, hf)),
        (lagrangian_x, lagrangian_u)
    ) = jax.value_and_grad(
        lagrangian_fun, argnums=(0, 1), has_aux=True
    )(x, u, lam, nu)

    primal_infeasibility = inf_norm(
        g,
        gi,
        jp.minimum(h + penalty * nu[0], 0.0) - h,
        jp.minimum(hf + penalty * nu[-1], 0.0) - hf,
    )

    dual_infeasibility = inf_norm(
        lagrangian_x,
        lagrangian_u,
    )

    return objective, primal_infeasibility, dual_infeasibility


  def _line_search(
      self,
      params: Any,
      lam_proximal: tuple[jax.Array, jax.Array],
      nu_proximal: tuple[jax.Array, jax.Array],
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
      val = self._ellf(x[-1], params)
      val += jp.sum(
          jax.vmap(
              lambda xt, ut: self._ell(xt, ut, params)
          )(x[:-1], u)
      )
      val += jp.sum(
          jax.vmap(
              lambda xt, ut, xtp1, lamt, lamt_proximal: (
                  squared(self._g(xt, ut, xtp1, params) + penalty * (lamt - lamt_proximal / 2)) / penalty
                  + squared(lamt) * penalty / 4
              )
          )(x[:-1], u, x[1:], lam[0], lam_proximal[0])
      )
      val += (
          squared(self._gi(x[0], params) + penalty * (lam[-1] - lam_proximal[-1] / 2)) / penalty
          + squared(lam[-1]) * penalty / 4
      )
      val += jp.sum(
          jax.vmap(
              lambda xt, ut, nut, nut_proximal: (
                  squared(jp.maximum(self._h(xt, ut, params) + penalty * (nut - nut_proximal / 2), 0.0)) / penalty
                  + squared(nut) * penalty / 4
              )
          )(x[:-1], u, nu[0], nu_proximal[0])
      )
      val += (
          squared(self._hf(x[-1], params) + penalty * (nu[-1] - nu_proximal[-1] / 2)) / penalty
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

    alpha, merit_trial, ls_iters = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (alpha, merit_trial, 0),
    )

    alpha = jp.where(merit_trial < merit_nominal + c1 * alpha * merit_slope, alpha, alpha)

    return step(alpha), alpha, ls_iters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Log:
  iters: int
  _objective: jax.Array
  _primal_infeasibility: jax.Array
  _dual_infeasibility: jax.Array
  _penalty_parameter: jax.Array
  _step_inf_norm: jax.Array
  _step_size: jax.Array
  _line_search_trials: jax.Array

  @property
  def primal_infeasibility(self) -> jax.Array:
    return jp.where(
        self.iters > 0,
        self._primal_infeasibility[self.iters - 1],
        jp.nan,
    )

  @property
  def dual_infeasibility(self) -> jax.Array:
    return jp.where(
        self.iters > 0,
        self._dual_infeasibility[self.iters - 1],
        jp.nan,
    )

  @property
  def step_size(self) -> jax.Array:
    return jp.where(
        self.iters > 0,
        self._step_size[self.iters - 1],
        jp.nan,
    )

  @classmethod
  def create(cls, capacity: int) -> 'Log':
    return Log(
        iters=0,
        _objective=jp.zeros(capacity),
        _primal_infeasibility=jp.zeros(capacity),
        _dual_infeasibility=jp.zeros(capacity),
        _penalty_parameter=jp.zeros(capacity),
        _step_inf_norm=jp.zeros(capacity),
        _step_size=jp.zeros(capacity),
        _line_search_trials=jp.zeros(capacity, dtype=jp.int32),
    )

  def append(
      self,
      objective: float | jax.Array,
      primal_infeasibility: float | jax.Array,
      dual_infeasibility: float | jax.Array,
      penalty_parameter: float | jax.Array,
      step_inf_norm: float | jax.Array,
      step_size: float | jax.Array,
      line_search_trials: int | jax.Array,
  ) -> 'Log':
    i = self.iters
    return Log(
        iters=self.iters + 1,
        _objective=self._objective.at[i].set(objective),
        _primal_infeasibility=self._primal_infeasibility.at[i].set(primal_infeasibility),
        _dual_infeasibility=self._dual_infeasibility.at[i].set(dual_infeasibility),
        _penalty_parameter=self._penalty_parameter.at[i].set(penalty_parameter),
        _step_inf_norm=self._step_inf_norm.at[i].set(step_inf_norm),
        _step_size=self._step_size.at[i].set(step_size),
        _line_search_trials=self._line_search_trials.at[i].set(line_search_trials),
    )

  def __str__(self) -> str:
    n = int(self.iters)

    if n == 0:
        return 'No iterations.'

    objective = np.array(self._objective)
    primal_infeasibility = np.array(self._primal_infeasibility[:n])
    dual_infeasibility = np.array(self._dual_infeasibility[:n])
    log_penalty_parameter = np.log10(np.array(self._penalty_parameter[:n]))
    step_inf_norm = np.array(self._step_inf_norm[:n])
    step_size = np.array(self._step_size[:n])
    line_search_trials = np.array(self._line_search_trials[:n])

    lines = []
    title = 'iter     objective     inf_pr    inf_du  lg(mu)   ||d||    alpha    ls'

    for i in range(n):
      if i % 10 == 0:
        lines.append(title)

      lines.append(
          f'{i:4d}  '
          f'{objective[i]:14.7e}  '
          f'{primal_infeasibility[i]:8.2e}  '
          f'{dual_infeasibility[i]:8.2e}  '
          f'{log_penalty_parameter[i]:4.1f}  '
          f'{step_inf_norm[i]:8.2e}  '
          f'{step_size[i]:8.2e}  '
          f'{line_search_trials[i]:2d}  '
      )

    return "\n".join(lines)
