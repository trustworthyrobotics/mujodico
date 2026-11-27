"""Initialize elastic restoration and reconstruct original-problem multipliers.

With a = μ_R/(2ρ), use q = a + a²/(hypot(a,c/2)+|c|/2),
p = q + max(−c,0), n = q + max(c,0). This is FATROP's central
initialization written without cancellation for large positive/negative c.
The original trajectory/slacks are retained and μ_R = max(μ, ‖c‖∞).
"""

from dataclasses import replace

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import equality_constrained_lqr as lqr
from mujodico.trajopt._src.ippas import ip_linesearch as ls
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas import ip_primal_dual_system_resto as resto
from mujodico.trajopt._src.ippas.ip_iterate import Iterate, all_finite, max_abs, distances, damping, complementarity
from mujodico.trajopt._src.ippas.ip_search_dir import dual_regularization
from mujodico.trajopt._src.ippas.ip_iterate import IterateData
from mujodico.trajopt._src.ippas.options import Options
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel
from mujodico.trajopt._src.types import Params


def elastic_pair(
    c: jax.Array, mu: jax.Array, rho: float
) -> tuple[jax.Array, jax.Array]:
  a = mu / (2 * rho)
  q = a + a * (a / (jp.hypot(a, c / 2) + jp.abs(c) / 2))
  return q + jp.maximum(-c, 0), q + jp.maximum(c, 0)


def initialize(
    it: Iterate, data: IterateData, mu: jax.Array, options: Options
) -> tuple[resto.RestorationIterate, jax.Array]:
  """Return a strictly positive elastic iterate and its independent barrier μ_R."""
  mu_r = jp.maximum(
      mu, max_abs(data.r0, data.r_f, data.r_e, data.r_g, data.r_gf)
  )
  c = resto.ElasticBlocks(data.r_e, data.r_g, data.r_gf)
  p = resto.tree(
      lambda c: elastic_pair(c, mu_r, options.resto_penalty_parameter)[0], c
  )
  n = resto.tree(
      lambda c: elastic_pair(c, mu_r, options.resto_penalty_parameter)[1], c
  )
  base = replace(
      it,
      lmbda=jp.zeros_like(it.lmbda),
      beta=jp.zeros_like(it.beta),
      y=jp.zeros_like(it.y),
      yf=jp.zeros_like(it.yf),
      **{
          name: jp.minimum(getattr(it, name), options.resto_penalty_parameter)
          for name in ('z_l', 'z_u', 'zf_l', 'zf_u')
      },
  )
  return (
      resto.RestorationIterate(
          base,
          p,
          n,
          resto.tree(lambda x: mu_r / x, p),
          resto.tree(lambda x: mu_r / x, n),
      ),
      mu_r,
  )


def initialize_original_multipliers(
    model: StagewiseOcpModel,
    entry: Iterate,
    point: Iterate,
    params: Params,
    mu: jax.Array,
    bounds: resto.Bounds,
    options: Options,
) -> Iterate:
  """FATROP-like return: total-displacement bound step, then joint LS duals.

  Estimate λ, β, y, yf together using a unit Hessian on (x,u,s,sf).
  Link constraints eliminate s with unit weight. This projection includes the
  original objective and slack damping, never restoration multipliers.
  """
  dz = []
  for s, old, zl, zu, lo, hi in (
      (point.s, entry.s, entry.z_l, entry.z_u, *bounds[:2]),
      (point.sf, entry.sf, entry.zf_l, entry.zf_u, *bounds[2:]),
  ):
    dl, du = distances(old, lo, hi)
    cl, cu = complementarity(old, zl, zu, lo, hi, mu)
    dz.extend((
        jp.where(jp.isfinite(lo), -(cl + zl * (s - old)) / dl, 0.0),
        jp.where(jp.isfinite(hi), -(cu - zu * (s - old)) / du, 0.0),
    ))
  alpha = jp.asarray(1.0, mu.dtype)
  for z, d, bound in zip(
      (entry.z_l, entry.z_u, entry.zf_l, entry.zf_u), dz, bounds
  ):
    alpha = jp.minimum(
        alpha,
        ls.maximum_step(
            z,
            d,
            jp.where(jp.isfinite(bound), 0.0, -jp.inf),
            jp.full_like(z, jp.inf),
            jp.maximum(options.tau_min, 1 - mu),
        ),
    )
  zs = [
      z + alpha * d
      for z, d in zip((entry.z_l, entry.z_u, entry.zf_l, entry.zf_u), dz)
  ]
  reset = max_abs(*zs) > options.bound_mult_reset_threshold
  final = []
  for z, distance, bound in zip(
      zs,
      (*distances(point.s, *bounds[:2]), *distances(point.sf, *bounds[2:])),
      bounds,
  ):
    central = mu / distance
    final.append(
        jp.where(
            jp.isfinite(bound),
            jp.clip(
                jp.where(reset, 1.0, z),
                central / options.kappa_sigma,
                central * options.kappa_sigma,
            ),
            0.0,
        )
    )
  it = replace(
      point,
      z_l=final[0],
      z_u=final[1],
      zf_l=final[2],
      zf_u=final[3],
      lmbda=jp.zeros_like(point.lmbda),
      beta=jp.zeros_like(point.beta),
      y=jp.zeros_like(point.y),
      yf=jp.zeros_like(point.yf),
  )
  b, _ = pd.evaluate(model, it, params, mu, bounds, options)
  gx, gu = b.Gx.swapaxes(-1, -2), b.Gu.swapaxes(-1, -2)
  reduced = lqr.EqualityConstrainedLQR.create(
      Q=jp.broadcast_to(jp.eye(model.nx_tangent, dtype=it.x.dtype), b.Hxx.shape)
      + jp.concatenate((gx @ b.Gx, (b.Gf.T @ b.Gf)[None])),
      N=gx @ b.Gu,
      R=jp.broadcast_to(jp.eye(model.nu, dtype=it.u.dtype), b.Huu.shape)
      + gu @ b.Gu,
      q=b.r_x
      + jp.concatenate((pd._matvec(gx, b.r_s), (b.Gf.T @ b.r_sf)[None])),
      r=b.r_u + pd._matvec(gu, b.r_s),
      A=b.A,
      B=b.B,
      Ex=b.Ex,
      Eu=b.Eu,
      c=jp.zeros_like(b.r_f),
      e=jp.zeros_like(b.r_e),
      x0=jp.zeros_like(b.r0),
      delta_c=dual_regularization(b.Eu, mu, options),
  )
  solved = lqr.solve(reduced)
  lam = solved.lmbda.at[0].set(-jp.linalg.solve(b.E0.T, solved.lmbda[0]))
  y = pd._matvec(b.Gx, solved.x[:-1]) + pd._matvec(b.Gu, solved.u) + b.r_s
  yf = b.Gf @ solved.x[-1] + b.r_sf
  estimates = (lam, solved.beta, y, yf)
  good = all_finite(estimates) & (max_abs(*estimates) <= options.lam_max)
  return replace(
      it,
      **{
          name: jp.where(good, val, jp.zeros_like(val))
          for name, val in zip(('lmbda', 'beta', 'y', 'yf'), estimates)
      },
  )
