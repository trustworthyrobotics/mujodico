"""Restoration stopping and original-filter re-entry (FATROP policy).

A return point must reduce original infeasibility and satisfy both the retained
filter and progress relative to the entry point. The original objective-rise
safeguard and Armijo switch are not applied to this feasibility-driven return.
A stationary restoration point with persistent violation signals only local
infeasibility; convergence at a rejected feasible point is restoration failure.
"""

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_filter
from mujodico.trajopt._src.ippas import ip_linesearch as ls
from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import ip_primal_dual_system_resto as pd
from mujodico.trajopt._src.ippas.options import Options


def error(
    system: pd.RestorationSystem,
    it: pd.RestorationIterate,
    mu: jax.Array,
    options: Options,
    *,
    barrier: bool = True,
) -> jax.Array:
  """Scaled full restoration KKT error at μ (or at zero for final convergence)."""
  b = system.base
  target = mu if barrier else jp.asarray(0.0, mu.dtype)
  correction = mu - target
  constraints = ip_iterate.max_abs(b.r0, b.r_f, b.r_e, b.r_g, b.r_gf)
  ds = ip_iterate.damping(b.g_l, b.g_u, correction, options)
  dsf = ip_iterate.damping(b.gf_l, b.gf_u, correction, options)
  dual = ip_iterate.max_abs(
      b.r_x,
      b.r_u,
      b.r_s - ds,
      b.r_sf - dsf,
      *pd.tree(lambda r: r - options.kappa_d * correction, system.r_p),
      *pd.tree(lambda r: r - options.kappa_d * correction, system.r_n),
  )
  comp = ip_iterate.max_abs(
      jp.where(jp.isfinite(b.g_l), b.r_c_l + correction, 0.0),
      jp.where(jp.isfinite(b.g_u), b.r_c_u + correction, 0.0),
      jp.where(jp.isfinite(b.gf_l), b.r_cf_l + correction, 0.0),
      jp.where(jp.isfinite(b.gf_u), b.r_cf_u + correction, 0.0),
      *pd.tree(lambda r: r + correction, system.r_cp),
      *pd.tree(lambda r: r + correction, system.r_cn),
  )
  # Barrier-error scaling uses all equality and bound multipliers. The elastic
  # duals are typically ρ, so scaling matters for the default ρ=1000.
  zs = (b.z_l, b.z_u, b.zf_l, b.zf_u, *system.zp, *system.zn)
  zsum = sum(jp.sum(jp.abs(z)) for z in zs)
  zcount = sum(
      jp.sum(jp.isfinite(v)) for v in (b.g_l, b.g_u, b.gf_l, b.gf_u)
  ) + sum(v.size for v in (*system.zp, *system.zn))
  eqs = (it.base.lmbda, it.base.beta, it.base.y, it.base.yf)
  eqsum = sum(jp.sum(jp.abs(v)) for v in eqs)
  eqcount = sum(v.size for v in eqs)
  sd = jp.maximum(
      1.0, (zsum + eqsum) / jp.maximum(1, zcount + eqcount) / options.s_max
  )
  sc = jp.maximum(1.0, zsum / jp.maximum(1, zcount) / options.s_max)
  return jp.maximum(constraints, jp.maximum(dual / sd, comp / sc))


def acceptable_return(
    point: ip_iterate.Iterate,
    evaluation: dict[str, jax.Array],
    entry_evaluation: tuple[jax.Array, jax.Array],
    entry_norm: jax.Array,
    mu: jax.Array,
    bounds: pd.Bounds,
    line_search: ls.LineSearchState,
    quaternion_addrs: tuple[int, ...],
    options: Options,
) -> jax.Array:
  """Return whether a point can leave restoration for the original problem."""
  phi, theta = ls.merit(point, evaluation, mu, bounds, options)
  # The entry iterate's barrier is supplied alongside its residuals.
  phi0, theta0 = entry_evaluation
  norm = ip_iterate.max_abs(
      *(evaluation[name] for name in ls._EQUALITY_CONSTRAINT_FIELDS)
  )
  limit = jp.maximum(
      options.required_infeasibility_reduction * entry_norm,
      min(options.tol, options.constr_viol_tol),
  )
  limit = jp.where(options.required_infeasibility_reduction == 0, 0.0, limit)
  progress = ls._le(theta, (1 - options.gamma_theta) * theta0, theta0) | ls._le(
      phi - phi0, -options.gamma_phi * theta0, phi0
  )
  return (
      ip_iterate.valid(point, bounds, quaternion_addrs)
      & jp.isfinite(phi)
      & jp.isfinite(theta)
      & (theta <= line_search.theta_max)
      & (norm <= limit)
      & progress
      & ip_filter.filter_acceptable(line_search.filter, theta, phi)
  )
