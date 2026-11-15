from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas.options import Options


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Iterate:
  x: jax.Array
  u: jax.Array
  s: jax.Array
  sf: jax.Array
  lmbda: jax.Array
  beta: jax.Array
  y: jax.Array
  yf: jax.Array
  z_l: jax.Array
  z_u: jax.Array
  zf_l: jax.Array
  zf_u: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IterateData:
  objective: jax.Array
  objective_gradx: jax.Array
  objective_gradu: jax.Array
  stationarity_x: jax.Array
  stationarity_u: jax.Array
  r0: jax.Array
  r_f: jax.Array
  r_e: jax.Array
  r_g: jax.Array
  r_gf: jax.Array


class Metrics(NamedTuple):
  primal: jax.Array
  dual: jax.Array
  complementarity: jax.Array
  error: jax.Array
  barrier_error: jax.Array


def metrics(
    it: Iterate,
    data: IterateData,
    mu: jax.Array,
    bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    original_bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    options: Options,
) -> Metrics:
  """Return raw residual norms and scaled NLP/barrier KKT errors.

  The scaled optimality error is

    Eμ = max(‖r_d(μ)‖∞ / s_d, ‖c‖∞, ‖r_comp(μ)‖∞ / s_c).   [IPOPT, eq. (5)]

  NLP error is E₀, used in the overall termination criterion (6);
  barrier_error is Eμ, used to decide whether to decrease μ.

  c stacks initial, dynamics, stage-equality, and slack-link residuals.
  r_d includes state/control and slack stationarity; for Eμ, slack
  stationarity includes one-sided damping (§3.7). r_comp(μ) contains bound
  complementarity products minus μ, with unbounded sides omitted.

  primal is max(‖c‖∞, original-bound violation of g = r_g + s); dual and
  complementarity are unscaled ∞-norms at μ = 0. Both E₀ and Eμ use the
  supplied effective bounds, which may be relaxed.
  """
  o = options
  g_l, g_u, gf_l, gf_u = bounds

  c_l, c_u = complementarity(it.s, it.z_l, it.z_u, g_l, g_u, 0.0)
  cf_l, cf_u = complementarity(it.sf, it.zf_l, it.zf_u, gf_l, gf_u, 0.0)

  cm_l, cm_u = complementarity(it.s, it.z_l, it.z_u, g_l, g_u, mu)
  cmf_l, cmf_u = complementarity(it.sf, it.zf_l, it.zf_u, gf_l, gf_u, mu)

  g = data.r_g + it.s
  gf = data.r_gf + it.sf

  rs = -it.y - it.z_l + it.z_u
  rsf = -it.yf - it.zf_l + it.zf_u

  violation = max_abs(
      jp.maximum(original_bounds[0] - g, 0),
      jp.maximum(g - original_bounds[1], 0),
      jp.maximum(original_bounds[2] - gf, 0),
      jp.maximum(gf - original_bounds[3], 0),
  )
  constraint_error = max_abs(data.r0, data.r_f, data.r_e, data.r_g, data.r_gf)
  primal = jp.maximum(violation, constraint_error)

  dual = max_abs(data.stationarity_x, data.stationarity_u, rs, rsf)

  comp = max_abs(c_l, c_u, cf_l, cf_u)

  zsum = jp.sum(it.z_l) + jp.sum(it.z_u) + jp.sum(it.zf_l) + jp.sum(it.zf_u)
  zcount = sum(jp.sum(jp.isfinite(v)) for v in bounds)
  eqsum = sum(jp.sum(jp.abs(v)) for v in (it.lmbda, it.beta, it.y, it.yf))
  eqcount = it.lmbda.size + it.beta.size + it.y.size + it.yf.size

  sd = jp.maximum(1.0, (zsum + eqsum) / jp.maximum(1, zcount + eqcount) / o.s_max)
  sc = jp.maximum(1.0, zsum / jp.maximum(1, zcount) / o.s_max)

  damp = damping(g_l, g_u, mu, options)
  dampf = damping(gf_l, gf_u, mu, options)

  error = jp.maximum(constraint_error, jp.maximum(dual / sd, comp / sc))

  barrier_error = jp.maximum(
      constraint_error,
      jp.maximum(
          max_abs(data.stationarity_x, data.stationarity_u, rs + damp, rsf + dampf) / sd,
          max_abs(cm_l, cm_u, cmf_l, cmf_u) / sc,
      ),
  )

  return Metrics(
      primal=primal,
      dual=dual,
      complementarity=comp,
      error=error,
      barrier_error=barrier_error,
  )


def valid(
    it: Iterate,
    bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    quaternion_positions: tuple[int, ...],
) -> jax.Array:
  """Return a scalar JAX boolean checking the iterate's numerical domain.

  Require every iterate array to be finite. For stage and terminal slacks,
  require s > lower and s < upper on finite-bound sides. Bound multipliers
  must be strictly positive on finite sides and exactly zero on unbounded
  sides. quaternion_positions gives the start index of each four-component
  quaternion block q in it.x. At every state, require |‖q‖₂² − 1| ≤ 10⁻⁶;
  an empty tuple skips quaternion checks.
  """
  valid = all_finite(it)
  for s, z_l, z_u, l, u in (
      (it.s, it.z_l, it.z_u, *bounds[:2]),
      (it.sf, it.zf_l, it.zf_u, *bounds[2:]),
  ):
    dist_l, dist_u = distances(s, l, u)
    valid &= jp.all((dist_l > 0) & (dist_u > 0))
    valid &= jp.all(jp.where(jp.isfinite(l), z_l > 0, z_l == 0))
    valid &= jp.all(jp.where(jp.isfinite(u), z_u > 0, z_u == 0))

  for i in quaternion_positions:
    valid &= jp.all(jp.abs(jp.sum(it.x[:, i : i + 4] ** 2, axis=1) - 1) <= 1e-6)

  return valid


def max_abs(*values: jax.Array) -> jax.Array:
  """Return the largest absolute entry; empty arrays contribute zero."""
  return jp.max(jp.stack([jp.max(jp.abs(v), initial=0.0) for v in values]))


def all_finite(tree: object) -> jax.Array:
  """Return whether all entries in every numeric JAX-pytree leaf are finite."""
  return jp.all(
      jp.stack([jp.all(jp.isfinite(v)) for v in jax.tree.leaves(tree)])
  )


def distances(
    s: jax.Array, lower: jax.Array, upper: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Return s − lower and upper − s; unbounded sides use unit distance."""
  return (
      jp.where(jp.isfinite(lower), s - lower, 1.0),
      jp.where(jp.isfinite(upper), upper - s, 1.0),
  )


def _mv(a: jax.Array, v: jax.Array) -> jax.Array:
  """Multiply matrices by vectors, retaining their shared leading batch axes."""
  return jp.einsum('...ij,...j->...i', a, v)


def complementarity(
    s: jax.Array,
    z_l: jax.Array,
    z_u: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    target: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Return (s − lower) zˡ − target and (upper − s) zᵘ − target.

  Unbounded sides contribute zero to their complementarity residual.
  """
  dist_l, dist_u = distances(s, lower, upper)
  return (
      jp.where(jp.isfinite(lower), dist_l * z_l - target, 0.0),
      jp.where(jp.isfinite(upper), dist_u * z_u - target, 0.0),
  )


def damping(
    lower: jax.Array, upper: jax.Array, mu: jax.Array, options: Options
) -> jax.Array:
  """Return κ_d μ for lower-only bounds, −κ_d μ for upper-only, else zero
  [IPOPT, sec 3.7]."""
  return (
      options.kappa_d
      * mu
      * (
          jp.isfinite(lower).astype(lower.dtype)
          - jp.isfinite(upper).astype(upper.dtype)
      )
  )
