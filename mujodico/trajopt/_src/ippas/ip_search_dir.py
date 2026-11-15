from dataclasses import replace

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas.ip_iterate import all_finite, max_abs
from mujodico.trajopt._src.ippas.options import Options


def compute(
    system: pd.IpPrimalDualSystem,
    mu: jax.Array,
    last_delta: jax.Array,
    options: Options,
) -> tuple[pd.IpPrimalDualSystem, jax.Array, jax.Array]:
  """Compute a direction with primal correction and dual regularization."""

  o = options
  delta_c = dual_regularization(system.Eu, mu, o)
  dual_valid = control_equalities_full_rank(system.Eu, o) | (
      jp.isfinite(delta_c) & (delta_c > 0)
  )
  H = jp.concatenate(
      (
          jp.concatenate((system.Hxx[:-1], system.Hxu), axis=-1),
          jp.concatenate((system.Hxu.swapaxes(-1, -2), system.Huu), axis=-1),
      ),
      axis=-2,
  )
  mineig = jp.minimum(
      jp.min(jp.linalg.eigvalsh(H)),
      jp.min(jp.linalg.eigvalsh(system.Hxx[-1])),
  )
  tolerance = (
      100 * jp.finfo(H.dtype).eps * jp.maximum(1, max_abs(H, system.Hxx[-1]))
  )

  def attempt(delta: jax.Array) -> tuple[pd.IpPrimalDualSystem, jax.Array]:
    shifted = replace(
        system,
        Hxx=system.Hxx + delta * jp.eye(system.Hxx.shape[-1]),
        Huu=system.Huu + delta * jp.eye(system.Huu.shape[-1]),
        delta_c=delta_c,
        _cache=None,
    )
    solved, accurate = pd.iterative_refinement_solve(
        shifted,
        linear_tol=o.linear_tol,
        max_refinement=o.max_refinement,
    )
    finite = all_finite((
        solved.dx,
        solved.du,
        solved.dlmbda,
        solved.dbeta,
        solved.ds,
        solved.dsf,
        solved.dz_l,
        solved.dz_u,
        solved.dzf_l,
        solved.dzf_u,
        solved.dy,
        solved.dyf,
    ))
    curvature_valid = mineig + delta >= -tolerance
    if o.print_level > 1:
      jax.debug.print(
          'Newton residual={r:.2e} reg={d:.2e} dual_reg={dc:.2e}',
          r=max_abs(*pd.system_defect(solved).values()),
          d=delta,
          dc=delta_c,
      )
    return solved, dual_valid & finite & accurate & curvature_valid

  margin = jp.maximum(o.delta_w0, tolerance)
  delta_w = jp.minimum(
      o.delta_w_max,
      jp.where(mineig < -tolerance, -mineig + margin, 0.0),
  )
  solved, good = attempt(delta_w)
  initial = (solved, good, delta_w, 1)

  def cond(carry) -> jax.Array:
    _, good, delta_w, count = carry
    return (
        dual_valid
        & ~good
        & (count < o.max_regularization)
        & (delta_w < o.delta_w_max)
    )

  def body(carry):
    _, _, delta_w, count = carry
    first = jp.where(
        last_delta > 0,
        jp.maximum(o.delta_wmin, o.kappa_wmin * last_delta),
        o.delta_w0,
    )
    growth = jp.where(last_delta > 0, o.kappa_wplus, o.kappa_wplusem)
    delta_w = jp.minimum(
        o.delta_w_max, jp.where(delta_w == 0, first, delta_w * growth)
    )
    solved, good = attempt(delta_w)
    return solved, good, delta_w, count + 1

  solved, good, delta_w, _ = jax.lax.while_loop(cond, body, initial)
  return solved, good, delta_w


def control_equalities_full_rank(Eu: jax.Array, options: Options) -> jax.Array:
  """Whether every control-equality Jacobian has numerically independent rows."""
  if not Eu.shape[1]:
    return jp.asarray(True)
  if Eu.shape[1] > Eu.shape[2]:
    return jp.asarray(False)
  sv = jp.linalg.svd(Eu, compute_uv=False)
  return jp.all(sv[:, -1] > options.rank_tol * jp.maximum(1, sv[:, 0]))


def dual_regularization(
    Eu: jax.Array, mu: jax.Array, options: Options
) -> jax.Array:
  """Fixed μ-dependent shift, applied only to stage equalities."""
  return jp.where(
      control_equalities_full_rank(Eu, options),
      jp.asarray(0.0, Eu.dtype),
      options.delta_c_stripe * mu**options.kappa_c,
  )
