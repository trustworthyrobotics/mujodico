from dataclasses import replace
from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas.ip_iterate import all_finite, max_abs
from mujodico.trajopt._src.ippas.options import Options


System = TypeVar("System")


def compute(
    system: pd.IpPrimalDualSystem,
    mu: jax.Array,
    last_delta: jax.Array,
    options: Options,
) -> tuple[pd.IpPrimalDualSystem, jax.Array, jax.Array]:
  """Compute a direction with primal correction and dual regularization."""

  o = options
  delta_c = dual_regularization(system.Eu, mu, o)

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
        linear_ref_tol=o.linear_ref_tol,
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
    if o.print_level > 1:
      jax.debug.print(
          "Newton residual={r:.2e} reg={d:.2e} dual_reg={dc:.2e}",
          r=max_abs(*pd.system_defect(solved).values()),
          d=delta,
          dc=delta_c,
      )
    return solved, finite & accurate

  return regularized_solve((system.Hxx, system.Hxu, system.Huu), last_delta, o, attempt)


def regularized_solve(
    hessian: tuple[jax.Array, jax.Array, jax.Array],
    last_delta: jax.Array,
    o: Options,
    attempt: Callable[[jax.Array], tuple[System, jax.Array]],
) -> tuple[System, jax.Array, jax.Array]:
  """Retry an injected structured solve using the common δw schedule."""
  Hxx, Hxu, Huu = hessian
  H = jp.concatenate(
      (
          jp.concatenate((Hxx[:-1], Hxu), axis=-1),
          jp.concatenate((Hxu.swapaxes(-1, -2), Huu), axis=-1),
      ),
      axis=-2,
  )
  mineig = jp.minimum(
      jp.min(jp.linalg.eigvalsh(H)),
      jp.min(jp.linalg.eigvalsh(Hxx[-1])),
  )
  tolerance = 100 * jp.finfo(H.dtype).eps * jp.maximum(1, max_abs(H, Hxx[-1]))

  def try_shift(delta: jax.Array) -> tuple[System, jax.Array]:
    solved, good = attempt(delta)
    return solved, good & (mineig + delta >= -tolerance)

  margin = jp.maximum(o.delta_w0, tolerance)
  delta_w = jp.minimum(
      o.delta_w_max,
      jp.where(mineig < -tolerance, -mineig + margin, 0.0),
  )
  solved, good = try_shift(delta_w)
  initial = (solved, good, delta_w)

  def next_delta(current: jax.Array) -> jax.Array:
    first = jp.where(
        last_delta > 0,
        jp.maximum(o.delta_w_min, o.kappa_wmin * last_delta),
        o.delta_w0,
    )
    growth = jp.where(last_delta > 0, o.kappa_wplus, o.kappa_wplusem)
    return jp.minimum(
        o.delta_w_max, jp.where(current == 0, first, current * growth)
    )

  def cond(carry) -> jax.Array:
    _, good, delta_w = carry
    following = next_delta(delta_w)
    return (~good
        & (delta_w < o.delta_w_max)
        & jp.isfinite(following)
        & (following > delta_w)
    )

  def body(carry):
    _, _, delta_w = carry
    delta_w = next_delta(delta_w)
    solved, good = try_shift(delta_w)
    return solved, good, delta_w

  solved, good, delta_w = jax.lax.while_loop(cond, body, initial)
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
