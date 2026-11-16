from collections.abc import Callable

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas.options import Options


def _barrier_candidate(mu: jax.Array, options: Options) -> jax.Array:
  """Return a nonincreasing barrier target using linear/superlinear reduction."""
  mu_floor = min(options.tol, options.compl_inf_tol) / (options.barrier_tol_factor + 1)
  return jp.minimum(
      mu,
      jp.maximum(
          mu_floor,
          jp.minimum(
              options.mu_linear_decrease_factor * mu,
              mu**options.mu_superlinear_decrease_power,
          ),
      ),
  )


def update(
    mu: jax.Array,
    error_at_mu: Callable[[jax.Array], jax.Array],
    options: Options,
    *,
    tiny_step_flag: bool | jax.Array = False,
    first_update: bool | jax.Array = False,
) -> jax.Array:
  """Reduce μ after solving its barrier problem or detecting repeated tiny steps.

  tiny_step_flag forces at most the first reduction past the usual error test.
  Further reductions require freshly evaluated errors. Disabled fast decrease
  permits one reduction per call, except for first_update (IPOPT's first-call
  exception). A forced update that returns μ unchanged signals stagnation to
  the caller; the solver reports TINY_STEP in that case.
  """

  def reduce_cond(carry) -> jax.Array:
    current, count = carry
    forced = tiny_step_flag & (count == 0)
    may_repeat = (count == 0) | options.mu_allow_fast_monotone_decrease | first_update
    return (
        ((error_at_mu(current) < options.barrier_tol_factor * current) | forced)
        & (_barrier_candidate(current, options) < current)
        & may_repeat
    )

  def reduce_body(carry):
    current, count = carry
    return _barrier_candidate(current, options), count + 1

  updated, _ = jax.lax.while_loop(
      reduce_cond, reduce_body, (mu, jp.asarray(0, jp.int32))
  )
  return updated
