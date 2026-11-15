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
) -> jax.Array:
  """Reduce mu while its current barrier subproblem is sufficiently solved."""

  def reduce_cond(current: jax.Array) -> jax.Array:
    return (
        (error_at_mu(current) < options.barrier_tol_factor * current)
        & (_barrier_candidate(current, options) < current)
    )

  return jax.lax.while_loop(
      reduce_cond, lambda current: _barrier_candidate(current, options), mu
  )
