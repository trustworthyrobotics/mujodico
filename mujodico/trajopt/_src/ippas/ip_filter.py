"""Fixed-capacity filter storage and strict-dominance tests.

Let θ = ‖c(x, u, s)‖₁ measure constraint infeasibility, including g(x, u)-s,
and let φ be the objective plus the barrier. The filter stores corners (θᵢ, φᵢ).
A trial pair (θ, φ) is acceptable to the filter iff

  θ ≤ θᵢ  or  φ ≤ φᵢ,  for every stored corner i.

Equivalently, reject a trial if θ > θᵢ and φ > φᵢ for any corner. Using the
strict-dominance convention, equality is allowed. An empty filter accepts
every pair; inactive buffer entries are ignored.

augment_filter() adds a reference pair (θ, φ) with acceptance margins:

  ((1 - γ_θ) θ, φ - γ_φ θ)

to the filter stored corners.
"""

from typing import NamedTuple

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas.options import Options


class Filter(NamedTuple):
  theta: jax.Array
  phi: jax.Array
  count: jax.Array


def empty_filter(capacity: int, dtype) -> Filter:
  return Filter(
      jp.zeros(capacity, dtype),
      jp.zeros(capacity, dtype),
      jp.asarray(0, jp.int32),
  )


def augment_filter(
    filter: Filter, theta: jax.Array, phi: jax.Array, options: Options
) -> Filter:
  """Insert a reference iterate with the line search's acceptance margins."""
  return filter._replace(
      theta=filter.theta.at[filter.count].set(
          (1 - options.gamma_theta) * theta
      ),
      phi=filter.phi.at[filter.count].set(phi - options.gamma_phi * theta),
      count=filter.count + 1,
  )


def filter_acceptable(
    state: Filter, theta: jax.Array, phi: jax.Array
) -> jax.Array:
  """Return a scalar boolean array for acceptance against all active corners."""
  active = jp.arange(state.theta.size) < state.count
  return jp.all(~active | (theta <= state.theta) | (phi <= state.phi))


def reset(state: Filter) -> Filter:
  """Clear active corners while preserving capacity."""
  return state._replace(count=jp.asarray(0, jp.int32))
