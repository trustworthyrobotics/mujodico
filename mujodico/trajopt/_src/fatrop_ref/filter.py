"""Fixed-capacity filter memory usable inside JAX control flow."""

from typing import NamedTuple
import jax.numpy as jnp


class FilterState(NamedTuple):
  theta: object
  phi: object
  count: object


def empty_filter(capacity, dtype=jnp.float64):
  if capacity < 1:
    raise ValueError('filter capacity must be positive')
  return FilterState(
      jnp.zeros(capacity, dtype),
      jnp.zeros(capacity, dtype),
      jnp.asarray(0, jnp.int32),
  )


def reset(state):
  return state._replace(count=jnp.asarray(0, jnp.int32))


def acceptable(state, theta, phi):
  """Accept unless a stored, already shifted corner strictly dominates."""
  active = jnp.arange(state.theta.size) < state.count
  return jnp.all(~active | (theta <= state.theta) | (phi <= state.phi))


def augment(state, theta, phi, options):
  """Store FATROP's margin-shifted corner for the reference iterate.

  A solve reserves at least max_iter + 1 entries, so normal operation never
  exhausts capacity. Standalone callers at capacity replace the final corner.
  """
  index = jnp.minimum(state.count, state.theta.size - 1)
  return FilterState(
      state.theta.at[index].set((1.0 - options.gamma_theta) * theta),
      state.phi.at[index].set(phi - options.gamma_phi * theta),
      jnp.minimum(state.count + 1, state.theta.size),
  )
