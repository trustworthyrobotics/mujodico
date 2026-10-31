# Portions follow FATROP's Ipopt-derived interior-point policies.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
# See licenses/LICENSE-EPL-2.0.txt and NOTICE.md.
"""Monotone barrier-parameter and fraction-to-boundary updates."""

import jax.numpy as jnp


def update_mu(mu, error, tol, options):
  """Perform one monotone reduction when the current barrier NLP is solved.

  ``error`` is the primal-dual residual measured at the current ``mu``. A caller
  wanting fast repeated reductions must recompute that residual after each
  reduction rather than repeatedly using an obsolete complementarity target.
  """
  floor = jnp.maximum(
      options.mu_min,
      jnp.minimum(tol, options.compl_inf_tol)
      / (options.barrier_tol_factor + 1.0),
  )
  candidate = jnp.maximum(
      floor,
      jnp.minimum(
          options.mu_linear_decrease_factor * mu,
          jnp.power(mu, options.mu_superlinear_decrease_power),
      ),
  )
  candidate = jnp.minimum(mu, candidate)
  should_reduce = jnp.isfinite(error) & (
      error < options.barrier_tol_factor * mu
  )
  updated = jnp.where(should_reduce, candidate, mu)
  return updated, updated != mu


def fraction_parameter(mu, options):
  return jnp.maximum(options.tau_min, 1.0 - mu)
