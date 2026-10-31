# Portions follow FATROP's Ipopt-derived interior-point policies.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
# See licenses/LICENSE-EPL-2.0.txt and NOTICE.md.
"""Array-valued termination decisions and residual reductions."""

from enum import IntEnum
import jax.numpy as jnp


class Status(IntEnum):
  RUNNING = 0
  CONTINUE = 0
  SUCCESS = 1
  ACCEPTABLE = 2
  MAX_ITER = 3
  NUMERICAL_ERROR = 4
  NUMERICAL_FAILURE = 4
  LINE_SEARCH_FAILED = 5
  RESTORATION_FAILED = 6
  INFEASIBLE = 7


def max_abs(value):
  return jnp.max(jnp.abs(value), initial=0.0)


def l1_norm(value):
  return jnp.sum(jnp.abs(value))


def scaled_error(dual, primal, complementarity, multipliers, options):
  """Infinity-norm KKT error with FATROP's large-multiplier scaling."""
  mean_multiplier = jnp.sum(jnp.abs(multipliers)) / max(multipliers.size, 1)
  scaling = jnp.maximum(1.0, mean_multiplier / options.smax)
  return jnp.maximum(
      max_abs(primal),
      jnp.maximum(max_abs(dual) / scaling, max_abs(complementarity) / scaling),
  )


def check_convergence(
    error, primal_error, iteration, acceptable_count, options
):
  finite = jnp.isfinite(error) & jnp.isfinite(primal_error)
  feasible = primal_error <= options.constr_viol_tol
  successful = finite & feasible & (error <= options.tol)
  acceptable = finite & feasible & (error <= options.acceptable_tol)
  count = jnp.where(acceptable, acceptable_count + 1, 0)
  status = jnp.where(
      iteration >= options.max_iter, Status.MAX_ITER, Status.CONTINUE
  )
  status = jnp.where(
      (options.acceptable_iter > 0) & (count >= options.acceptable_iter),
      Status.ACCEPTABLE,
      status,
  )
  status = jnp.where(successful, Status.SUCCESS, status)
  status = jnp.where(finite, status, Status.NUMERICAL_FAILURE)
  return jnp.asarray(status, jnp.int32), jnp.asarray(count, jnp.int32)
