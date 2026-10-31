"""Interior pushing and scan-based least-squares multiplier initialization.

Policy based on FATROP (BSD-2-Clause); see NOTICE.md.
"""

import jax
import jax.numpy as jnp
from . import iterate as bounds
from . import search_direction as regularization
from . import linear_solver as ls
from .problem import Primal
from .iterate import Iterate, Metrics, metrics


def initialize(problem, params, initial_guess, options, preserve_primal=False):
  if initial_guess is None:
    primal = problem.initialize(params)
  elif isinstance(initial_guess, Primal):
    if preserve_primal:
      primal = initial_guess
    else:
      primal = problem.initialize(
          params, initial_guess.xs, initial_guess.us, initial_guess.ut
      )
      primal = primal._replace(s=initial_guess.s, st=initial_guess.st)
  else:
    primal = problem.initialize(params, initial_guess.xs, initial_guess.us)
  original_lower, original_upper = problem.bounds(primal, params)
  lower, upper = bounds.relax(
      original_lower,
      original_upper,
      options.bound_relax_factor,
      options.constr_viol_tol,
  )
  if not preserve_primal:
    primal = bounds.push(
        primal, lower, upper, options.bound_push, options.bound_frac
    )
  e = problem.evaluate(primal, params)
  eq, eqT = jnp.zeros_like(e.eq), jnp.zeros_like(e.eqT)
  bd = bounds.data(primal, lower, upper)
  zl, zu = bounds.initialize_duals(bd)
  iterate = Iterate(primal, eq, eqT, zl, zu)
  if options.initialize_dual:
    system, rhs = problem.linearize(primal, eq, eqT, params)
    from .search_direction import add_bounds

    zero = bounds.tree_map(jnp.zeros_like, primal)
    force = bounds.tree_map(lambda l, u: -l + u, zl, zu)
    system, rhs = add_bounds(problem, primal, system, rhs, zero, force)
    system = regularization.identity_system(
        system, problem.nxt, problem.nut, primal.s.shape[-1]
    )
    dc = jnp.maximum(options.delta_c, regularization.dual_floor(system))
    system = regularization.shifted(
        system, 0.0, dc, problem.nxt, problem.nut, primal.s.shape[-1]
    )
    rhs = rhs._replace(c=jnp.zeros_like(rhs.c), cT=jnp.zeros_like(rhs.cT))
    backend = (
        'parallel'
        if options.linear_backend == 'pas'
        else options.linear_backend
    )
    factors = ls.factorize(system, backend)
    d = ls.solve_rhs(factors, rhs, backend)
    good = (
        (factors.status == 0)
        & jnp.isfinite(bounds.tree_max(d))
        & (
            jnp.maximum(
                jnp.max(jnp.abs(d.beta), initial=0.0),
                jnp.max(jnp.abs(d.betaT), initial=0.0),
            )
            <= options.lam_max
        )
    )
    iterate = iterate._replace(
        eq=jnp.where(good, d.beta, eq), eqT=jnp.where(good, d.betaT, eqT)
    )
  return iterate, lower, upper, original_lower, original_upper
