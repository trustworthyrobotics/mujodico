# Portions follow FATROP's Ipopt-derived interior-point policies.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
"""Fully compiled primal-dual OCP solver and device-side globalization."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from . import line_search
from . import barrier
from . import filter as filters
from . import iterate as bounds
from .convergence import Status, check_convergence
from .options import Options
from .problem import Problem, Primal
from .initialization import Iterate, initialize, metrics
from . import search_direction


class InitialGuess(NamedTuple):
  xs: object = None
  us: object = None


class Trace(NamedTuple):
  objective: object
  error: object
  primal: object
  dual: object
  complementarity: object
  mu: object
  alpha_primal: object
  alpha_dual: object
  step_type: object
  backtracks: object
  soc_steps: object
  delta_w: object
  delta_c: object
  linear_residual: object
  restoration_iterations: object
  linear_refinements: object
  factorizations: object
  valid: object


class SolveResult(NamedTuple):
  xs: object
  us: object
  slacks: object
  terminal_slacks: object
  eq: object
  eqT: object
  zl: object
  zu: object
  status: object
  iterations: object
  objective: object
  primal_error: object
  dual_error: object
  complementarity: object
  error: object
  trace: Trace
  primal: Primal
  initial_multiplier: object

  @property
  def accepted_steps(self):
    return jnp.sum(
        self.trace.valid
        & (self.trace.step_type > 0)
        & (self.trace.step_type != 6)
    )


class Snapshot(NamedTuple):
  iterate: object
  filter: object
  mu: object
  phi: object
  theta: object
  dphi: object
  alpha: object


class State(NamedTuple):
  iterate: Iterate
  mu: object
  iteration: object
  status: object
  acceptable_count: object
  filter: object
  watchdog: object
  trace: Trace
  metrics: object
  last_delta: object
  soft_count: object
  filter_rejections: object
  filter_resets: object
  tiny: object


def _trace(size, dtype):
  z = lambda: jnp.zeros(size, dtype)
  i = lambda: jnp.zeros(size, jnp.int32)
  return Trace(
      z(),
      z(),
      z(),
      z(),
      z(),
      z(),
      z(),
      z(),
      i(),
      i(),
      i(),
      z(),
      z(),
      z(),
      i(),
      i(),
      i(),
      jnp.zeros(size, bool),
  )


def _record(
    trace,
    index,
    m,
    mu,
    alpha=0.0,
    ad=0.0,
    kind=0,
    backtracks=0,
    soc_steps=0,
    dw=0.0,
    dc=0.0,
    linear_residual=0.0,
    resto=0,
    refinements=0,
    factorizations=0,
):
  values = (
      m.objective,
      m.error,
      m.primal,
      m.dual,
      m.complementarity,
      mu,
      alpha,
      ad,
      kind,
      backtracks,
      soc_steps,
      dw,
      dc,
      linear_residual,
      resto,
      refinements,
      factorizations,
      True,
  )
  return Trace(*(x.at[index].set(v) for x, v in zip(trace, values)))


def _run(
    problem,
    params,
    initial_guess,
    options,
    allow_restore=True,
    finish_test=None,
    initial_mu=None,
):
  current, lower, upper, original_lower, original_upper = initialize(
      problem, params, initial_guess, options
  )
  dtype = current.primal.xs.dtype
  mu = jnp.asarray(options.mu_init if initial_mu is None else initial_mu, dtype)
  m = metrics(
      problem,
      current,
      params,
      mu,
      lower,
      upper,
      original_lower,
      original_upper,
      options,
  )
  status, count = check_convergence(m.error, m.primal, 0, 0, options)
  x0 = problem.initial_state(params)
  xl, xu = problem.original_x_bounds
  initial_violation = jnp.max(
      jnp.maximum(jnp.maximum(jnp.asarray(xl) - x0, x0 - jnp.asarray(xu)), 0.0),
      initial=0.0,
  )
  status = jnp.where(
      initial_violation > options.constr_viol_tol, Status.INFEASIBLE, status
  ).astype(jnp.int32)
  fs = filters.empty_filter(options.max_iter + 1, dtype)
  zero = jnp.asarray(0.0, dtype)
  cp = Snapshot(current, fs, mu, zero, zero, zero, jnp.asarray(1.0, dtype))
  state = State(
      current,
      mu,
      jnp.asarray(0, jnp.int32),
      status,
      count,
      fs,
      line_search.watchdog_create(cp),
      _record(_trace(options.max_iter + 1, dtype), 0, m, mu),
      m,
      zero,
      jnp.asarray(0, jnp.int32),
      jnp.asarray(0, jnp.int32),
      jnp.asarray(0, jnp.int32),
      jnp.asarray(False),
  )

  def cond(st):
    return (st.status == Status.RUNNING) & (st.iteration < options.max_iter)

  def body(st):
    old = st.iterate
    new_mu, changed = barrier.update_mu(
        st.mu,
        jnp.where(st.tiny, 0.0, st.metrics.barrier_error),
        options.tol,
        options,
    )
    search = search_direction.compute(
        problem,
        old.primal,
        old.eq,
        old.eqT,
        old.zl,
        old.zu,
        params,
        new_mu,
        st.last_delta,
        lower,
        upper,
        options,
    )
    globalization = line_search.find_acceptable_trial_point(
        problem,
        st,
        search,
        params,
        new_mu,
        changed,
        lower,
        upper,
        original_lower,
        original_upper,
        options,
        allow_restore,
    )
    it = globalization.iterate
    new_mu = globalization.mu
    fs, wd, step = (
        globalization.filter,
        globalization.watchdog,
        globalization.step,
    )
    kind, accepted, rollback = (
        globalization.kind,
        globalization.accepted,
        globalization.rollback,
    )
    restored_iters, restore_status = (
        globalization.restoration_iterations,
        globalization.restoration_status,
    )
    rejected, reset_count = globalization.rejections, globalization.resets
    n = st.iteration + 1
    m = metrics(
        problem,
        it,
        params,
        new_mu,
        lower,
        upper,
        original_lower,
        original_upper,
        options,
    )
    status, count = check_convergence(
        m.error, m.primal, n, st.acceptable_count, options
    )
    failure = jnp.where(
        restored_iters > 0, Status.RESTORATION_FAILED, Status.LINE_SEARCH_FAILED
    )
    failure = jnp.where(
        (restored_iters > 0)
        & (
            (restore_status == Status.SUCCESS)
            | (restore_status == Status.ACCEPTABLE)
        ),
        Status.INFEASIBLE,
        failure,
    )
    status = jnp.where(~accepted & ~rollback, failure, status)
    status = jnp.where(
        search.status != 0, Status.NUMERICAL_ERROR, status
    ).astype(jnp.int32)
    if finish_test is not None:
      status = jnp.where(finish_test(it), Status.SUCCESS, status).astype(
          jnp.int32
      )
    tiny = line_search.detect_tiny_step(search.direction, dtype)
    return State(
        it,
        new_mu,
        n,
        status,
        count,
        fs,
        wd,
        _record(
            st.trace,
            n,
            m,
            new_mu,
            step.alpha,
            step.alpha_dual,
            kind,
            step.backtracks,
            step.soc_steps,
            search.delta_w,
            search.delta_c,
            search.residual,
            restored_iters,
            search.refinements,
            search.attempts,
        ),
        m,
        jnp.where(search.delta_w > 0, search.delta_w, st.last_delta),
        jnp.where(step.kind == 3, st.soft_count + 1, 0),
        rejected,
        reset_count,
        tiny,
    )

  state = jax.lax.while_loop(cond, body, state)
  it, m = state.iterate, state.metrics

  def initial_lag(d):
    x0 = problem.state_manifold.retract(it.primal.xs[0], d, params)
    e = problem.evaluate(
        it.primal._replace(xs=it.primal.xs.at[0].set(x0)), params
    )
    return e.objective + jnp.sum(it.eq * e.eq) + jnp.sum(it.eqT * e.eqT)

  initial_multiplier = jax.grad(initial_lag)(jnp.zeros(problem.nxt, dtype))
  return SolveResult(
      it.primal.xs,
      it.primal.us,
      it.primal.s,
      it.primal.st,
      it.eq,
      it.eqT,
      it.zl,
      it.zu,
      state.status,
      state.iteration,
      m.objective,
      m.primal,
      m.dual,
      m.complementarity,
      m.error,
      state.trace,
      it.primal,
      initial_multiplier,
  )


class IppasSolver(Problem):
  """Construct once, then call ``jax.jit(solver.solve)(params, guess)``.

  Numerical parameters remain dynamic; callback structure, dimensions and
  ``Options`` are static. The solve has no Python callbacks or host transfers.
  Reverse differentiation of the while-loop solve is not an implicit
  sensitivity API; differentiate the individual problem callbacks normally.
  """

  def __init__(self, *, options=None, **kwargs):
    super().__init__(**kwargs)
    self.options = options or Options()

  def solve(self, params, initial_guess=None):
    return _run(self, params, initial_guess, self.options)


Solver = IppasSolver
