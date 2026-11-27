from dataclasses import dataclass
from dataclasses import replace
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import ip_initializer
from mujodico.trajopt._src.ippas import ip_filter
from mujodico.trajopt._src.ippas import ip_resto_phase_min_cl1 as resto
from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import ip_linesearch
from mujodico.trajopt._src.ippas import ip_mu_update
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas import ip_search_dir
from mujodico.trajopt._src.ippas.ip_iterate import Iterate
from mujodico.trajopt._src.ippas.ip_iterate import IterateData
from mujodico.trajopt._src.ippas.ip_iterate import Metrics
from mujodico.trajopt._src.ippas.ip_iterate import all_finite, max_abs
from mujodico.trajopt._src.ippas.options import Options
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel
from mujodico.trajopt._src.types import Params


class Status(IntEnum):
  RUNNING = 0
  SUCCESS = 1
  ACCEPTABLE = 2
  MAX_ITER = 3
  NUMERICAL_ERROR = 4
  LINE_SEARCH_FAILED = 5
  INVALID_INITIAL_GUESS = 6
  TINY_STEP = 7
  RESTORATION_FAILED = 8
  LOCAL_INFEASIBILITY = 9


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class InitialGuess:
  xs: jax.Array
  us: jax.Array
  # A warm start is explicit; ordinary solves always initialize fresh duals.
  primal_dual: Iterate | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolveResult:
  iterate: Iterate
  status: jax.Array
  iterations: jax.Array
  objective: jax.Array
  primal_error: jax.Array
  dual_error: jax.Array
  complementarity: jax.Array
  error: jax.Array
  mu: jax.Array
  intermediates: Iterate | None = None

  @property
  def xs(self):
    return self.iterate.x

  @property
  def us(self):
    return self.iterate.u

  @property
  def success(self):
    return (self.status == Status.SUCCESS) | (self.status == Status.ACCEPTABLE)


class State(NamedTuple):
  iterate: Iterate
  system: pd.IpPrimalDualSystem
  data: IterateData
  metrics: Metrics
  iteration: jax.Array
  status: jax.Array
  acceptable_count: jax.Array
  mu: jax.Array
  line_search: ip_linesearch.LineSearchState
  last_delta: jax.Array
  filter_rejections: jax.Array
  filter_resets: jax.Array
  tiny_step_last_iteration: jax.Array
  tiny_step_flag: jax.Array
  watchdog: ip_linesearch.WatchdogState
  soft_count: jax.Array
  skip_first: jax.Array


class _Transition(NamedTuple):
  iterate: Iterate
  line_search: ip_linesearch.LineSearchState
  watchdog: ip_linesearch.WatchdogState
  soft_count: jax.Array
  skip_first: jax.Array
  accepted: jax.Array
  filter_rejected: jax.Array
  tiny: jax.Array
  iteration: jax.Array
  status: jax.Array
  last_delta: jax.Array
  acceptable_seed: jax.Array
  phase: jax.Array


class Solver:
  """Primal-dual interior-point solver taking a StagewiseOcpModel directly.

  Recovery phases in print_level output: 0 normal, 1 watchdog, 2 soft
  restoration, 3 full restoration, 4 watchdog rollback. All phases share
  max_iter; only successful original-problem convergence sets success.


  The structured backend requires N>=1, nu>=1, and no terminal equalities.
  Enable JAX float64 before constructing models.
  """

  def __init__(self, model: StagewiseOcpModel, options: Options | None = None):
    if not isinstance(model, StagewiseOcpModel):
      raise TypeError('model must be a StagewiseOcpModel.')
    if options is not None and not isinstance(options, Options):
      raise TypeError('options must be an Options instance.')
    if model.N < 1 or model.nu < 1:
      raise ValueError('IPPAS currently requires N >= 1 and nu >= 1.')

    self.model = model
    self.options = options or Options()

  def solve(self, params: Params, initial_guess: InitialGuess) -> SolveResult:
    if not isinstance(initial_guess, InitialGuess):
      raise TypeError('initial_guess must be an InitialGuess instance.')
    if not jax.config.x64_enabled:
      raise ValueError(
          'IPPAS requires float64; enable jax_enable_x64 before constructing'
          ' the model.'
      )

    m, o = self.model, self.options
    with jax.ensure_compile_time_eval():
      m.initialize(params)

    if m.constraint_sizes[2] != 0:
      raise ValueError('IPPAS currently does not support terminal equalities.')

    bounds = ip_initializer.get_bounds(m, o)
    original_bounds = ip_initializer.get_bounds(m, o, original=True)

    it = ip_initializer.initialize(m, params, initial_guess, bounds, o)
    if initial_guess.primal_dual is None:
      projection, _ = pd.evaluate(m, it, params, jp.asarray(0.0), bounds, o)
      lmbda, beta = ip_initializer.initialize_multipliers(projection, o)
      it = replace(it, lmbda=lmbda, beta=beta)

    mu = jp.asarray(o.mu_init)
    system, data = pd.evaluate(m, it, params, mu, bounds, o)
    metrics = ip_iterate.metrics(it, data, mu, bounds, original_bounds, o)

    zero = jp.asarray(0, jp.int32)
    status = check_convergence(
        m, it, system, data, metrics, zero, zero, bounds, o
    )
    status = jp.where(
        ip_iterate.valid(it, bounds, m.quaternion_addrs),
        status,
        Status.INVALID_INITIAL_GUESS,
    ).astype(jp.int32)

    def theta(data: IterateData) -> jax.Array:
      return sum(
          jp.sum(jp.abs(v))
          for v in (data.r0, data.r_f, data.r_e, data.r_g, data.r_gf)
      )

    state = State(
        iterate=it,
        system=system,
        data=data,
        metrics=metrics,
        iteration=zero,
        status=status,
        acceptable_count=zero,
        mu=mu,
        line_search=ip_linesearch.LineSearchState.create(
            o.max_iter + 1, theta(data), o
        ),
        last_delta=jp.asarray(0.0),
        filter_rejections=zero,
        filter_resets=zero,
        tiny_step_last_iteration=jp.asarray(False),
        tiny_step_flag=jp.asarray(False),
        watchdog=ip_linesearch.WatchdogState.create(it),
        soft_count=zero,
        skip_first=jp.asarray(False),
    )

    def step(state: State, new_mu: jax.Array) -> State:
      changed = new_mu != state.mu
      line_search = jax.lax.cond(
          changed,
          lambda _: ip_linesearch.LineSearchState.create(
              o.max_iter + 1, theta(state.data), o
          ),
          lambda _: state.line_search,
          None,
      )
      watchdog = state.watchdog.for_barrier_change(changed)
      system = pd.refresh_barrier_terms(state.system, state.iterate, new_mu, bounds, o)
      direction, good, delta = ip_search_dir.compute(
          system, new_mu, state.last_delta, o
      )
      phi0, theta0 = ip_linesearch.merit(
          state.iterate, vars(state.data), new_mu, bounds, o
      )
      tiny_direction = ip_linesearch.detect_tiny_step(
          state.iterate, direction, theta0, o
      )
      start = (
          (o.watchdog_shortened_iter_trigger > 0)
          & (watchdog.shortened >= o.watchdog_shortened_iter_trigger)
          & ~watchdog.active
          & (state.soft_count == 0)
          & ~tiny_direction
      )
      watchdog = jax.lax.cond(
          start,
          lambda _: watchdog._replace(
              active=jp.asarray(True),
              trials=zero,
              iterate=state.iterate,
              phi=phi0,
              theta=theta0,
              dphi=ip_linesearch.directional_derivative(
                  state.iterate, state.data, direction, new_mu, bounds, o
              ),
              alpha=ip_linesearch.maximum_step_lengths(
                  state.iterate, direction, new_mu, bounds, o
              )[0],
              last_delta=state.last_delta,
              acceptable_count=state.acceptable_count,
          ),
          lambda _: watchdog,
          None,
      )
      inactive = watchdog._replace(
          active=jp.asarray(False), shortened=zero, trials=zero
      )
      initial = _Transition(
          state.iterate,
          line_search,
          watchdog,
          state.soft_count,
          jp.asarray(False),
          jp.asarray(False),
          jp.asarray(False),
          jp.asarray(False),
          state.iteration + 1,
          jp.asarray(Status.RUNNING, jp.int32),
          jp.where(delta > 0, delta, state.last_delta),
          state.acceptable_count,
          zero,
      )

      def restore(t):
        if not o.restoration_enabled:
          return t._replace(
              status=jp.asarray(Status.LINE_SEARCH_FAILED, jp.int32)
          )
        result = resto.perform_restoration(
            m,
            state.iterate,
            state.data,
            params,
            new_mu,
            bounds,
            t.line_search,
            t.iteration,
            o,
        )
        status = jp.select(
            [
                result.status == resto.Status.RETURNED,
                result.status == resto.Status.LOCAL_INFEASIBILITY,
                result.status == resto.Status.MAX_ITER,
            ],
            [Status.RUNNING, Status.LOCAL_INFEASIBILITY, Status.MAX_ITER],
            default=Status.RESTORATION_FAILED,
        ).astype(jp.int32)
        return t._replace(
            iterate=result.iterate,
            iteration=result.iteration,
            status=status,
            accepted=result.status == resto.Status.RETURNED,
            watchdog=inactive,
            soft_count=zero,
            acceptable_seed=zero,
            phase=jp.asarray(3, jp.int32),
        )

      def soft(t, entering):
        # On entry augment once; a continuation keeps all existing corners.
        fs = jax.lax.cond(
            entering,
            lambda f: f._replace(
                filter=ip_filter.augment_filter(f.filter, theta0, phi0, o)
            ),
            lambda f: f,
            t.line_search,
        )
        trial, ordinary = ip_linesearch.try_soft_resto_step(
            m,
            state.iterate,
            state.data,
            direction,
            new_mu,
            params,
            bounds,
            fs,
            o,
        )
        allowed = entering | (state.soft_count <= o.max_soft_resto_iters)
        accepted = trial.accepted & allowed
        updated = t._replace(
            line_search=fs, watchdog=inactive, phase=jp.asarray(2, jp.int32)
        )
        return jax.lax.cond(
            accepted,
            lambda _: updated._replace(
                iterate=trial.iterate,
                accepted=jp.asarray(True),
                soft_count=jp.where(
                    ordinary, 0, jp.where(entering, 1, state.soft_count + 1)
                ).astype(jp.int32),
            ),
            lambda _: restore(updated._replace(soft_count=zero)),
            None,
        )

      def normal(t):
        result = ip_linesearch.find_acceptable_trial_point(
            m,
            state.iterate,
            state.data,
            direction,
            new_mu,
            params,
            bounds,
            t.line_search,
            o,
            skip_first=state.skip_first & ~changed,
        )
        updated = t._replace(
            iterate=result.search.iterate,
            line_search=result.state,
            accepted=result.search.accepted,
            filter_rejected=result.filter_rejected,
            tiny=result.tiny_step,
            soft_count=zero,
            watchdog=inactive._replace(
                shortened=jp.where(
                    result.backtracked, watchdog.shortened + 1, 0
                )
            ),
        )
        # A rejected trial never becomes the entry iterate of a recovery phase.
        return jax.lax.cond(
            result.search.accepted,
            lambda _: updated,
            lambda _: soft(
                t._replace(filter_rejected=result.filter_rejected),
                jp.asarray(True),
            ),
            None,
        )

      def watchdog_search(t):
        trial, eligible, fs = ip_linesearch.watchdog_trial(
            m,
            state.iterate,
            direction,
            new_mu,
            params,
            bounds,
            t.line_search,
            watchdog,
            o,
        )
        success = trial.accepted & ~tiny_direction
        provisional = (
            eligible
            & ~tiny_direction
            & (watchdog.trials + 1 < o.watchdog_trial_iter_max)
        )

        def rollback(_):
          return t._replace(
              iterate=watchdog.iterate,
              watchdog=inactive,
              soft_count=zero,
              skip_first=jp.asarray(True),
              last_delta=watchdog.last_delta,
              acceptable_seed=watchdog.acceptable_count,
              phase=jp.asarray(4, jp.int32),
          )

        return jax.lax.cond(
            success | provisional,
            lambda _: t._replace(
                iterate=trial.iterate,
                accepted=jp.asarray(True),
                line_search=fs,
                watchdog=jax.lax.cond(
                    success,
                    lambda _: inactive,
                    lambda _: watchdog._replace(trials=watchdog.trials + 1),
                    None,
                ),
                phase=jp.asarray(1, jp.int32),
            ),
            rollback,
            None,
        )

      transition = jax.lax.cond(
          good,
          lambda _: jax.lax.cond(
              watchdog.active,
              watchdog_search,
              lambda t: jax.lax.cond(
                  (state.soft_count > 0) & ~tiny_direction,
                  lambda t: soft(t, jp.asarray(False)),
                  normal,
                  t,
              ),
              initial,
          ),
          lambda _: initial._replace(
              status=jp.asarray(Status.NUMERICAL_ERROR, jp.int32),
              iterate=jax.tree.map(
                  lambda a, b: jp.where(watchdog.active, a, b),
                  watchdog.iterate,
                  state.iterate,
              ),
              watchdog=inactive,
          ),
          None,
      )

      def assess(it, seed, accepted):
        system, data = pd.evaluate(m, it, params, new_mu, bounds, o)
        mm = ip_iterate.metrics(it, data, new_mu, bounds, original_bounds, o)
        count = jp.where(
            accepted, jp.where(mm.error <= o.tol_acceptable, seed + 1, 0), seed
        )
        status = check_convergence(
            m, it, system, data, mm, count, transition.iteration, bounds, o
        )
        return system, data, mm, count, status

      system, data, mm, count, status = assess(
          transition.iterate, transition.acceptable_seed, transition.accepted
      )
      # Convergence validates the point even if the watchdog filter reference
      # rejected it. Otherwise budget exhaustion returns the saved anchor.
      pending = (
          transition.watchdog.active
          & (transition.iteration >= o.max_iter)
          & (status != Status.SUCCESS)
          & (status != Status.ACCEPTABLE)
      )
      transition = transition._replace(
          iterate=jax.tree.map(
              lambda a, b: jp.where(pending, a, b),
              transition.watchdog.iterate,
              transition.iterate,
          ),
          watchdog=transition.watchdog._replace(
              active=transition.watchdog.active & ~pending
          ),
          accepted=transition.accepted & ~pending,
          acceptable_seed=jp.where(
              pending,
              transition.watchdog.acceptable_count,
              transition.acceptable_seed,
          ),
          last_delta=jp.where(
              pending, transition.watchdog.last_delta, transition.last_delta
          ),
      )
      it = transition.iterate
      system, data, mm, count, status = jax.lax.cond(
          pending,
          lambda _: assess(it, transition.acceptable_seed, jp.asarray(False)),
          lambda _: (system, data, mm, count, status),
          None,
      )
      status = jp.where(
          (transition.status != Status.RUNNING)
          & (status != Status.SUCCESS)
          & (status != Status.ACCEPTABLE),
          transition.status,
          status,
      ).astype(jp.int32)
      # Provisional watchdog trials do not modify filter-reset history.
      track = (
          transition.accepted
          & ~transition.watchdog.active
          & (transition.phase == 0)
      )
      fs, rejections, resets = jax.lax.cond(
          track,
          lambda _: ip_linesearch.update_filter_rejections(
              transition.line_search,
              jp.asarray(True),
              transition.filter_rejected,
              jp.where(changed, 0, state.filter_rejections),
              state.filter_resets,
              o,
          ),
          lambda _: (
              transition.line_search,
              jp.where(changed, 0, state.filter_rejections),
              state.filter_resets,
          ),
          None,
      )
      tiny = transition.accepted & transition.tiny
      small_dual = (
          max_abs(
              direction.dlmbda, direction.dbeta, direction.dy, direction.dyf
          )
          < o.tiny_step_y_tol
      )
      if o.print_level:
        jax.debug.print(
            'IPPAS {i}: phase={phase} f={f:.8e} primal={p:.2e} dual={d:.2e}'
            ' compl={c:.2e} mu={mu:.2e}',
            i=transition.iteration,
            phase=transition.phase,
            f=data.objective,
            p=mm.primal,
            d=mm.dual,
            c=mm.complementarity,
            mu=new_mu,
        )
      return State(
          iterate=it,
          system=system,
          data=data,
          metrics=mm,
          iteration=transition.iteration,
          status=status,
          acceptable_count=count,
          mu=new_mu,
          line_search=fs,
          last_delta=transition.last_delta,
          filter_rejections=rejections,
          filter_resets=resets,
          tiny_step_last_iteration=tiny & small_dual,
          tiny_step_flag=tiny & state.tiny_step_last_iteration,
          watchdog=transition.watchdog,
          soft_count=transition.soft_count,
          skip_first=transition.skip_first,
      )

    def body(state: State) -> State:
      new_mu = ip_mu_update.update(
          state.mu,
          lambda mu: ip_iterate.metrics(
              state.iterate, state.data, mu, bounds, original_bounds, o
          ).barrier_error,
          o,
          tiny_step_flag=state.tiny_step_flag,
          first_update=state.iteration == 0,
      )
      # Repeated tiny steps force μ down. At its floor, retain the last accepted
      # iterate and report stagnation.
      stagnant = state.tiny_step_flag & (new_mu == state.mu)
      return jax.lax.cond(
          stagnant,
          lambda _: state._replace(status=jp.asarray(Status.TINY_STEP, jp.int32)),
          lambda _: step(state, new_mu),
          None,
      )

    intermediates = None
    if o.record_intermediates:
      intermediates = jax.tree.map(
          lambda v: jp.zeros(
              (o.max_iter + 1, *v.shape), dtype=v.dtype
          ).at[0].set(v),
          state.iterate,
      )

      def body_with_history(carry):
        state, history = carry
        state = body(state)
        history = jax.tree.map(
            lambda h, v: h.at[state.iteration].set(v), history, state.iterate
        )
        return state, history

      final, intermediates = jax.lax.while_loop(
          lambda carry: carry[0].status == Status.RUNNING,
          body_with_history,
          (state, intermediates),
      )
    else:
      final = jax.lax.while_loop(
          lambda state: state.status == Status.RUNNING, body, state
      )
    mm = final.metrics

    return SolveResult(
        final.iterate,
        final.status,
        final.iteration,
        final.data.objective,
        mm.primal,
        mm.dual,
        mm.complementarity,
        mm.error,
        final.mu,
        intermediates,
    )


def check_convergence(
    model: StagewiseOcpModel,
    it: ip_iterate.Iterate,
    system: pd.IpPrimalDualSystem,
    data: ip_iterate.IterateData,
    metrics: ip_iterate.Metrics,
    count: jax.Array,
    iteration: jax.Array,
    bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    options: Options,
) -> jax.Array:
  o = options
  status = jp.where(iteration >= o.max_iter, Status.MAX_ITER, Status.RUNNING)
  feasible = metrics.primal <= o.constr_viol_tol
  status = jp.where(
      (o.acceptable_iter > 0) & (count >= o.acceptable_iter) & feasible,
      Status.ACCEPTABLE,
      status,
  )
  status = jp.where((metrics.error <= o.tol) & feasible, Status.SUCCESS, status)
  status = jp.where(
      ip_iterate.valid(it, bounds, model.quaternion_addrs)
      & all_finite((
          system.A,
          system.B,
          system.Ex,
          system.Eu,
          system.Gx,
          system.Gu,
          system.Gf,
          system.E0,
          data,
          metrics,
      )),
      status,
      Status.NUMERICAL_ERROR,
  )
  return status.astype(jp.int32)
