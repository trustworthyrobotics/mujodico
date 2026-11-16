from dataclasses import dataclass
from dataclasses import replace
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import ip_initializer
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


class Solver:
  """Primal-dual interior-point solver taking a StagewiseOcpModel directly.

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
          'IPPAS requires float64; enable jax_enable_x64 before constructing the model.'
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
    status = check_convergence(m, it, system, data, metrics, zero, zero, bounds, o)
    status = jp.where(
        ip_iterate.valid(it, bounds, m.quaternion_positions),
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
        line_search=ip_linesearch.LineSearchState.create(o.max_iter + 1, theta(data), o),
        last_delta=jp.asarray(0.0),
        filter_rejections=zero,
        filter_resets=zero,
        tiny_step_last_iteration=jp.asarray(False),
        tiny_step_flag=jp.asarray(False),
    )

    def step(state: State, new_mu: jax.Array) -> State:
      line_search = jax.lax.cond(
          new_mu != state.mu,
          lambda _: ip_linesearch.LineSearchState.create(
              o.max_iter + 1,
              theta(state.data),
              o,
          ),
          lambda _: state.line_search,
          None,
      )

      system = pd.refresh_barrier_terms(state.system, state.iterate, new_mu, bounds, o)

      direction, good, delta = ip_search_dir.compute(
          system, new_mu, state.last_delta, o
      )

      found, line_search, any_filter_rejected, tiny_step_taken = (
          ip_linesearch.find_acceptable_trial_point(
              m,
              state.iterate,
              state.data,
              direction,
              new_mu,
              params,
              bounds,
              line_search,
              o,
          )
      )
      accepted = good & found.accepted

      line_search, rejections, resets = ip_linesearch.update_filter_rejections(
          line_search,
          accepted,
          any_filter_rejected,
          state.filter_rejections,
          state.filter_resets,
          o,
      )

      it = jax.tree.map(
          lambda a, b: jp.where(accepted, a, b), found.iterate, state.iterate
      )

      system, data = jax.lax.cond(
          accepted,
          lambda _: pd.evaluate(m, it, params, new_mu, bounds, o),
          lambda _: (system, state.data),
          None,
      )

      mm = ip_iterate.metrics(it, data, new_mu, bounds, original_bounds, o)

      count = jp.where(mm.error <= o.tol_acceptable, state.acceptable_count + 1, 0)
      iteration = state.iteration + 1
      status = check_convergence(m, it, system, data, mm, count, iteration, bounds, o)

      status = jp.where(~accepted, Status.LINE_SEARCH_FAILED, status)
      status = jp.where(~good, Status.NUMERICAL_ERROR, status).astype(jp.int32)

      if o.print_level:
        jax.debug.print(
            'IPPAS {i}: f={f:.8e} primal={p:.2e} dual={d:.2e} compl={c:.2e}'
            ' mu={mu:.2e} alpha={a:.2e} reg={r:.2e}',
            i=iteration,
            f=data.objective,
            p=mm.primal,
            d=mm.dual,
            c=mm.complementarity,
            mu=new_mu,
            a=found.alpha,
            r=delta,
        )

      tiny = accepted & tiny_step_taken
      small_dual = (
          max_abs(direction.dlmbda, direction.dbeta, direction.dy, direction.dyf)
          < o.tiny_step_y_tol
      )
      return State(
          iterate=it,
          system=system,
          data=data,
          metrics=mm,
          iteration=iteration,
          status=status,
          acceptable_count=count,
          mu=new_mu,
          line_search=line_search,
          last_delta=jp.where(delta > 0, delta, state.last_delta),
          filter_rejections=rejections,
          filter_resets=resets,
          tiny_step_last_iteration=tiny & small_dual,
          tiny_step_flag=tiny & state.tiny_step_last_iteration,
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
      ip_iterate.valid(it, bounds, model.quaternion_positions)
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
