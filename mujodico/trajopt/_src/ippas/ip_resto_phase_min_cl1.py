"""Run elastic ℓ₁ restoration using the shared filter/SOC search kernel.

Outer μ and the original filter remain fixed during an episode. The nested
barrier solve has its own μ_R and filter, but consumes the global iteration
budget. It cannot recursively restore. Failure returns the untouched entry
iterate; successful return reconstructs original-problem multipliers.
"""

from dataclasses import replace
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_convergence_check_resto as conv
from mujodico.trajopt._src.ippas import ip_initializer_resto as initializer
from mujodico.trajopt._src.ippas import ip_linesearch as ls
from mujodico.trajopt._src.ippas import ip_mu_update
from mujodico.trajopt._src.ippas import ip_primal_dual_system_resto as pd
from mujodico.trajopt._src.ippas import ip_search_dir
from mujodico.trajopt._src.ippas.ip_iterate import Iterate, IterateData, all_finite, max_abs, valid
from mujodico.trajopt._src.ippas.options import Options
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel
from mujodico.trajopt._src.types import Params


class Status(IntEnum):
  RUNNING = 0
  RETURNED = 1
  FAILED = 2
  LOCAL_INFEASIBILITY = 3
  MAX_ITER = 4


class Result(NamedTuple):
  iterate: Iterate
  iteration: jax.Array
  status: jax.Array


class _State(NamedTuple):
  iterate: pd.RestorationIterate
  system: pd.RestorationSystem
  data: IterateData
  mu: jax.Array
  iteration: jax.Array
  steps: jax.Array
  status: jax.Array
  line_search: ls.LineSearchState
  last_delta: jax.Array


def merit(
    it: pd.RestorationIterate,
    values: dict[str, jax.Array],
    mu: jax.Array,
    bounds: pd.Bounds,
    options: Options,
) -> tuple[jax.Array, jax.Array]:
  phi, theta = ls.merit(it.base, values, mu, bounds, options)
  for value in (*it.p, *it.n):
    phi += -mu * jp.sum(jp.log(value)) + options.kappa_d * mu * jp.sum(value)
  return phi, theta


def maximum_step_lengths(
    it: pd.RestorationIterate,
    direction: pd.RestorationSystem,
    mu: jax.Array,
    bounds: pd.Bounds,
    options: Options,
) -> tuple[jax.Array, jax.Array]:
  ap, ad = ls.maximum_step_lengths(it.base, direction.base, mu, bounds, options)
  tau = jp.maximum(options.tau_min, 1 - mu)
  for v, d in zip((*it.p, *it.n), (*direction.dp, *direction.dn)):
    ap = jp.minimum(
        ap,
        ls.maximum_step(v, d, jp.zeros_like(v), jp.full_like(v, jp.inf), tau),
    )
  for v, d in zip((*it.zp, *it.zn), (*direction.dzp, *direction.dzn)):
    ad = jp.minimum(
        ad,
        ls.maximum_step(v, d, jp.zeros_like(v), jp.full_like(v, jp.inf), tau),
    )
  return ap, ad


def candidate(
    model: StagewiseOcpModel,
    it: pd.RestorationIterate,
    direction: pd.RestorationSystem,
    ap: jax.Array,
    ad: jax.Array,
    mu: jax.Array,
    params: Params,
    bounds: pd.Bounds,
    options: Options,
) -> pd.RestorationIterate:
  base = ls.candidate(
      model, it.base, direction.base, ap, ad, mu, params, bounds, options
  )
  p = pd.tree(lambda p, d: p + ap * d, it.p, direction.dp)
  n = pd.tree(lambda n, d: n + ap * d, it.n, direction.dn)

  def dual(z, dz, value):
    central = mu / value
    return jp.clip(
        z + ad * dz,
        central / options.kappa_sigma,
        central * options.kappa_sigma,
    )

  return pd.RestorationIterate(
      base,
      p,
      n,
      pd.tree(dual, it.zp, direction.dzp, p),
      pd.tree(dual, it.zn, direction.dzn, n),
  )


def valid_iterate(
    it: pd.RestorationIterate,
    bounds: pd.Bounds,
    quaternion_addrs: tuple[int, ...],
) -> jax.Array:
  return (
      valid(it.base, bounds, quaternion_addrs)
      & all_finite(it)
      & jp.all(
          jp.stack([jp.all(v > 0) for v in (*it.p, *it.n, *it.zp, *it.zn)])
      )
  )


def compute(
    system: pd.RestorationSystem,
    mu: jax.Array,
    last_delta: jax.Array,
    options: Options,
) -> tuple[pd.RestorationSystem, jax.Array, jax.Array]:
  """Apply the original δw schedule to the full restoration Newton equations."""
  b = system.base
  dc = ip_search_dir.dual_regularization(b.Eu, mu, options)

  def attempt(delta):
    shifted = replace(
        system,
        base=replace(
            b,
            Hxx=b.Hxx + delta * jp.eye(b.Hxx.shape[-1]),
            Huu=b.Huu + delta * jp.eye(b.Huu.shape[-1]),
            delta_c=dc,
            _cache=None,
        ),
        Hs=system.Hs + delta,
        Hsf=system.Hsf + delta,
        delta_w=delta,
        _cache=None,
    )
    return pd.iterative_refinement_solve(
        shifted, options.linear_ref_tol, options.max_refinement
    )

  return ip_search_dir.regularized_solve(
      (b.Hxx, b.Hxu, b.Huu), last_delta, options, attempt
  )


def perform_restoration(
    model: StagewiseOcpModel,
    entry: Iterate,
    data: IterateData,
    params: Params,
    mu: jax.Array,
    bounds: pd.Bounds,
    line_search: ls.LineSearchState,
    iteration: jax.Array,
    options: Options,
) -> Result:
  """Solve one bounded restoration episode, returning original-shaped arrays."""
  it, mu_r = initializer.initialize(entry, data, mu, options)
  zeta = jp.sqrt(mu)
  system, values = pd.evaluate(
      model, it, params, mu_r, bounds, options, entry, zeta
  )
  _, theta = merit(it, vars(values), mu_r, bounds, options)
  entry_merit = ls.merit(entry, vars(data), mu, bounds, options)
  entry_norm = max_abs(data.r0, data.r_f, data.r_e, data.r_g, data.r_gf)
  zero = jp.asarray(0, jp.int32)
  initial = _State(
      it,
      system,
      values,
      mu_r,
      iteration,
      zero,
      zero,
      ls.LineSearchState.create(options.max_iter + 1, theta, options),
      jp.asarray(0.0, mu.dtype),
  )

  def stopping(st):
    original = ls.evaluate_trial(model, st.iterate.base, params)
    returned = (st.steps > 0) & conv.acceptable_return(
        st.iterate.base,
        original,
        entry_merit,
        entry_norm,
        mu,
        bounds,
        line_search,
        model.quaternion_addrs,
        options,
    )
    stationary = (
        conv.error(st.system, st.iterate, st.mu, options, barrier=False)
        <= options.tol
    )
    original_norm = max_abs(
        *(original[name] for name in ls._EQUALITY_CONSTRAINT_FIELDS)
    )
    local = stationary & (original_norm > options.constr_viol_tol)
    status = jp.where(
        st.iteration >= options.max_iter, Status.MAX_ITER, Status.RUNNING
    )
    status = jp.where(
        stationary,
        jp.where(local, Status.LOCAL_INFEASIBILITY, Status.FAILED),
        status,
    )
    status = jp.where(returned, Status.RETURNED, status)
    finite = valid_iterate(
        st.iterate, bounds, model.quaternion_addrs
    ) & all_finite((st.system, st.data))
    return st._replace(
        status=jp.where(finite, status, Status.FAILED).astype(jp.int32)
    )

  def body(st):
    new_mu = ip_mu_update.update(
        st.mu,
        lambda target: conv.error(
            pd.refresh_barrier_terms(st.system, st.mu, target, options),
            st.iterate,
            target,
            options,
        ),
        options,
        first_update=st.steps == 0,
    )
    system = pd.refresh_barrier_terms(st.system, st.mu, new_mu, options)
    fs = jax.lax.cond(
        new_mu != st.mu,
        lambda _: ls.LineSearchState.create(
            options.max_iter + 1,
            merit(st.iterate, vars(st.data), new_mu, bounds, options)[1],
            options,
        ),
        lambda _: st.line_search,
        None,
    )
    direction, good, delta = compute(system, new_mu, st.last_delta, options)
    phi0, theta0 = merit(st.iterate, vars(st.data), new_mu, bounds, options)
    dphi = ls.directional_derivative(
        st.iterate.base, st.data, direction.base, new_mu, bounds, options
    )
    dphi += jp.sum(
        system.Hs * (st.iterate.base.s - entry.s) * direction.base.ds
    )
    dphi += jp.sum(
        system.Hsf * (st.iterate.base.sf - entry.sf) * direction.base.dsf
    )
    for v, d in zip(
        (*st.iterate.p, *st.iterate.n), (*direction.dp, *direction.dn)
    ):
      dphi += jp.sum(
          (
              options.resto_penalty_parameter
              + options.kappa_d * new_mu
              - new_mu / v
          )
          * d
      )
    found = ls.search(
        st.iterate,
        vars(st.data),
        direction,
        fs,
        options,
        phi0,
        theta0,
        dphi,
        lambda d, ap, ad: candidate(
            model, st.iterate, d, ap, ad, new_mu, params, bounds, options
        ),
        lambda point: pd.evaluate_trial(
            model, point, params, entry, zeta, options
        ),
        lambda point, values: merit(point, values, new_mu, bounds, options),
        lambda point: valid_iterate(point, bounds, model.quaternion_addrs),
        lambda d: maximum_step_lengths(st.iterate, d, new_mu, bounds, options),
        lambda d: pd.iterative_refinement_solve(
            d, options.linear_ref_tol, options.max_refinement
        ),
    )
    accepted = good & found.search.accepted
    it = jax.tree.map(
        lambda a, b: jp.where(accepted, a, b), found.search.iterate, st.iterate
    )
    system, data = pd.evaluate(
        model, it, params, new_mu, bounds, options, entry, zeta
    )
    updated = _State(
        it,
        system,
        data,
        new_mu,
        st.iteration + 1,
        st.steps + 1,
        st.status,
        found.state,
        jp.where(delta > 0, delta, st.last_delta),
    )
    updated = stopping(updated)
    updated = updated._replace(
        status=jp.where(accepted, updated.status, Status.FAILED).astype(
            jp.int32
        )
    )
    if options.print_level:
      jax.debug.print(
          'IPPAS R {i}: f_R={f:.8e} E_R={e:.2e} mu_R={m:.2e}',
          i=updated.iteration,
          f=data.objective,
          e=conv.error(system, it, new_mu, options),
          m=new_mu,
      )
    return updated

  initial = stopping(initial)
  # Budget exhaustion prevents even an initial restoration optimization check
  # from overriding the global limit when no iteration remains.
  initial = initial._replace(
      status=jp.where(
          iteration >= options.max_iter, Status.MAX_ITER, initial.status
      ).astype(jp.int32)
  )
  final = jax.lax.while_loop(
      lambda st: st.status == Status.RUNNING, body, initial
  )
  point = jax.lax.cond(
      final.status == Status.RETURNED,
      lambda _: initializer.initialize_original_multipliers(
          model, entry, final.iterate.base, params, mu, bounds, options
      ),
      lambda _: entry,
      None,
  )
  good_return = valid(point, bounds, model.quaternion_addrs) & all_finite(
      ls.evaluate_trial(model, point, params)
  )
  status = jp.where(
      (final.status == Status.RETURNED) & ~good_return,
      Status.FAILED,
      final.status,
  )
  point = jax.tree.map(
      lambda a, b: jp.where(status == Status.RETURNED, a, b), point, entry
  )
  return Result(point, final.iteration, status.astype(jp.int32))
