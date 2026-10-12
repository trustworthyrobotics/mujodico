import time

import jax
import numpy as np

from mujodico.trajopt._src.stagewise_nlp_model import StagewiseNLPModel
from mujodico.trajopt._src.types import (
  Bounds,
  Callable,
  InitialEqualityConstraint,
  LocalEqualityConstraint,
  LocalInequalityConstraint,
  Params,
  StageCost,
  TerminalCost,
  TerminalEqualityConstraint,
  TerminalInequalityConstraint,
)


class IpoptTrajOpt:
  """Solves the following stagewise multiple-shooting NLP with IPOPT.

    minimizeₓ,ᵤ   ∑ₖ₌₀ᴺ⁻¹ stage_cost(xₖ, uₖ, params) + terminal_cost(x_N, params)
    subject to    initial_equality_constraint(x₀, params) = 0
                  local_equality_constraint(xₖ, uₖ, xₖ₊₁, params) = 0    for k=0,...,N-1
                  local_inequality_constraint(xₖ, uₖ, xₖ₊₁, params) ≥ 0  for k=0,...,N-1
                  terminal_equality_constraint(x_N, params) = 0
                  terminal_inequality_constraint(x_N, params) ≥ 0
                  x_lb ≤ xₖ ≤ x_ub                                      for k=0,...,N
                  u_lb ≤ uₖ ≤ u_ub                                      for k=0,...,N-1
  """

  def __init__(
      self,
      N: int,
      nx: int,
      nu: int,
      *,
      stage_cost: StageCost,
      terminal_cost: TerminalCost,
      initial_equality_constraint: InitialEqualityConstraint | None = None,
      local_equality_constraint: LocalEqualityConstraint | None = None,
      local_inequality_constraint: LocalInequalityConstraint | None = None,
      terminal_equality_constraint: TerminalEqualityConstraint | None = None,
      terminal_inequality_constraint: TerminalInequalityConstraint | None = None,
      x_bounds: Bounds | None = None,
      u_bounds: Bounds | None = None,
      gauss_newton: bool = True,
      params_trace: Params | None = None,
      update_params: Callable[[dict, Params], Params] | None = None,
  ):
    self._nlp = StagewiseNLPModel(
        N=N,
        nx=nx,
        nu=nu,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        initial_equality_constraint=initial_equality_constraint,
        local_equality_constraint=local_equality_constraint,
        local_inequality_constraint=local_inequality_constraint,
        terminal_equality_constraint=terminal_equality_constraint,
        terminal_inequality_constraint=terminal_inequality_constraint,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        gauss_newton=gauss_newton,
        params=params_trace,
        update_params=update_params,
    )

  def solve(
      self,
      params: Params,
      xs_init: jax.Array | np.ndarray,
      us_init: jax.Array | np.ndarray,
      **kwargs,
  ) -> tuple[np.ndarray, np.ndarray, dict]:
    nlp = self._nlp
    nlp.set_params(params)

    vars_init = nlp._pack(np.asarray(xs_init), np.asarray(us_init))

    problem = nlp.to_cyipopt()
    for key, value in kwargs.items():
      problem.add_option(key, value)

    tic = time.perf_counter()
    vars_opt, info = problem.solve(vars_init)
    total_time = time.perf_counter() - tic

    info.pop('x', None)
    info['total_time'] = total_time
    info.update(nlp.stats)

    xs_opt, us_opt = nlp._unpack(vars_opt)
    return np.asarray(xs_opt), np.asarray(us_opt), info
