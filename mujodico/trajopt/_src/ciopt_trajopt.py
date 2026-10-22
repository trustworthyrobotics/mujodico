from enum import Enum
import importlib
import time
import jax
import numpy as np
from mujodico.trajopt._src.ccopt_trajopt import CCOptTrajOpt
from mujodico.trajopt._src.stagewise_nlp_model import StagewiseNLPModel
from mujodico.trajopt._src.types import (
  Bounds,
  InitialEqualityConstraint,
  LocalComplementarityConstraint,
  LocalEqualityConstraint,
  LocalInequalityConstraint,
  Params,
  StageCost,
  TerminalCost,
  TerminalEqualityConstraint,
  TerminalInequalityConstraint,
)


class CIOptTrajOpt(CCOptTrajOpt):
  """Solves the following stagewise MPCC with CIOpt

    minimizeₓ,ᵤ   ∑ₖ₌₀ᴺ⁻¹ stage_cost(xₖ, uₖ, params) + terminal_cost(x_N, params)
    subject to    initial_equality_constraint(x₀, params) = 0
                  local_equality_constraint(xₖ, uₖ, xₖ₊₁, params) = 0    for k=0,...,N-1
                  local_inequality_constraint(xₖ, uₖ, xₖ₊₁, params) ≥ 0  for k=0,...,N-1
                  terminal_equality_constraint(x_N, params) = 0
                  terminal_inequality_constraint(x_N, params) ≥ 0
                  x_lb ≤ xₖ ≤ x_ub                                      for k=0,...,N
                  u_lb ≤ uₖ ≤ u_ub                                      for k=0,...,N-1
                  local complementarity constraints                    for k=0,...,N
  """

  class Method(Enum):
    IMPACT = 'impact'

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
      local_complementarity_constraint: LocalComplementarityConstraint | None = None,
      terminal_equality_constraint: TerminalEqualityConstraint | None = None,
      terminal_inequality_constraint: TerminalInequalityConstraint | None = None,
      x_bounds: Bounds | None = None,
      u_bounds: Bounds | None = None,
      gauss_newton: bool = True,
      params_trace: Params | None = None,
  ):
    try:
      juliacall = importlib.import_module('juliacall')
      self._jl = juliacall.Main
    except ImportError as e:
      raise ImportError(
          'Optional Julia bridge dependency not installed. Install juliacall with\n\n'
          '    pip install juliacall'
      ) from e

    super().__init__(
        N=N,
        nx=nx,
        nu=nu,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        initial_equality_constraint=initial_equality_constraint,
        local_equality_constraint=local_equality_constraint,
        local_inequality_constraint=local_inequality_constraint,
        local_complementarity_constraint=local_complementarity_constraint,
        terminal_equality_constraint=terminal_equality_constraint,
        terminal_inequality_constraint=terminal_inequality_constraint,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        gauss_newton=gauss_newton,
        params_trace=params_trace,
    )

  def solve(
      self,
      params: Params,
      xs_init: jax.Array | np.ndarray,
      us_init: jax.Array | np.ndarray,
      method: Method,
      **kwargs,
  ) -> tuple[np.ndarray, np.ndarray, dict]:
    if not self._has_complementarity_constraint:
      return super().solve(
          params=params,
          xs_init=xs_init,
          us_init=us_init,
          **kwargs,
      )

    nlp = self._nlp
    nlp.set_params(params)

    vars_init = nlp._pack(np.asarray(xs_init), np.asarray(us_init))

    jl = self._jl
    jl_nlp = nlp.to_nlpmodels_jl(vars_init)

    self._ensure_jl_packages(('CIOpt', 'MPCCModels'))
    mpcc = self._make_jl_mpcc(jl_nlp, convert_vertical_form=False)

    if method == self.Method.IMPACT:
      solver = jl.CIOpt.ImpactSolver(mpcc, **kwargs)
      tic = time.perf_counter()
      stats = getattr(jl.CIOpt, 'solve!')(solver)

    total_time = time.perf_counter() - tic
    info = {'total_time': total_time}
    info.update(nlp.stats)
    for name in ['objective', 'status', 'iter']:
      try:
        info[name] = getattr(stats, name)
      except AttributeError:
        pass

    vars_ops = np.asarray(stats.z)
    xs_opt, us_opt = nlp._unpack(vars_ops)

    return np.asarray(xs_opt), np.asarray(us_opt), info
