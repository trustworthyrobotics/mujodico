import importlib
import time

import jax
import numpy as np

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


class CcoptTrajOpt:
  """Solves the following stagewise MPCC with CCOpt.

    minimizeₓ,ᵤ   ∑ₖ₌₀ᴺ⁻¹ stage_cost(xₖ, uₖ, params) + terminal_cost(x_N, params)
    subject to    initial_equality_constraint(x₀, params) = 0
                  local_equality_constraint(xₖ, uₖ, xₖ₊₁, params) = 0    for k=0,...,N-1
                  local_inequality_constraint(xₖ, uₖ, xₖ₊₁, params) ≥ 0  for k=0,...,N-1
                  terminal_equality_constraint(x_N, params) = 0
                  terminal_inequality_constraint(x_N, params) ≥ 0
                  x_lb ≤ xₖ ≤ x_ub                                      for k=0,...,N
                  u_lb ≤ uₖ ≤ u_ub                                      for k=0,...,N-1
                  plus complementarity constraints                     for k=0,...,N

  Falls back to IPOPT if there are no complementarity constraints.
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
      local_complementarity_constraint: LocalComplementarityConstraint | None = None,
      terminal_equality_constraint: TerminalEqualityConstraint | None = None,
      terminal_inequality_constraint: TerminalInequalityConstraint | None = None,
      x_bounds: Bounds | None = None,
      u_bounds: Bounds | None = None,
      params_trace: Params | None = None,
      gauss_newton: bool = True,
  ):
    try:
      juliacall = importlib.import_module('juliacall')
      self._julia = juliacall.Main
    except ImportError as e:
      raise ImportError(
          'Optional Julia bridge dependency not installed. Install juliacall with\n\n'
          '    pip install juliacall'
      ) from e

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
        params=params_trace,
        gauss_newton=gauss_newton,
    )
    self._local_complementarity_constraint = local_complementarity_constraint

    self._has_complementarity_constraint = (
        local_complementarity_constraint is not None
        and local_complementarity_constraint.size > 0
        and N > 0
    )

  def solve(
      self,
      params: Params,
      xs_init: jax.Array | np.ndarray,
      us_init: jax.Array | np.ndarray,
      **kwargs,
  ) -> tuple[np.ndarray, np.ndarray, dict]:
    nlp = self._nlp
    nlp.update_params(params)

    vars_init = nlp._pack(np.asarray(xs_init), np.asarray(us_init))

    julia_nlp = nlp.to_nlpmodels_jl(vars_init)

    if self._has_complementarity_constraint:
      self._ensure_julia_packages(('CCOpt', 'MPCCModels'))
      mpcc = self._make_julia_mpcc(julia_nlp)
      solver = self._julia.CCOpt.RelaxationSolver(mpcc, **kwargs)
      tic = time.perf_counter()
      stats = getattr(self._julia.CCOpt, 'solve_homotopy!')(solver)

    else:
      self._ensure_julia_packages(('NLPModelsIpopt',))
      tic = time.perf_counter()
      stats = self._julia.NLPModelsIpopt.ipopt(julia_nlp, **kwargs)

    total_time = time.perf_counter() - tic
    info = {'total_time': total_time}
    info.update(nlp.stats)
    for name in ['objective', 'status', 'iter', 'primal_feas', 'dual_feas', 'multipliers']:
      try:
        info[name] = getattr(stats, name)
      except AttributeError:
        pass

    # Ignore the vertical form variables in the solution.
    vars_ops = np.asarray(stats.solution)[:nlp.num_vars]
    xs_opt, us_opt = nlp._unpack(vars_ops)

    return np.asarray(xs_opt), np.asarray(us_opt), info

  def _ensure_julia_packages(self, packages: tuple[str, ...]) -> None:
    using_statement = '; '.join(f'using {package}' for package in packages)
    try:
      self._julia.seval(using_statement)
      return
    except Exception:
      package_list = ', '.join(f'"{package}"' for package in packages)
      try:
        self._julia.seval(f'import Pkg; Pkg.add([{package_list}])')
        self._julia.seval(using_statement)
      except Exception as e:
        names = ', '.join(packages)
        raise RuntimeError(
            f'Could not load or install Julia package(s): {names}.'
        ) from e

  def _make_julia_mpcc(self, julia_nlp):
    nlp = self._nlp
    n_initial_eq, n_local_eq, n_local_ineq, _, _ = nlp.constraint_sizes
    n_local_constraints = n_local_eq + n_local_ineq

    ind_cc1, ind_cc2, cc_names = [], [], []

    def map_term(term, stage):
      side, index = term
      if side is LocalComplementarityConstraint.Side.X:
        if index >= nlp.nx:
          raise ValueError(f'x index {index} is outside the local state')
        return stage * (nlp.nx + nlp.nu) + index, False
      if side is LocalComplementarityConstraint.Side.U:
        if index >= nlp.nu:
          raise ValueError(f'u index {index} is outside the local control')
        return stage * (nlp.nx + nlp.nu) + nlp.nx + index, False
      if side is LocalComplementarityConstraint.Side.XNEXT:
        if index >= nlp.nx:
          raise ValueError(f'xnext index {index} is outside the next state')
        return (stage + 1) * (nlp.nx + nlp.nu) + index, False
      if side is LocalComplementarityConstraint.Side.INEQ:
        if index >= n_local_ineq:
          raise ValueError(f'local inequality index {index} is outside the inequality size')
        row = n_initial_eq + stage * n_local_constraints + n_local_eq + index
        return row, True
      raise ValueError(f'unsupported complementarity side: {side!r}')

    for stage in range(nlp.N):
      for first, second in self._local_complementarity_constraint.pairs:
        index_1, is_con_1 = map_term(first, stage)
        index_2, is_con_2 = map_term(second, stage)
        ind_cc1.append(index_1 + 1)
        ind_cc2.append(index_2 + 1)
        if is_con_1 and is_con_2:
          cc_names.append('ConCon')
        elif is_con_1:
          cc_names.append('ConVar')
        elif is_con_2:
          cc_names.append('VarCon')
        else:
          cc_names.append('VarVar')

    julia = self._julia
    cc_type = julia.MPCCModels.CCType
    cc_values = [getattr(julia.MPCCModels, name) for name in cc_names]
    mpcc = julia.MPCCModels.MPCCModel(
        julia_nlp,
        julia.collect(julia.Int64, ind_cc1),
        julia.collect(julia.Int64, ind_cc2),
        julia.collect(cc_type, cc_values),
    )
    return julia.MPCCModels.vertical_form(mpcc)
