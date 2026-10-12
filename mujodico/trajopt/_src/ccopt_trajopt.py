from enum import Enum
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
                  local complementarity constraints                    for k=0,...,N

  Falls back to IPOPT if there are no complementarity constraints.
  """

  class Method(Enum):
    RELAXATION = 'relaxation'
    PENALTY = 'penalty'
    CROSSOVER = 'crossover'

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
      method: Method,
      **kwargs,
  ) -> tuple[np.ndarray, np.ndarray, dict]:
    nlp = self._nlp
    nlp.set_params(params)

    vars_init = nlp._pack(np.asarray(xs_init), np.asarray(us_init))

    jl = self._jl
    jl_nlp = nlp.to_nlpmodels_jl(vars_init)

    if not self._has_complementarity_constraint:
      self._ensure_jl_packages(('NLPModelsIpopt',))
      tic = time.perf_counter()
      stats = jl.NLPModelsIpopt.ipopt(jl_nlp, **kwargs)
    else:
      self._ensure_jl_packages(('CCOpt', 'MPCCModels'))
      mpcc = self._make_jl_mpcc(jl_nlp)
      if not jl.MPCCModels.is_vertical(mpcc):
        mpcc = jl.MPCCModels.vertical_form(mpcc)

      if method == self.Method.RELAXATION:
        solver = jl.CCOpt.RelaxationSolver(mpcc, **kwargs)
        tic = time.perf_counter()
        stats = getattr(jl.CCOpt, 'solve_homotopy!')(solver)
      elif method == self.Method.PENALTY:
        solver = jl.CCOpt.PenaltySolver(mpcc, **kwargs)
        tic = time.perf_counter()
        stats = getattr(jl.CCOpt, 'solve_homotopy!')(solver)
      elif method == self.Method.CROSSOVER:
        jl_kwargs = self._kwargs_to_namedtuple(kwargs)
        options = jl.CCOpt.CrossoverOptions(
            lpcc_solver_kwargs=jl_kwargs,
            bnlp_solver_kwargs=jl_kwargs,
        )
        tic = time.perf_counter()
        stats = jl.CCOpt.crossover(mpcc, mpcc.nlp.meta.parent.x0, options)

    total_time = time.perf_counter() - tic
    info = {'total_time': total_time}
    info.update(nlp.stats)
    for name in ['objective', 'status', 'iter', 'primal_feas', 'dual_feas']:
      try:
        info[name] = getattr(stats, name)
      except AttributeError:
        pass

    # Ignore the vertical form variables in the solution.
    vars_ops = np.asarray(stats.solution)[:nlp.num_vars]
    xs_opt, us_opt = nlp._unpack(vars_ops)

    return np.asarray(xs_opt), np.asarray(us_opt), info

  def _kwargs_to_namedtuple(self, kwargs):
    jl = self._jl
    kwargs_keys = tuple(jl.Symbol(key) for key in kwargs)
    kwargs_values = tuple(kwargs.values())
    jl.seval(
        '_kwargs_to_namedtuple(keys, values) = '
        'NamedTuple{keys}(values)'
    )
    return jl._kwargs_to_namedtuple(kwargs_keys, kwargs_values)

  def _make_jl_mpcc(self, jl_nlp):
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
        if is_con_1 and is_con_2:
          cc_names.append('ConCon')
        elif is_con_2:
          cc_names.append('VarCon')
        elif is_con_1:
          index_1, index_2 = index_2, index_1
          cc_names.append('VarCon')
        else:
          cc_names.append('VarVar')
        ind_cc1.append(index_1 + 1)
        ind_cc2.append(index_2 + 1)

    jl = self._jl
    jl_ind_cc1 = jl.collect(jl.Int64, ind_cc1)
    jl_ind_cc2 = jl.collect(jl.Int64, ind_cc2)

    if np.all([name == 'VarVar' for name in cc_names]):
      mpcc = jl.MPCCModels.MPCCModel(jl_nlp, jl_ind_cc1, jl_ind_cc2)
      assert jl.MPCCModels.is_vertical(mpcc)
    elif np.all([name == 'VarCon' for name in cc_names]):
      mpcc = jl.MPCCModels.MPCCModelVarCon(jl_nlp, jl_ind_cc1, jl_ind_cc2)
      assert not jl.MPCCModels.is_vertical(mpcc)
    elif np.all([name == 'ConCon' for name in cc_names]):
      mpcc = jl.MPCCModels.MPCCModelConCon(jl_nlp, jl_ind_cc1, jl_ind_cc2)
      assert not jl.MPCCModels.is_vertical(mpcc)
    else:
      cc_types = jl.collect(
          jl.MPCCModels.CCType,
          [getattr(jl.MPCCModels, name) for name in cc_names],
      )
      # MPCCModels auto converts this case into vertical form.
      mpcc = jl.MPCCModels.MPCCModel(jl_nlp, jl_ind_cc1, jl_ind_cc2, cc_types)
      assert jl.MPCCModels.is_vertical(mpcc)

    return mpcc

  def _ensure_jl_packages(self, packages: tuple[str, ...]) -> None:
    using_statement = '; '.join(f'using {package}' for package in packages)
    try:
      self._jl.seval(using_statement)
      return
    except Exception:
      package_list = ', '.join(f'"{package}"' for package in packages)
      try:
        self._jl.seval(f'import Pkg; Pkg.add([{package_list}])')
        self._jl.seval(using_statement)
      except Exception as e:
        names = ', '.join(packages)
        raise RuntimeError(
            f'Could not load or install Julia package(s): {names}.'
        ) from e
