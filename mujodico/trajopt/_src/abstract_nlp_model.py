from abc import ABC, abstractmethod
import importlib
import logging
import numpy as np


class AbstractNLPModel(ABC):

  @property
  @abstractmethod
  def num_vars(self) -> int:
    """Number of decision variables in the optimization problem."""

  @property
  @abstractmethod
  def var_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper variable bounds, each shaped ``(num_vars,)``."""

  @property
  @abstractmethod
  def num_constraints(self) -> int:
    """Number of nonlinear constraints in the optimization problem."""

  @property
  @abstractmethod
  def constraint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper constraint bounds, each shaped ``(num_constraints,)``."""

  @abstractmethod
  def objective(self, x: np.ndarray) -> float:
    """Evaluate the scalar objective at a decision vector shaped ``(num_vars,)``."""

  @abstractmethod
  def gradient(self, x: np.ndarray) -> np.ndarray:
    """Evaluate the objective gradient, returned as an array shaped ``(num_vars,)``."""

  @abstractmethod
  def constraints(self, x: np.ndarray) -> np.ndarray:
    """Evaluate the constraint vector, returned as an array shaped ``(num_constraints,)``."""

  @abstractmethod
  def jacobian_structure(self) -> list[tuple[int, int]]:
    """Return 0-based ``(row, column)`` indices for the sparse constraint Jacobian.

    The order of the pairs defines the order of values returned by
    :meth:`jacobian_entries`.
    """

  @abstractmethod
  def jacobian_entries(self, x: np.ndarray) -> np.ndarray:
    """Return Jacobian values in the order specified by :meth:`jacobian_structure`."""

  @abstractmethod
  def hessian_structure(self) -> list[tuple[int, int]]:
    """Return 0-based ``(row, column)`` indices for the sparse Lagrangian Hessian.

    The order of the pairs defines the order of values returned by
    :meth:`hessian_entries`. Use a triangular representation when required by
    the target solver.
    """

  @abstractmethod
  def hessian_entries(
      self,
      x: np.ndarray,
      lagrange: np.ndarray,
      objective_factor: float,
  ) -> np.ndarray:
    """Return Hessian values in the order specified by :meth:`hessian_structure`."""

  def intermediate(self, stats: dict) -> None:
    """Called after each solver iteration."""
    pass

  def to_nlpmodels_jl(self, x0: np.ndarray | None = None):
    """Wrap this NLP as an ``NLPModels.AbstractNLPModel`` in Julia.
    The returned Julia model can be passed directly to solvers such as
    NLPModelsIpopt or MadNLP.
    ``x0`` is the initial guess required by NLPModels.
    """
    return _to_nlpmodels_jl(self, x0)

  def to_cyipopt(self):
    """Build and return a ``cyipopt.Problem`` for this NLP.
    The returned problem can be solved with ``problem.solve(x0)``.
    """
    return _to_cyipopt(self)

  def to_knitro(self):
    """Return a Knitro adapter with ``add_option`` and ``solve(x0)`` methods.

    Set ``nlp_algorithm`` to 1 for interior-point/direct, 2 for
    interior-point/CG, or 4 for SQP. Otherwise Knitro selects the algorithm.
    ``solve`` returns ``(x, info)``; ``info`` contains ``status``, ``obj_val``,
    ``x``, and signed multipliers ``mult_g`` (constraints) and ``mult_x``
    (variable bounds). Each solve creates and frees its own solver context.
    ``intermediate`` receives ``x``, ``iter``, ``objective``, ``inf_pr``, and
    ``inf_du`` after each iteration.
    Requires the optional ``knitro`` package and a valid Knitro license.
    """
    return _to_knitro(self)


_JULIA_ADAPTER_SETUP = r"""
using NLPModels
using PythonCall

mutable struct PythonNLPModel{T,S<:AbstractVector{T}} <: NLPModels.AbstractNLPModel{T,S}
  meta::NLPModels.NLPModelMeta{T,S}
  counters::NLPModels.Counters
  py_nlp::PythonCall.Py
  jrows::Vector{Int}
  jcols::Vector{Int}
  hrows::Vector{Int}
  hcols::Vector{Int}
end

function PythonNLPModel(py_nlp, x0, lvar, uvar, lcon, ucon, jrows, jcols, hrows, hcols)
  n, m = length(x0), length(lcon)
  meta = NLPModels.NLPModelMeta(n; x0=Float64.(x0), lvar=Float64.(lvar),
      uvar=Float64.(uvar), ncon=m, lcon=Float64.(lcon), ucon=Float64.(ucon),
      nnzh=length(hrows), nnzj=length(jrows), sparse_jacobian=true, sparse_hessian=true,
      jprod_available=false, jtprod_available=false, hprod_available=false)
  return PythonNLPModel(meta, NLPModels.Counters(), PythonCall.Py(py_nlp),
      Int.(jrows), Int.(jcols), Int.(hrows), Int.(hcols))
end

function NLPModels.obj(nlp::PythonNLPModel, x::AbstractVector{T}) where {T}
  return PythonCall.pyconvert(Float64,
      PythonCall.pycall(PythonCall.pygetattr(nlp.py_nlp, "objective"), x))
end

function NLPModels.grad!(nlp::PythonNLPModel,
                         x::AbstractVector{T}, g::AbstractVector{T}) where {T}
  copyto!(g, PythonCall.pyconvert(Vector{Float64},
      PythonCall.pycall(PythonCall.pygetattr(nlp.py_nlp, "gradient"), x)))
  return g
end

function NLPModels.cons_lin!(nlp::PythonNLPModel,
                             x::AbstractVector{T}, c::AbstractVector{T}) where {T}
  @assert isempty(c)
  return c
end

function NLPModels.cons_nln!(nlp::PythonNLPModel,
                             x::AbstractVector{T}, c::AbstractVector{T}) where {T}
  copyto!(c, PythonCall.pyconvert(Vector{Float64},
      PythonCall.pycall(PythonCall.pygetattr(nlp.py_nlp, "constraints"), x)))
  return c
end

function NLPModels.jac_lin_structure!(nlp::PythonNLPModel,
                                      rows::AbstractVector{I},
                                      cols::AbstractVector{I}) where {I<:Integer}
  @assert isempty(rows) && isempty(cols)
  return rows, cols
end

function NLPModels.jac_nln_structure!(nlp::PythonNLPModel,
                                  rows::AbstractVector{I},
                                  cols::AbstractVector{I}) where {I<:Integer}
  copyto!(rows, nlp.jrows); copyto!(cols, nlp.jcols)
  return rows, cols
end

function NLPModels.jac_lin_coord!(nlp::PythonNLPModel,
                                  x::AbstractVector{T},
                                  vals::AbstractVector{T}) where {T}
  @assert isempty(vals)
  return vals
end

function NLPModels.jac_nln_coord!(nlp::PythonNLPModel,
                              x::AbstractVector{T},
                              vals::AbstractVector{T}) where {T}
  copyto!(vals, PythonCall.pyconvert(Vector{Float64},
      PythonCall.pycall(PythonCall.pygetattr(nlp.py_nlp, "jacobian_entries"), x)))
  return vals
end

function NLPModels.hess_structure!(nlp::PythonNLPModel,
                                   rows::AbstractVector{I},
                                   cols::AbstractVector{I}) where {I<:Integer}
  copyto!(rows, nlp.hrows); copyto!(cols, nlp.hcols)
  return rows, cols
end

function NLPModels.hess_coord!(nlp::PythonNLPModel,
                               x::AbstractVector{T},
                               vals::AbstractVector{T};
                               obj_weight=1) where {T}
  y = zeros(nlp.meta.ncon)
  return NLPModels.hess_coord!(nlp, x, y, vals; obj_weight=obj_weight)
end

function NLPModels.hess_coord!(nlp::PythonNLPModel,
                               x::AbstractVector{T},
                               y::AbstractVector{T},
                               vals::AbstractVector{T};
                               obj_weight=1) where {T}
  copyto!(vals, PythonCall.pyconvert(Vector{Float64},
      PythonCall.pycall(PythonCall.pygetattr(nlp.py_nlp, "hessian_entries"),
                        x, y, obj_weight)))
  return vals
end
"""
_julia_module = None


def _to_nlpmodels_jl(nlp: AbstractNLPModel, x0: np.ndarray | None = None):
  try:
    juliacall = importlib.import_module('juliacall')
  except ImportError as e:
    raise ImportError(
        'Optional Julia bridge dependency not installed. Install juliacall with\n\n'
        '    pip install juliacall'
    ) from e

  x0 = np.asarray(x0, dtype=np.float64) if x0 is not None else np.zeros(nlp.num_vars)
  if x0.shape != (nlp.num_vars,):
    raise ValueError(f'x0 must have shape ({nlp.num_vars},), got {x0.shape}')

  var_lb, var_ub = _validate_bounds(nlp.var_bounds, nlp.num_vars)
  con_lb, con_ub = _validate_bounds(nlp.constraint_bounds, nlp.num_constraints)

  jr, jc = _validate_structure(
      nlp.jacobian_structure(), nlp.num_constraints, nlp.num_vars
  )
  hr, hc = _validate_structure(
      nlp.hessian_structure(), nlp.num_vars, nlp.num_vars
  )

  global _julia_module
  if _julia_module is None:
    try:
      juliacall.Main.seval('using NLPModels')
    except juliacall.JuliaError:
      try:
        juliacall.Main.seval('import Pkg; Pkg.add("NLPModels")')
        juliacall.Main.seval('using NLPModels')
      except juliacall.JuliaError as e:
        raise RuntimeError(
            'Could not install NLPModels in juliacall\'s Julia environment.'
        ) from e

    julia_module = juliacall.newmodule('PythonNLPModels')
    julia_module.seval(_JULIA_ADAPTER_SETUP)
    _julia_module = julia_module

  # Julia NLPModels uses 1-based triplet indices; Python solvers use 0-based.
  return _julia_module.PythonNLPModel(
      nlp, x0, var_lb, var_ub, con_lb, con_ub, jr + 1, jc + 1, hr + 1, hc + 1
  )


class _CyIpoptCallbacks:
  """Adapt an ``AbstractNLPModel`` to CyIpopt's callback interface."""

  def __init__(self, nlp: AbstractNLPModel):
    self._nlp = nlp
    self._problem = None
    self._jac_rows, self._jac_cols = _validate_structure(
        nlp.jacobian_structure(), nlp.num_constraints, nlp.num_vars
    )
    self._hess_rows, self._hess_cols = _validate_structure(
        nlp.hessian_structure(), nlp.num_vars, nlp.num_vars
    )

  def objective(self, x):
    return self._nlp.objective(x)

  def gradient(self, x):
    return self._nlp.gradient(x)

  def constraints(self, x):
    return self._nlp.constraints(x)

  def jacobianstructure(self):
    return self._jac_rows, self._jac_cols

  def jacobian(self, x):
    return self._nlp.jacobian_entries(x)

  def hessianstructure(self):
    return self._hess_rows, self._hess_cols

  def hessian(self, x, lagrange, objective_factor):
    return self._nlp.hessian_entries(x, lagrange, objective_factor)

  def intermediate(
      self, alg_mod, iter, objective, inf_pr, inf_du, mu,
      d_norm, regularization_size, alpha_du, alpha_pr, ls_trials,
  ):
    current = self._problem.get_current_iterate(scaled=False)
    self._nlp.intermediate({
        'x': np.asarray(current['x']).copy(),
        'alg_mod': alg_mod,
        'iter': iter,
        'objective': objective,
        'inf_pr': inf_pr,
        'inf_du': inf_du,
        'mu': mu,
        'd_norm': d_norm,
        'regularization_size': regularization_size,
        'alpha_du': alpha_du,
        'alpha_pr': alpha_pr,
        'ls_trials': ls_trials,
    })


def _to_cyipopt(nlp: AbstractNLPModel):
  try:
    cyipopt = importlib.import_module('cyipopt')
    logging.getLogger('cyipopt').setLevel(logging.WARNING)
  except ImportError as e:
    raise ImportError(
        'Optional solver dependency not installed. Install it with:\n\n'
        '    pip install cyipopt'
    ) from e

  if cyipopt.IPOPT_VERSION < (3, 14, 0):
    version = '.'.join(str(part) for part in cyipopt.IPOPT_VERSION)
    raise ImportError(
        'IPOPT >=3.14.0 is required for intermediate iterate access; '
        f'found {version}. Upgrade IPOPT and reinstall cyipopt.'
    )

  callbacks = _CyIpoptCallbacks(nlp)
  var_lb, var_ub = _validate_bounds(nlp.var_bounds, nlp.num_vars)
  con_lb, con_ub = _validate_bounds(nlp.constraint_bounds, nlp.num_constraints)

  problem = cyipopt.Problem(
      n=nlp.num_vars,
      m=nlp.num_constraints,
      problem_obj=callbacks,
      lb=var_lb,
      ub=var_ub,
      cl=con_lb,
      cu=con_ub,
  )
  callbacks._problem = problem
  return problem


class _KnitroProblem:

  def __init__(self, nlp: AbstractNLPModel, knitro):
    self._nlp = nlp
    self._knitro = knitro
    self._options = {}

  def add_option(self, key, value):
    self._options[key] = value

  def solve(self, x0):
    nlp = self._nlp
    kn = self._knitro

    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (nlp.num_vars,):
      raise ValueError(f'x0 must have shape ({nlp.num_vars},), got {x0.shape}')

    var_lb, var_ub = _validate_bounds(nlp.var_bounds, nlp.num_vars)
    con_lb, con_ub = _validate_bounds(nlp.constraint_bounds, nlp.num_constraints)
    jr, jc = _validate_structure(
        nlp.jacobian_structure(), nlp.num_constraints, nlp.num_vars
    )
    hr, hc = _validate_structure(
        nlp.hessian_structure(), nlp.num_vars, nlp.num_vars
    )

    # Knitro expects the upper triangle. Keep the original entry ordering.
    hr, hc = np.minimum(hr, hc), np.maximum(hr, hc)
    if len(np.unique(np.column_stack((hr, hc)), axis=0)) != len(hr):
      raise ValueError('Hessian structure must contain each symmetric entry once')

    with kn.KN_new() as kc:
      kn.KN_add_vars(kc, nlp.num_vars)
      kn.KN_set_var_lobnds(kc, xLoBnds=np.maximum(var_lb, -kn.KN_INFINITY))
      kn.KN_set_var_upbnds(kc, xUpBnds=np.minimum(var_ub, kn.KN_INFINITY))
      if nlp.num_constraints:
        kn.KN_add_cons(kc, nlp.num_constraints)
        kn.KN_set_con_lobnds(kc, cLoBnds=np.maximum(con_lb, -kn.KN_INFINITY))
        kn.KN_set_con_upbnds(kc, cUpBnds=np.minimum(con_ub, kn.KN_INFINITY))
      kn.KN_set_obj_goal(kc, kn.KN_OBJGOAL_MINIMIZE)

      cb = kn.KN_add_eval_callback(kc, funcCallback=self._eval_fc)
      kn.KN_set_cb_grad(
          kc,
          cb,
          objGradIndexVars=kn.KN_DENSE,
          jacIndexCons=jr,
          jacIndexVars=jc,
          gradCallback=self._eval_ga,
      )
      kn.KN_set_cb_hess(
          kc,
          cb,
          hessIndexVars1=hr,
          hessIndexVars2=hc,
          hessCallback=self._eval_h,
      )

      kn.KN_set_newpt_callback(kc, self._intermediate)
      kn.KN_set_var_primal_init_values(kc, xInitVals=x0)

      kn.KN_set_int_param(kc, 'hessian_no_f', 1)
      for key, value in self._options.items():
        if isinstance(value, (int, np.integer)):
          kn.KN_set_int_param(kc, key, int(value))
        elif isinstance(value, (float, np.floating)):
          kn.KN_set_double_param(kc, key, float(value))
        elif isinstance(value, str):
          kn.KN_set_char_param(kc, key, value)
        else:
          raise TypeError(f'unsupported Knitro option value for {key!r}: {value!r}')

      kn.KN_solve(kc)
      status, objective, x, multipliers = kn.KN_get_solution(kc)
      x = np.asarray(x, dtype=np.float64).copy()
      multipliers = np.asarray(multipliers, dtype=np.float64)
      info = {
          'status': status,
          'obj_val': objective,
          'x': x.copy(),
          'mult_g': multipliers[: nlp.num_constraints].copy(),
          'mult_x': multipliers[nlp.num_constraints :].copy(),
      }
    return x, info

  def _eval_fc(self, kc, cb, request, result, user_params):
    x = np.asarray(request.x, dtype=np.float64)
    result.obj = self._nlp.objective(x)
    if self._nlp.num_constraints:
      result.c[:] = self._nlp.constraints(x)
    return 0

  def _eval_ga(self, kc, cb, request, result, user_params):
    x = np.asarray(request.x, dtype=np.float64)
    result.objGrad[:] = self._nlp.gradient(x)
    if result.jac is not None:
      result.jac[:] = self._nlp.jacobian_entries(x)
    return 0

  def _eval_h(self, kc, cb, request, result, user_params):
    x = np.asarray(request.x, dtype=np.float64)
    # Knitro appends variable-bound multipliers after constraint multipliers.
    lagrange = np.asarray(request.lambda_, dtype=np.float64)
    if result.hess is not None:
      result.hess[:] = self._nlp.hessian_entries(
          x, lagrange[: self._nlp.num_constraints], request.sigma
      )
    return 0

  def _intermediate(self, kc, x, multipliers, user_params):
    kn = self._knitro
    self._nlp.intermediate({
        'x': np.asarray(x, dtype=np.float64).copy(),
        'iter': kn.KN_get_number_iters(kc),
        'objective': kn.KN_get_obj_value(kc),
        'inf_pr': kn.KN_get_abs_feas_error(kc),
        'inf_du': kn.KN_get_abs_opt_error(kc),
    })
    return 0


def _to_knitro(nlp: AbstractNLPModel):
    try:
      knitro = importlib.import_module('knitro')
    except ImportError as e:
      raise ImportError(
          'Optional solver dependency not installed. Install it with:\n\n'
          '    pip install knitro\n\n'
          'A valid Knitro license is also required.'
      ) from e
    return _KnitroProblem(nlp, knitro)


def _validate_bounds(bounds, size: int):
  lb, ub = (np.asarray(bound, dtype=np.float64) for bound in bounds)
  expected_shape = (size,)
  if lb.shape != expected_shape or ub.shape != expected_shape:
    raise ValueError(f'bounds must each have shape {expected_shape}')
  if not np.all(lb <= ub):
    raise ValueError('lower bounds must be less than or equal to upper bounds')
  return lb, ub


def _validate_structure(
    structure: list[tuple[int, int]], nrows: int, ncols: int
) -> tuple[np.ndarray, np.ndarray]:
  indices = np.asarray(structure, dtype=np.int64)
  if indices.size == 0:
    indices = indices.reshape((0, 2))
  if indices.ndim != 2 or indices.shape[1] != 2:
    raise ValueError('sparse structure must be a list of (row, column) pairs')
  rows, cols = indices[:, 0], indices[:, 1]
  if np.any(rows < 0) or np.any(rows >= nrows) or np.any(cols < 0) or np.any(cols >= ncols):
    raise ValueError('sparse structure indices must be 0-based and within bounds')
  return rows, cols
