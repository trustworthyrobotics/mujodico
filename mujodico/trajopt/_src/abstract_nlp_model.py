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
      nnzj=length(jrows), nnzh=length(hrows), sparse_jacobian=true,
      sparse_hessian=true, hess_available=true)
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
function NLPModels.cons!(nlp::PythonNLPModel,
                         x::AbstractVector{T}, c::AbstractVector{T}) where {T}
  copyto!(c, PythonCall.pyconvert(Vector{Float64},
      PythonCall.pycall(PythonCall.pygetattr(nlp.py_nlp, "constraints"), x)))
  return c
end
function NLPModels.jac_structure!(nlp::PythonNLPModel,
                                  rows::AbstractVector{I},
                                  cols::AbstractVector{I}) where {I<:Integer}
  copyto!(rows, nlp.jrows); copyto!(cols, nlp.jcols)
  return rows, cols
end
function NLPModels.jac_coord!(nlp::PythonNLPModel,
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
                               obj_weight=1, y=zeros(nlp.meta.ncon)) where {T}
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
    self._jac_rows, self._jac_cols = _validate_structure(
        nlp.jacobian_structure(), nlp.num_constraints, nlp.num_vars
    )
    self._hess_rows, self._hess_cols = _validate_structure(
        nlp.hessian_structure(), nlp.num_vars, nlp.num_vars
    )
    self._x = None

  def objective(self, x):
    self._x = x
    return self._nlp.objective(x)

  def gradient(self, x):
    self._x = x
    return self._nlp.gradient(x)

  def constraints(self, x):
    self._x = x
    return self._nlp.constraints(x)

  def jacobianstructure(self):
    return self._jac_rows, self._jac_cols

  def jacobian(self, x):
    self._x = x
    return self._nlp.jacobian_entries(x)

  def hessianstructure(self):
    return self._hess_rows, self._hess_cols

  def hessian(self, x, lagrange, objective_factor):
    self._x = x
    return self._nlp.hessian_entries(x, lagrange, objective_factor)

  def intermediate(
      self, alg_mod, iter, objective, inf_pr, inf_du, mu,
      d_norm, regularization_size, alpha_du, alpha_pr, ls_trials,
  ):
    self._nlp.intermediate({
        'x': self._x,
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

  callbacks = _CyIpoptCallbacks(nlp)
  var_lb, var_ub = _validate_bounds(nlp.var_bounds, nlp.num_vars)
  con_lb, con_ub = _validate_bounds(nlp.constraint_bounds, nlp.num_constraints)

  return cyipopt.Problem(
      n=nlp.num_vars,
      m=nlp.num_constraints,
      problem_obj=callbacks,
      lb=var_lb,
      ub=var_ub,
      cl=con_lb,
      cu=con_ub,
  )


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
