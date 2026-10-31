"""Primal-dual trajectory optimization with serial or associative scans."""

from .convergence import Status
from .manifold import Euclidean, Quaternion, ProductManifold, CustomManifold
from .options import Options, SolverOptions
from .problem import Problem, OcpProblem, Primal
from .solver import IppasSolver, Solver, InitialGuess, SolveResult

IppasOptions = Options
SolveStatus = Status

__all__ = [
    'IppasSolver',
    'Solver',
    'Options',
    'IppasOptions',
    'SolverOptions',
    'InitialGuess',
    'SolveResult',
    'SolveStatus',
    'Status',
    'Problem',
    'OcpProblem',
    'Primal',
    'Euclidean',
    'Quaternion',
    'ProductManifold',
    'CustomManifold',
]
