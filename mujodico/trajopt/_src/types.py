from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
import jax
import numpy as np


_X = jax.Array
_U = jax.Array
Params = object
StageCost = Callable[[_X, _U, Params], jax.Array]
TerminalCost = Callable[[_X, Params], jax.Array]
InitialEqualityConstraint = Callable[[_X, Params], jax.Array]
LocalEqualityConstraint = Callable[[_X, _U, _X, Params], jax.Array]
LocalInequalityConstraint = Callable[[_X, _U, _X, Params], jax.Array]
TerminalEqualityConstraint = Callable[[_X, Params], jax.Array]
TerminalInequalityConstraint = Callable[[_X, Params], jax.Array]
Bounds = tuple[np.ndarray, np.ndarray]


@dataclass
class LocalComplementarityConstraint:
  """Complementarity pairs over local variables and inequality constraints.

  A term is a ``(side, index)`` pair. The index is 0-based within the local
  vector identified by ``side``.
  """

  class Side(Enum):
    X = 'x'
    U = 'u'
    XNEXT = 'xnext'
    INEQ = 'ineq'

  Term = tuple[Side, int]

  pairs: list[tuple[Term, Term]] = field(default_factory=list)

  @property
  def size(self):
    return len(self.pairs)

  def add_ctrl_ineq_complementarity(self, ctrl_index, ineq_index) -> None:
    self._add_index_pairs(self.Side.U, ctrl_index, self.Side.INEQ, ineq_index)

  def add_ineq_ineq_complementarity(self, ineq_index_1, ineq_index_2) -> None:
    self._add_index_pairs(self.Side.INEQ, ineq_index_1, self.Side.INEQ, ineq_index_2)

  def _add_index_pairs(self, side_1, indices_1, side_2, indices_2):
    values_1 = self._indices(indices_1)
    values_2 = self._indices(indices_2)
    if len(values_1) != len(values_2):
      raise ValueError('index sequences must have equal lengths')
    self.pairs.extend(
        ((side_1, index_1), (side_2, index_2))
        for index_1, index_2 in zip(values_1, values_2)
    )

  @staticmethod
  def _indices(value) -> list[int]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
      values = list(value)
    else:
      values = [value]
    if not values:
      raise ValueError('at least one complementarity index is required')
    if any(
        not isinstance(index, Integral) or isinstance(index, bool)
        for index in values
    ):
      raise TypeError('indices must be integers or iterables of integers')
    if any(index < 0 for index in values):
      raise ValueError('complementarity indices must be nonnegative integers')
    return [int(index) for index in values]
