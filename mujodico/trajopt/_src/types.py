from collections.abc import Callable
from typing import TypeVar
import jax


X = jax.Array
U = jax.Array
Params = object
StageCost = Callable[[X, U, Params], jax.Array]
TerminalCost = Callable[[X, Params], jax.Array]
InitialEqualityConstraint = Callable[[X, Params], jax.Array]
LocalEqualityConstraint = Callable[[X, U, X, Params], jax.Array]
LocalInequalityConstraint = Callable[[X, U, X, Params], jax.Array]
TerminalEqualityConstraint = Callable[[X, Params], jax.Array]
TerminalInequalityConstraint = Callable[[X, Params], jax.Array]
Bounds = tuple[jax.Array, jax.Array]
