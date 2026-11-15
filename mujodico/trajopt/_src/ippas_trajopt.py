from collections.abc import Callable
from dataclasses import replace

import jax

from mujodico.trajopt._src.ippas import IppasInitialGuess
from mujodico.trajopt._src.ippas import IppasOptions
from mujodico.trajopt._src.ippas import IppasSolver
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel
from mujodico.trajopt._src.types import Bounds
from mujodico.trajopt._src.types import Params
from mujodico.trajopt._src.types import StageFunction
from mujodico.trajopt._src.types import StateFunction


class IppasTrajOpt:

  def __init__(
      self,
      N: int,
      nx: int,
      nu: int,
      *,
      initial_state: Callable[[Params], jax.Array],
      dynamics: StageFunction,
      stage_cost: StageFunction,
      terminal_cost: StateFunction,
      stage_equality_constraint: StageFunction | None = None,
      stage_inequality_constraint: StageFunction | None = None,
      stage_inequality_bounds: Bounds | None = None,
      terminal_inequality_constraint: StateFunction | None = None,
      terminal_inequality_bounds: Bounds | None = None,
      x_bounds: Bounds | None = None,
      u_bounds: Bounds | None = None,
      quaternion_positions: tuple[int, ...] = (),
      gauss_newton: bool = True,
      full_dynamics_jacobian: bool = False,
      params_trace: Params | None = None,
      options: IppasOptions | None = None,
  ):
    self._model = StagewiseOcpModel(
        N,
        nx,
        nu,
        initial_state=initial_state,
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        stage_equality_constraint=stage_equality_constraint,
        stage_inequality_constraint=stage_inequality_constraint,
        stage_inequality_bounds=stage_inequality_bounds,
        terminal_inequality_constraint=terminal_inequality_constraint,
        terminal_inequality_bounds=terminal_inequality_bounds,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        quaternion_positions=quaternion_positions,
        gauss_newton=gauss_newton,
        full_dynamics_jacobian=full_dynamics_jacobian,
        params_trace=params_trace,
    )
    self._solver = IppasSolver(self._model, options)

  def solve(self, params, xs_init, us_init, *, warm_start=None, **options):
    solver = self._solver
    if options:
      solver = IppasSolver(self._model, replace(solver.options, **options))

    result = solver.solve(params, IppasInitialGuess(xs_init, us_init, warm_start))

    info = {
        "status": result.status,
        "success": result.success,
        "obj_val": result.objective,
        "iter_count": result.iterations,
        "primal_error": result.primal_error,
        "dual_error": result.dual_error,
        "complementarity": result.complementarity,
        "error": result.error,
        "mu": result.mu,
        "warm_start": result.iterate,
    }
    return result.xs, result.us, info
