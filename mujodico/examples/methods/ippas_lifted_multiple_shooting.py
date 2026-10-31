"""Compiled IPPAS method exposed through the examples runner."""

import time

import jax
import jax.numpy as jp
import numpy as np

from mujodico.examples.methods.transcription_lifted_multiple_shooting import (
    LiftedMultipleShootingTranscription,
)
from mujodico.trajopt._src.fatrop_ref import InitialGuess, IppasSolver, Options
from mujodico.trajopt._src.fatrop_ref.manifold import quaternion_product


class IppasLiftedMultipleShooting:

  def __init__(self, task, *, gauss_newton: bool = True):
    # TODO(wei-chen): Remove this smoothing change after IPPAS is stable.
    object.__setattr__(task, "contact_smoothing", task.contact_smoothing * 10)
    self.transcription = LiftedMultipleShootingTranscription(task)
    # Preserve the example's historical flag: ``cost`` is a cost-only Hessian,
    # not a claim that arbitrary scalar objectives provide GN residuals.
    self.hessian_mode = 'cost' if gauss_newton else 'exact_lagrangian'
    self.state_manifold = quaternion_product(
        self.transcription.nx, task.quaternion_w_idxs
    )
    self._solver_cache = {}
    self.solver = None

  def _make_solver(self, options):
    trans = self.transcription

    def dynamics(x, w, params):
      del params
      return trans.dynamics(x, w)

    def equality(x, w, params):
      del params
      return trans.stage_equality(x, w)

    def inequality(x, w, params):
      del params
      return trans.stage_inequality(x, w)

    return IppasSolver(
        horizon=trans.N,
        num_states=trans.nx,
        num_controls=trans.nw,
        initial_state=lambda params: params.x_0,
        stage_cost=trans.stage_cost,
        terminal_cost=trans.terminal_cost,
        dynamics=dynamics,
        stage_equality=equality,
        stage_inequality=inequality,
        stage_inequality_bounds=(0.0, np.inf),
        x_bounds=trans.x_bounds,
        u_bounds=trans.w_bounds,
        state_manifold=self.state_manifold,
        options=options,
    )

  def solve(
      self,
      traj_init: dict,
      *,
      iterations: int,
      stagewise: bool = False,
      gpu_line_search: bool = False,
      max_soc: int = 4,
  ):
    """Return ``xs/us/cfs`` using the existing example-method interface.

    ``gpu_line_search`` is retained as a compatibility option: both values use
    compiled device-side globalization in this implementation.
    """
    trans = self.transcription
    xs_init = jp.asarray(traj_init['xs'])
    us_init = jp.asarray(traj_init['us'], dtype=xs_init.dtype)
    nu = us_init.shape[-1]
    if nu > trans.nw:
      raise ValueError(
          'Initial physical control dimension exceeds lifted control size'
      )
    cfs_init = jp.asarray(
        traj_init.get('cfs', np.zeros((trans.N, trans.nw - nu))),
        dtype=xs_init.dtype,
    )
    ws_init = jp.concatenate((us_init, cfs_init), axis=1)
    options = Options(
        max_iter=iterations,
        linear_backend='parallel',
        hessian_mode=self.hessian_mode,
        stagewise=stagewise,
        use_gpu_line_search=gpu_line_search,
        max_soc=max_soc,
    )
    if options not in self._solver_cache:
      solver = self._make_solver(options)
      self._solver_cache[options] = solver, jax.jit(solver.solve)
    self.solver, compiled_solve = self._solver_cache[options]

    started = time.perf_counter()
    result = compiled_solve(trans.params, InitialGuess(xs_init, ws_init))
    jax.block_until_ready(result.xs)
    elapsed = time.perf_counter() - started
    print(
        f'\nMethod: {self.__class__.__name__}\n'
        f'  total time: {elapsed:.6f} sec\n'
        f'  solver status: {int(result.status)}\n'
        f'  iterations: {int(result.iterations)}\n'
    )
    return {
        'xs': np.asarray(result.xs),
        'us': np.asarray(result.us[:, :nu]),
        'cfs': np.asarray(result.us[:, nu:]),
    }
