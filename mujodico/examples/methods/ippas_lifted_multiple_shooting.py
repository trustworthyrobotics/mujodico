import time

import jax
import jax.numpy as jp
import numpy as np

from mujodico import trajopt
from mujodico.examples import methods


class IppasLiftedMultipleShooting:

  def __init__(self, task):
    self.transcription = methods.LiftedMultipleShootingTranscription(task)
    trans = self.transcription

    self.solver = trajopt.IppasTrajOpt(
        N=trans.N,
        nx=trans.nx,
        nu=trans.nw,
        initial_state=lambda params: params.x_0,
        dynamics=lambda x, w, k, params: trans.dynamics(x, w),
        stage_cost=lambda x, w, k, params: trans.stage_cost(x, w, params),
        terminal_cost=trans.terminal_cost,
        stage_equality_constraint=lambda x, w, k, params: (
            trans.stage_equality(x, w)
        ),
        stage_inequality_constraint=lambda x, w, k, params: (
            trans.stage_inequality(x, w)
        ),
        x_bounds=trans.x_bounds,
        u_bounds=trans.w_bounds,
        quaternion_positions=task.quaternion_w_idxs,
        params_trace=trans.params,
    )
    self._solver_cache = {}

  def update(self, task):
    self.transcription.update(task)

  def solve(self, traj_init: dict, *, iterations: int = 500, verbose: bool = True):
    trans = self.transcription
    xs = traj_init['xs']
    us = traj_init['us']
    nu = us.shape[-1]
    cfs = traj_init.get('cfs', np.zeros((trans.N, trans.nw - nu)))
    ws = jp.concatenate((us, cfs), axis=1)

    key = (iterations,)
    if key not in self._solver_cache:
      self._solver_cache[key] = jax.jit(
          lambda p, x, w: self.solver.solve(p, x, w, max_iter=iterations)
      )

    started = time.perf_counter()
    xs, ws, info = self._solver_cache[key](trans.params, xs, ws)
    jax.block_until_ready((xs, ws, info))

    if verbose:
      print(
          f'Method: {self.__class__.__name__}\n'
          f'  total time: {time.perf_counter() - started:.6f} sec\n'
          f'  solver status: {int(info["status"])}\n'
          f'  iterations: {int(info["iter_count"])}\n'
      )
    return {
        'xs': np.asarray(xs),
        'us': np.asarray(ws[:, :nu]),
        'cfs': np.asarray(ws[:, nu:]),
    }
