import time

import jax
import jax.numpy as jp
import numpy as np

from mujodico import trajopt
from mujodico.examples import methods
from mujodico.examples.tasks import Task


class IppasMultipleShooting:

  def __init__(self, task: Task, *, gauss_newton: bool = True):
    self.transcription = methods.MultipleShootingTranscription(task)
    trans = self.transcription

    self.solver = trajopt.IppasTrajOpt(
        N=trans.N,
        nx=trans.nx,
        nu=trans.nu,
        initial_state=lambda params: params.x_0,
        dynamics=lambda x, u, k, params: trans.dynamics(x, u),
        stage_cost=lambda x, u, k, params: trans.stage_cost(x, u, params),
        terminal_cost=trans.terminal_cost,
        x_bounds=trans.x_bounds,
        u_bounds=trans.u_bounds,
        quaternion_positions=task.quaternion_w_idxs,
        gauss_newton=gauss_newton,
        params_trace=trans.params,
    )

    self._solver_cache = {}

  def update(self, task: Task):
    self.transcription.update(task)

  def solve(self, traj_init: dict, *, iterations: int = 500, verbose: bool = True):
    trans = self.transcription
    xs = traj_init['xs']
    us = traj_init['us']

    key = (iterations,)
    if key not in self._solver_cache:
      self._solver_cache[key] = jax.jit(
          lambda p, x, u: self.solver.solve(p, x, u, max_iter=iterations)
      )

    started = time.perf_counter()
    xs, us, info = self._solver_cache[key](trans.params, xs, us)
    jax.block_until_ready((xs, us, info))

    if verbose:
      print(
          f'Method: {self.__class__.__name__}\n'
          f'  total time: {time.perf_counter() - started:.6f} sec\n'
          f'  solver status: {int(info["status"])}\n'
          f'  iterations: {int(info["iter_count"])}\n'
      )
    return {
        'xs': np.asarray(xs),
        'us': np.asarray(us),
    }
