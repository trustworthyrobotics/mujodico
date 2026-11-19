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
        quaternion_addrs=task.quaternion_addrs,
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

    args = jax.device_put((trans.params, xs, ws))
    key = (iterations,)
    if key not in self._solver_cache:
      compiled = (
          jax.jit(
              lambda p, x, w: self.solver.solve(
                  p, x, w, max_iter=iterations, record_intermediates=True
              )
          )
          .lower(*args)
          .compile()
      )
      # Exclude compilation and first-use setup from the measured solve.
      jax.block_until_ready(compiled(*args))
      self._solver_cache[key] = compiled

    jax.block_until_ready(args)
    started = time.perf_counter()
    xs, ws, info = jax.block_until_ready(self._solver_cache[key](*args))
    solve_time = time.perf_counter() - started

    count = int(info['iter_count']) + 1
    xs_history = np.asarray(info['intermediates'].x[:count])
    ws_history = np.asarray(info['intermediates'].u[:count])
    history = [
        {'iter': i, 'xs': x, 'us': w[:, :nu], 'cfs': w[:, nu:]}
        for i, (x, w) in enumerate(zip(xs_history, ws_history))
    ]

    status = trajopt.IppasStatus(int(info['status'])).name
    if verbose:
      print(
          f'Method: {self.__class__.__name__}\n'
          f'  solve time: {solve_time:.6f} sec\n'
          f'  solver status: {status}\n'
          f'  iterations: {int(info["iter_count"])}\n'
      )
    return {
        'xs': np.asarray(xs),
        'us': np.asarray(ws[:, :nu]),
        'cfs': np.asarray(ws[:, nu:]),
        'history': history,
        'solve_time': solve_time,
        'status': status,
    }
