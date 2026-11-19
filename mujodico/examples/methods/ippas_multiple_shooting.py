import time

import jax
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
        stage_inequality_constraint=(
            (lambda x, u, k, params: trans.stage_inequality(x, u))
            if trans.stage_inequality is not None else None
        ),
        x_bounds=trans.x_bounds,
        u_bounds=trans.u_bounds,
        quaternion_addrs=task.quaternion_addrs,
        gauss_newton=gauss_newton,
        params_trace=trans.params,
    )

    self._solver_cache = {}

  def update(self, task: Task):
    self.transcription.update(task)

  def solve(
      self, traj_init: dict, *, iterations: int = 500, verbose: bool = True
  ):
    trans = self.transcription
    xs = traj_init['xs']
    us = traj_init['us']

    args = jax.device_put((trans.params, xs, us))
    key = (iterations,)
    if key not in self._solver_cache:
      compiled = (
          jax.jit(
              lambda p, x, u: self.solver.solve(p, x, u, max_iter=iterations)
          )
          .lower(*args)
          .compile()
      )
      # Exclude compilation and first-use setup from the measured solve.
      jax.block_until_ready(compiled(*args))
      self._solver_cache[key] = compiled

    jax.block_until_ready(args)
    started = time.perf_counter()
    xs, us, info = jax.block_until_ready(self._solver_cache[key](*args))
    solve_time = time.perf_counter() - started

    if verbose:
      status = trajopt.IppasStatus(info['status'])
      print(
          f'Method: {self.__class__.__name__}\n'
          f'  solve time: {solve_time:.6f} sec\n'
          f'  solver status: {status.name}\n'
          f'  iterations: {int(info["iter_count"])}\n'
      )
    return {
        'xs': np.asarray(xs),
        'us': np.asarray(us),
        'solve_time': solve_time,
    }
