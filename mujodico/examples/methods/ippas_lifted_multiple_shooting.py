"""JAX/PAS FATROP method exposed through the examples runner."""

import time

import jax
import jax.numpy as jp
import numpy as np

from mujodico.examples import methods
from mujodico.examples.tasks import Task
from mujodico.trajopt._src.fatrop_ref import FatropRefOptions
from mujodico.trajopt._src.fatrop_ref import FatropReferenceJAX


class IppasLiftedMultipleShooting:

  def __init__(self, task: Task, *, gauss_newton: bool = True):
    self.transcription = methods.LiftedMultipleShootingTranscription(task)
    self.hessian_mode = 'cost' if gauss_newton else 'exact_lagrangian'
    self.solver = self._make_solver()

  def _make_solver(self) -> FatropReferenceJAX:
    trans = self.transcription

    n_stage_ineq = trans.stage_inequality(jp.zeros(trans.nx), np.zeros(trans.nw)).size

    x_lb, x_ub = trans.x_bounds
    w_lb, w_ub = trans.w_bounds
    x_lb_idx = jp.asarray(np.flatnonzero(np.isfinite(x_lb)).astype(np.int32))
    x_ub_idx = jp.asarray(np.flatnonzero(np.isfinite(x_ub)).astype(np.int32))
    w_lb_idx = jp.asarray(np.flatnonzero(np.isfinite(w_lb)).astype(np.int32))
    w_ub_idx = jp.asarray(np.flatnonzero(np.isfinite(w_ub)).astype(np.int32))
    n_ineq = x_lb_idx.size + x_ub_idx.size + w_lb_idx.size + w_ub_idx.size + n_stage_ineq

    ineq_lb = np.zeros(n_ineq)
    ineq_ub = np.full(n_ineq, np.inf)

    def stage_inequality_constraint(x, w):
      values = []
      if x_lb_idx.size:
        values.append(x[x_lb_idx] - jp.asarray(x_lb)[x_lb_idx])
      if x_ub_idx.size:
        values.append(jp.asarray(x_ub)[x_ub_idx] - x[x_ub_idx])
      if w_lb_idx.size:
        values.append(w[w_lb_idx] - jp.asarray(w_lb)[w_lb_idx])
      if w_ub_idx.size:
        values.append(jp.asarray(w_ub)[w_ub_idx] - w[w_ub_idx])
      values.append(trans.stage_inequality(x, w))
      return jp.concatenate(values) if values else jp.zeros((0,))

    return FatropReferenceJAX(
        horizon=trans.N,
        num_states=trans.nx,
        num_controls=trans.nw,
        x0=trans.params.x_0,
        stage_cost=lambda x, w: trans.stage_cost(x, w, trans.params),
        terminal_cost=lambda x: trans.terminal_cost(x, trans.params),
        transition=lambda x, w: trans.dynamics(x, w),
        stage_equality=lambda x, w: trans.stage_equality(x, w),
        stage_inequality=stage_inequality_constraint,
        ineq_lb=ineq_lb,
        ineq_ub=ineq_ub,
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
    xs_init = traj_init['xs']
    us_init = traj_init['us']
    trans = self.transcription
    nu = us_init.shape[-1]
    cfs_init = traj_init.get('cfs', np.zeros((trans.N, trans.nw - nu)))
    ws_init = np.concatenate([us_init, cfs_init], axis=1)

    options = FatropRefOptions(
        max_iter=iterations,
        linear_backend='pas',
        hessian_mode=self.hessian_mode,
        enable_watchdog=False,
        max_soc=max_soc,
        use_gpu_line_search=gpu_line_search,
    )

    tic = time.perf_counter()
    result = self.solver.solve(xs_init, ws_init, options, stagewise=stagewise)
    jax.block_until_ready(result.x)
    total_time = time.perf_counter() - tic

    print(f'''
Method: {self.__class__.__name__}
  total time: {total_time:.6f} sec
  solver status: {result.status}
  accepted steps: {result.accepted_steps}
''')

    return {
        'xs': np.asarray(result.x),
        'us': np.asarray(result.u[:, :nu]),
        'cfs': np.asarray(result.u[:, nu:]),
    }
