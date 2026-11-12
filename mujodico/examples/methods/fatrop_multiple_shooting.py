import numpy as np

from mujodico import trajopt
from mujodico.examples import methods
from mujodico.examples.tasks import Task


class FatropMultipleShooting:

  def __init__(self, task: Task):
    self.transcription = methods.MultipleShootingTranscription(task)
    trans = self.transcription
    self._history = []

    def update_params(stats, params):
      self._history.append({
          'iter': stats['iter'],
          'xs': np.asarray(stats['xs']).copy(),
          'us': np.asarray(stats['us']).copy(),
      })
      return params

    self.solver = trajopt.FatropTrajOpt(
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
        params_trace=trans.params,
        update_params=update_params,
    )

  def update(self, task: Task):
    self.transcription.update(task)

  def solve(
      self, traj_init: dict, *, iterations: int = 500, verbose: bool = True
  ):
    trans = self.transcription
    self._history = []
    xs_opt, us_opt, info = self.solver.solve(
        trans.params,
        traj_init['xs'],
        traj_init['us'],
        max_iter=iterations,
        print_level=5 if verbose else 0,
        suppress_banner=not verbose,
    )

    total_time = info['total_time']
    component_time = {
        'callback': info['callback_time'],
        'conversion': info['conversion_time'],
    }
    component_time['fatrop'] = total_time - sum(component_time.values())
    print(
        f'Method: {self.__class__.__name__}\n'
        f'  solver status: {info["status"]}\n'
        f'  total time: {total_time:.6f} sec\n'
        + '\n'.join([
            f'    {name} time: {time:.6f} sec / {time/total_time*100:.2f}%'
            for name, time in component_time.items()
        ])
    )
    return {
        'xs': xs_opt,
        'us': us_opt,
        'history': self._history,
    }
