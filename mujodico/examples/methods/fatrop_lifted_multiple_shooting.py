import numpy as np

from mujodico import trajopt
from mujodico.examples import methods
from mujodico.examples.tasks import Task


class FatropLiftedMultipleShooting:

  def __init__(self, task: Task):
    self.transcription = methods.LiftedMultipleShootingTranscription(task)
    trans = self.transcription
    self._history = []

    def update_params(stats, params):
      ws = np.asarray(stats['us'])
      self._history.append({
          'iter': stats['iter'],
          'xs': np.asarray(stats['xs']).copy(),
          'us': ws[:, : task.nu].copy(),
          'cfs': ws[:, task.nu :].copy(),
      })
      return params

    self.solver = trajopt.FatropTrajOpt(
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
        update_params=update_params,
    )

  def update(self, task: Task):
    self.transcription.update(task)

  def solve(
      self, traj_init: dict, *, iterations: int = 500, verbose: bool = True
  ):
    trans = self.transcription
    us_init = np.asarray(traj_init['us'])
    nu = us_init.shape[-1]
    cfs_init = traj_init.get('cfs', np.zeros((trans.N, trans.nw - nu)))
    ws_init = np.hstack([us_init, cfs_init])
    self._history = []
    xs_opt, ws_opt, info = self.solver.solve(
        trans.params,
        traj_init['xs'],
        ws_init,
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
        'us': ws_opt[:, :nu],
        'cfs': ws_opt[:, nu:],
        'history': self._history,
    }
