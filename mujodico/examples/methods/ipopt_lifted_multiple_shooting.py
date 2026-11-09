import numpy as np
import jax.numpy as jp
from mujodico import trajopt
from mujodico.examples import methods
from mujodico.examples.tasks import Task


class IpoptLiftedMultipleShooting:

  def __init__(self, task: Task):
    self.transcription = methods.LiftedMultipleShootingTranscription(task)

    trans = self.transcription
    self._history = []

    def update_params(stats, params):
      xs, ws = self.solver._nlp._unpack(stats['x'])
      self._history.append({
          'iter': stats['iter'],
          'xs': xs,
          'us': ws[:, :task.nu],
          'cfs': ws[:, task.nu:],
      })
      return params

    self.solver = trajopt.IPOptTrajOpt(
        N=trans.N,
        nx=trans.nx,
        nu=trans.nw,
        local_cost=lambda x, w, xnext, k, params: trans.stage_cost(x, w, params),
        terminal_cost=trans.terminal_cost,
        initial_equality_constraint=lambda x, params: x - params.x_0,
        local_equality_constraint=lambda x, w, xnext, k, params: (
            jp.concatenate([
                xnext - trans.dynamics(x, w),
                trans.stage_equality(x, w),
            ])
        ),
        local_inequality_constraint=lambda x, w, xnext, k, params: (
            trans.stage_inequality(x, w)
        ),
        x_bounds=trans.x_bounds,
        u_bounds=trans.w_bounds,
        params_trace=self.transcription.params,
        update_params=update_params,
    )

  def update(self, task: Task):
    self.transcription.update(task)

  def solve(self, traj_init: dict, *, iterations: int = 500, verbose: bool = True):
    xs_init = traj_init['xs']
    us_init = traj_init['us']
    N = self.transcription.N
    nw = self.transcription.nw
    nu = us_init.shape[-1]
    cfs_init = traj_init.get('cfs', np.zeros((N, nw - nu)))
    ws_init = np.hstack([us_init, cfs_init])
    self._history = []

    xs_opt, ws_opt, info = self.solver.solve(
        self.transcription.params,
        xs_init,
        ws_init,
        max_iter=iterations,
        print_level=5 if verbose else 0,
    )

    total_time = info['total_time']
    component_time = {
      'callback': info['callback_time'],
      'conversion': info['conversion_time'],
    }
    component_time['ipopt'] = total_time - sum(component_time.values())
    print(
        f'Method: {self.__class__.__name__}\n'
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
