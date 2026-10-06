import numpy as np
import jax.numpy as jp
from mujodico import trajopt
from mujodico.examples import methods
from mujodico.examples.tasks import Task


class IpoptLiftedMultipleShooting:

  def __init__(self, task: Task):
    self.transcription = methods.LiftedMultipleShootingTranscription(task)

    trans = self.transcription
    self.solver = trajopt.IpoptMultipleShootingSolver(
        N=trans.N,
        nx=trans.nx,
        nu=trans.nw,
        stage_cost=trans.stage_cost,
        terminal_cost=trans.terminal_cost,
        initial_equality_constraint=lambda x, params: x - params.x_0,
        local_equality_constraint=lambda x, w, xnext, params: (
            jp.concatenate([
                xnext - trans.dynamics(x, w, params),
                trans.stage_equality_constraint(x, w, params),
            ])
        ),
        local_inequality_constraint=lambda x, w, xnext, params: (
            trans.stage_inequality_constraint(x, w, params)
        ),
        x_bounds=trans.x_bounds,
        u_bounds=trans.w_bounds,
        params_trace=self.transcription.params,
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

    xs_opt, ws_opt, info = self.solver.solve(
        self.transcription.params,
        xs_init,
        ws_init,
        max_iter=iterations,
        print_level=5 if verbose else 0,
    )

    total_time = info['total_time_sec']
    eval_time = info['eval_time_sec']
    print(f'''
Method: {self.__class__.__name__}
  total time: {total_time:.6f} sec
    dynamics time: {eval_time:.6f} sec / {eval_time/total_time*100:.2f}%
    ipopt time:    {total_time-eval_time:.6f} sec / {(1-eval_time/total_time)*100:.2f}%
''')

    return {
        'xs': xs_opt,
        'us': ws_opt[:, :nu],
        'cfs': ws_opt[:, nu:],
    }
