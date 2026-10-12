import numpy as np
import jax.numpy as jp
from mujodico import trajopt
from mujodico.examples import methods
from mujodico.examples.tasks import Task


class IpoptRelaxationContactImplicit:

  def __init__(self, task: Task):
    self.transcription = methods.ContactImplicitTranscription(task)
    trans = self.transcription

    side_offsets = {
        trajopt.LocalComplementarityConstraint.Side.X: 0,
        trajopt.LocalComplementarityConstraint.Side.U: trans.nx,
        trajopt.LocalComplementarityConstraint.Side.XNEXT: trans.nx + trans.nw,
        trajopt.LocalComplementarityConstraint.Side.INEQ: 2 * trans.nx + trans.nw,
    }
    complementarity_indices = np.asarray([
        [side_offsets[side] + index for side, index in pair]
        for pair in trans.local_complementarity.pairs
    ])

    def local_equality_constraint(x, w, xnext, params):
      ineq = trans.local_inequality(x, w, xnext)
      local_variables = jp.concatenate((x, w, xnext, ineq))
      complementarity_terms = local_variables[complementarity_indices]
      complementarity = complementarity_terms[:, 0] * complementarity_terms[:, 1]

      mu = task.contact_smoothing
      return jp.concatenate((
          xnext - trans.dynamics(x, w),
          complementarity - mu,
      ))

    def local_inequality_constraint(x, w, xnext, params):
      return trans.local_inequality(x, w, xnext)

    self.solver = trajopt.IpoptTrajOpt(
        N=trans.N,
        nx=trans.nx,
        nu=trans.nw,
        stage_cost=trans.stage_cost,
        terminal_cost=trans.terminal_cost,
        initial_equality_constraint=lambda x, params: x - params.x_0,
        local_equality_constraint=local_equality_constraint,
        local_inequality_constraint=local_inequality_constraint,
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
    }
