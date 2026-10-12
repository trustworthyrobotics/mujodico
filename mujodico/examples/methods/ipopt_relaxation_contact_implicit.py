import numpy as np
import jax.numpy as jp
from mujodico import trajopt
from mujodico.examples import methods
from mujodico.examples.tasks import Task


class IpoptRelaxationContactImplicit:

  def __init__(self, task: Task):
    self.transcription = methods.ContactImplicitTranscription(task)
    trans = self.transcription

    self.initial_smoothing = jp.array(task.contact_smoothing)

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

      smoothing = params[1]
      return jp.concatenate((
          xnext - trans.dynamics(x, w),
          complementarity - smoothing,
      ))

    def local_inequality_constraint(x, w, xnext, params):
      return trans.local_inequality(x, w, xnext)

    def update_params(stats, params):
      # TODO(wei-chen): Find better homotopy update.
      smoothing = params[1]
      iter = stats['iter']
      if iter % 100 == 0 and iter != 0:
        smoothing = jp.maximum(smoothing * 0.1, 1e-8)
      return params[0], smoothing

    self.solver = trajopt.IpoptTrajOpt(
        N=trans.N,
        nx=trans.nx,
        nu=trans.nw,
        stage_cost=lambda x, w, params: trans.stage_cost(x, w, params[0]),
        terminal_cost=lambda x, params: trans.terminal_cost(x, params[0]),
        initial_equality_constraint=lambda x, params: x - params[0].x_0,
        local_equality_constraint=local_equality_constraint,
        local_inequality_constraint=local_inequality_constraint,
        x_bounds=trans.x_bounds,
        u_bounds=trans.w_bounds,
        params_trace=(self.transcription.params, self.initial_smoothing),
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

    xs_opt, ws_opt, info = self.solver.solve(
        (self.transcription.params, self.initial_smoothing),
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
