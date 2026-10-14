import numpy as np
from mujodico import trajopt
from mujodico.examples import methods
from mujodico.examples.tasks import Task


class _CcoptContactImplicit:

  def __init__(self, task: Task, method: trajopt.CCOptTrajOpt.Method):
    self.transcription = methods.ContactImplicitTranscription(task)
    self.method = method

    trans = self.transcription
    self.solver = trajopt.CCOptTrajOpt(
        N=trans.N,
        nx=trans.nx,
        nu=trans.nw,
        stage_cost=trans.stage_cost,
        terminal_cost=trans.terminal_cost,
        initial_equality_constraint=lambda x, params: x - params.x_0,
        local_equality_constraint=lambda x, w, x_next, params: (
            x_next - trans.dynamics(x, w)
        ),
        local_inequality_constraint=lambda x, w, x_next, params: (
            trans.local_inequality(x, w, x_next)
        ),
        local_complementarity_constraint=trans.local_complementarity,
        x_bounds=trans.x_bounds,
        u_bounds=trans.w_bounds,
        params_trace=trans.params,
    )

  def update(self, task: Task):
    self.transcription.update(task)

  def solve(
      self,
      traj_init: dict,
      *,
      iterations: int = 500,
      verbose: bool = True,
  ):
    trans = self.transcription
    xs_init = traj_init['xs']
    us_init = traj_init['us']
    N = self.transcription.N
    nw = self.transcription.nw
    nu = us_init.shape[-1]
    cfs_init = traj_init.get('cfs', np.zeros((N, nw - nu)))
    ws_init = np.hstack([us_init, cfs_init])

    xs_opt, ws_opt, info = self.solver.solve(
        trans.params,
        xs_init,
        ws_init,
        method=self.method,
        max_iter=iterations,
        print_level=3 if verbose else 5,
    )

    total_time = info['total_time']
    component_time = {
      'callback': info['callback_time'],
      'conversion': info['conversion_time'],
    }
    component_time['ccopt'] = total_time - sum(component_time.values())
    print(
        f'Method: {self.__class__.__name__}\n'
        f'  status: {info['status']}\n'
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
        'info': info,
    }


class CcoptRelaxationContactImplicit(_CcoptContactImplicit):
  def __init__(self, task: Task):
    super().__init__(
        task=task,
        method=trajopt.CCOptTrajOpt.Method.RELAXATION,
    )


class CcoptPenaltyContactImplicit(_CcoptContactImplicit):
  def __init__(self, task: Task):
    super().__init__(
        task=task,
        method=trajopt.CCOptTrajOpt.Method.PENALTY,
    )


class CcoptCrossoverContactImplicit(_CcoptContactImplicit):
  def __init__(self, task: Task):
    super().__init__(
        task=task,
        method=trajopt.CCOptTrajOpt.Method.CROSSOVER,
    )
