from mujodico import trajopt
from mujodico.examples import methods
from mujodico.examples.tasks import Task


class IpoptMultipleShooting:

  def __init__(self, task: Task):
    self.transcription = methods.MultipleShootingTranscription(task)

    trans = self.transcription
    self.solver = trajopt.IpoptMultipleShootingSolver(
        trans.N,
        trans.nx,
        trans.nu,
        stage_cost=trans.stage_cost,
        terminal_cost=trans.terminal_cost,
        initial_equality_constraint=lambda x, params: x - params.x_0,
        local_equality_constraint=lambda x, u, xnext, params: (
            xnext - trans.dynamics(x, u, params)
        ),
        x_bounds=trans.x_bounds,
        u_bounds=trans.u_bounds,
        params_trace=self.transcription.params,
    )

  def update(self, task: Task):
    self.transcription.update(task)

  def solve(self, traj_init: dict, *, iterations: int = 500, verbose: bool = True):
    xs_opt, us_opt, info = self.solver.solve(
        self.transcription.params,
        traj_init['xs'],
        traj_init['us'],
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

    return {'xs': xs_opt, 'us': us_opt}
