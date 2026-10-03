import time
import importlib
import logging
import jax
import jax.numpy as jp
from mujodico import mjx
from mujodico.examples import methods
from mujodico.examples.methods import support
from mujodico.examples.tasks import Task
import numpy as np


class IpoptSingleShooting:

  def __init__(
      self,
      task: Task,
  ):
    try:
      self._cyipopt = importlib.import_module('cyipopt')
      logging.getLogger('cyipopt').setLevel(logging.WARNING)
    except ImportError as e:
      raise ImportError(
          'Optional solver dependency not installed. Install it with:\n\n'
          '    pip install cyipopt'
      ) from e

    self.transcription = methods.SingleShootingTranscription(task)
    trans = self.transcription

    N = trans.N
    nu = trans.nu

    def objective(vars, params):
      us = vars.reshape(N, nu)
      return trans.objective(us, params)

    vars_trace = jp.zeros((N, nu)).ravel()
    params_trace = trans.params

    objective_grad = jax.jit(jax.grad(objective)).lower(vars_trace, params_trace).compile()
    objective = jax.jit(objective).lower(vars_trace, params_trace).compile()

    self._eval_time = [0] * 2

    def timed(index):
      def decorate(fn):
        def wrapped(*args, **kwargs):
          tic = time.perf_counter()
          out = fn(*args, **kwargs)
          self._eval_time[index] += time.perf_counter() - tic
          return out
        return wrapped
      return decorate

    class NLP:
      def __init__(self, params):
        self.params = params

      @timed(index=0)
      def objective(self, vars):
        return float(objective(jp.asarray(vars), self.params))

      @timed(index=1)
      def gradient(self, vars):
        return np.asarray(objective_grad(jp.asarray(vars), self.params))

    self._NLP = NLP
    self.update(task)

  def update(self, task: Task):
    self.transcription.update(task)
    self.task = task
    self._refresh_problem()

  def _refresh_problem(self):
    trans = self.transcription
    N = trans.N

    u_lb, u_ub = trans.u_bounds
    us_lb = np.repeat(np.asarray(u_lb)[None, :], N, axis=0)
    us_ub = np.repeat(np.asarray(u_ub)[None, :], N, axis=0)

    self.problem = self._cyipopt.Problem(
        n=us_lb.size,
        m=0,
        problem_obj=self._NLP(trans.params),
        lb=us_lb.ravel(),
        ub=us_ub.ravel(),
    )

  def solve(
      self,
      traj_init: dict,
      *,
      iterations: int = 500,
      verbose: bool = True,
  ):
    us_init = traj_init['us']

    N, nu = self.transcription.N, self.transcription.nu
    assert us_init.shape == (N, nu)

    vars_init = us_init.ravel()

    self.problem.add_option('max_iter', iterations)
    if not verbose:
      self.problem.add_option('print_level', 0)

    self._eval_time = [0] * 2
    tic = time.perf_counter()

    us_opt, info = self.problem.solve(vars_init)

    total_time = time.perf_counter() - tic

    print(f'''
Method: {self.__class__.__name__}
  total time: {total_time:.6f} sec
    dynamics time: {self._eval_time[0]:.6f} sec / {self._eval_time[0]/total_time*100:.2f}%
    gradient time: {self._eval_time[1]:.6f} sec / {self._eval_time[1]/total_time*100:.2f}%
    ipopt time:    {total_time-sum(self._eval_time):.6f} sec / {(1-sum(self._eval_time)/total_time)*100:.2f}%
''')

    us_opt = jp.array(us_opt).reshape(N, nu)

    xs_rollout = self.transcription.rollout(
        self.transcription.params.x_0, us_opt
    )

    return {
        'us': us_opt,
        'xs': xs_rollout,
    }
