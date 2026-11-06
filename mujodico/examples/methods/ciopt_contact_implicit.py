from abc import ABC, abstractmethod
from enum import Enum
import importlib
import time
import jax
import numpy as np
from mujodico.examples import methods
from mujodico.examples.tasks import Task
from mujodico.trajopt._src.ccopt_trajopt import CCOptTrajOpt


class CIOptTrajOpt(CCOptTrajOpt):

  class Method(Enum):
    IMPACT = 'impact'

  def __init__(
      self,
      N: int,
      nx: int,
      nu: int,
      *,
      local_cost,
      terminal_cost,
      initial_equality_constraint,
      local_equality_constraint,
      local_inequality_constraint,
      local_complementarity_constraint,
      x_bounds,
      u_bounds,
      gauss_newton: bool,
      params_trace,
  ):
    try:
      juliacall = importlib.import_module('juliacall')
      self._jl = juliacall.Main
    except ImportError as e:
      raise ImportError(
          'Optional Julia bridge dependency not installed. Install juliacall with\n\n'
          '    pip install juliacall'
      ) from e

    super().__init__(
        N=N,
        nx=nx,
        nu=nu,
        local_cost=local_cost,
        terminal_cost=terminal_cost,
        initial_equality_constraint=initial_equality_constraint,
        local_equality_constraint=local_equality_constraint,
        local_inequality_constraint=local_inequality_constraint,
        local_complementarity_constraint=local_complementarity_constraint,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        gauss_newton=gauss_newton,
        params_trace=params_trace,
    )

  def solve(
      self,
      params,
      xs_init: jax.Array | np.ndarray,
      us_init: jax.Array | np.ndarray,
      method: Method,
      **kwargs,
  ) -> tuple[np.ndarray, np.ndarray, dict]:
    if not self._has_complementarity_constraint:
      return super().solve(
          params=params,
          xs_init=xs_init,
          us_init=us_init,
          **kwargs,
      )

    nlp = self._nlp
    nlp.set_params(params)

    vars_init = nlp._pack(np.asarray(xs_init), np.asarray(us_init))

    jl = self._jl
    jl_nlp = nlp.to_nlpmodels_jl(vars_init)

    self._ensure_jl_packages(('CIOpt', 'MPCCModels'))
    mpcc = self._make_jl_mpcc(jl_nlp, convert_vertical_form=False)

    if method == self.Method.IMPACT:
      solver = jl.CIOpt.ImpactSolver(mpcc, **kwargs)
      tic = time.perf_counter()
      stats = getattr(jl.CIOpt, 'solve!')(solver)

    total_time = time.perf_counter() - tic
    info = {'total_time': total_time}
    info.update(nlp.stats)
    for name in ['objective', 'status', 'iter']:
      try:
        info[name] = getattr(stats, name)
      except AttributeError:
        pass

    vars_ops = np.asarray(stats.z)
    xs_opt, us_opt = nlp._unpack(vars_ops)

    return np.asarray(xs_opt), np.asarray(us_opt), info


class _CioptContactImplicit(ABC):

  def __init__(self, task: Task, method: CIOptTrajOpt.Method):
    self.transcription = methods.ContactImplicitTranscription(task)
    self.method = method

    trans = self.transcription
    self.solver = CIOptTrajOpt(
        N=trans.N,
        nx=trans.nx,
        nu=trans.nw,
        local_cost=lambda x, w, xnext, k, params: trans.stage_cost(x, w, params),
        terminal_cost=trans.terminal_cost,
        initial_equality_constraint=lambda x, params: x - params.x_0,
        local_equality_constraint=lambda x, w, x_next, k, params: (
            x_next - trans.dynamics(x, w)
        ),
        local_inequality_constraint=lambda x, w, x_next, k, params: (
            trans.local_inequality(x, w, x_next)
        ),
        local_complementarity_constraint=trans.local_complementarity,
        x_bounds=trans.x_bounds,
        u_bounds=trans.w_bounds,
        gauss_newton=True,
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
        **self._convert_kwargs(iterations=iterations, verbose=verbose),
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

  @abstractmethod
  def _convert_kwargs(self, **kwargs) -> dict:
    pass


class CioptImpactContactImplicit(_CioptContactImplicit):
  def __init__(self, task: Task):
    super().__init__(
        task=task,
        method=CIOptTrajOpt.Method.IMPACT,
    )

  def _convert_kwargs(self, **kwargs) -> dict:
    return {
        'max_total_inner_iters': kwargs['iterations'],
        'print_level': 1 if kwargs['verbose'] else 0,
    }
