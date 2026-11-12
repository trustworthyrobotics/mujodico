import importlib
import time
from collections.abc import Callable

import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel
from mujodico.trajopt._src.types import (
    Bounds,
    Params,
    StageFunction,
    StateFunction,
)


class FatropTrajOpt:

  def __init__(
      self,
      N: int,
      nx: int,
      nu: int,
      *,
      initial_state: Callable[[Params], jax.Array],
      dynamics: StageFunction,
      stage_cost: StageFunction,
      terminal_cost: StateFunction,
      stage_equality_constraint: StageFunction | None = None,
      stage_inequality_constraint: StageFunction | None = None,
      stage_inequality_bounds: Bounds | None = None,
      terminal_equality_constraint: StateFunction | None = None,
      terminal_inequality_constraint: StateFunction | None = None,
      terminal_inequality_bounds: Bounds | None = None,
      x_bounds: Bounds | None = None,
      u_bounds: Bounds | None = None,
      quaternion_positions: tuple[int, ...] = (),
      gauss_newton: bool = True,
      params_trace: Params | None = None,
      update_params: Callable[[dict, Params], Params] | None = None,
  ):
    try:
      self._cyfatrop = importlib.import_module('cyfatrop')
    except (ImportError, OSError) as e:
      raise ImportError(
          'FatropTrajOpt requires the optional cyfatrop package. Install it'
          ' with:\n\n    pip install'
          ' git+https://github.com/wei-chen-li/cyfatrop.git'
      ) from e

    self._model = StagewiseOcpModel(
        N,
        nx,
        nu,
        initial_state=initial_state,
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        stage_equality_constraint=stage_equality_constraint,
        stage_inequality_constraint=stage_inequality_constraint,
        stage_inequality_bounds=stage_inequality_bounds,
        terminal_equality_constraint=terminal_equality_constraint,
        terminal_inequality_constraint=terminal_inequality_constraint,
        terminal_inequality_bounds=terminal_inequality_bounds,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        quaternion_positions=quaternion_positions,
        gauss_newton=gauss_newton,
    )

    class Problem(_FatropProblem, self._cyfatrop.TrajOptAbstract):
      pass

    self._problem = Problem(
        self._model, params_trace=params_trace, update_params=update_params
    )

  def solve(
      self,
      params: Params,
      xs_init: jax.Array | np.ndarray,
      us_init: jax.Array | np.ndarray,
      **options,
  ) -> tuple[np.ndarray, np.ndarray, dict]:
    self._problem.prepare(xs_init, us_init, params)
    self._problem.reset_stats()

    solver = self._cyfatrop.Solver(self._problem, options=options)

    tic = time.perf_counter()
    result = solver.solve()
    total_time = time.perf_counter() - tic

    info = dict(result.stats)
    info.update(
        status=result.status,
        success=result.success,
        obj_val=result.objective,
        total_time=total_time,
        **self._problem.stats,
    )
    return np.asarray(result.x), np.asarray(result.u), info


class _FatropProblem:

  def __init__(
      self, model: StagewiseOcpModel, *, params_trace=None, update_params=None
  ):
    self.model = model
    self.update_params = update_params
    self.reset_stats()
    self._compiled_signature = None
    if params_trace is not None:
      self._initialize(params_trace)

  def reset_stats(self):
    self._callback_time = self._conversion_time = 0.0

  @property
  def stats(self):
    return {
        'callback_time': self._callback_time,
        'conversion_time': self._conversion_time,
    }

  @staticmethod
  def _parameter_signature(params):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    avals = tuple(jax.typeof(v) for v in leaves)
    return treedef, tuple((v.shape, v.dtype, v.weak_type) for v in avals)

  def prepare(self, xs_init, us_init, params):
    m = self.model

    xs, us = np.asarray(xs_init, dtype=float), np.asarray(us_init, dtype=float)
    if xs.shape != (m.N + 1, m.nx):
      raise ValueError(f'xs_init must have shape ({m.N + 1}, {m.nx}).')
    if us.shape != (m.N, m.nu):
      raise ValueError(f'us_init must have shape ({m.N}, {m.nu}).')
    for i in m.quaternion_positions:
      quaternions = xs[:, i:i+4]
      if (
        not np.all(np.isfinite(quaternions)) or
        not np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, rtol=0, atol=1e-6)
      ):
        raise ValueError(
            f'xs_init quaternion block at index {i} must contain finite unit'
            ' quaternions (norm tolerance 1e-6).'
        )

    self._xs_init, self._us_init = xs.copy(), us.copy()

    if self._compiled_signature is None:
      self._initialize(params)
      return

    signature = self._parameter_signature(params)
    if signature != self._compiled_signature:
      raise ValueError(
          'params must match the params_trace tree, shapes, dtypes, and weak'
          ' types; construct a new FatropTrajOpt for a different signature.'
      )
    self._params = params

  def _initialize(self, params):
    m = self.model
    m.initialize(params)

    self._params = params
    self._params_signature = self._parameter_signature(params)
    if self._compiled_signature == self._params_signature:
      return

    self._stage_ng, self._stage_nh, self._ng_terminal, self._nh_terminal = m.constraint_sizes
    self._ng = self._stage_ng if m.N else self._ng_terminal
    self._nh = self._stage_nh if m.N else self._nh_terminal

    if self._ng_terminal > self._ng or self._nh_terminal > self._nh:
      raise ValueError(
          'cyfatrop requires terminal constraint counts not to exceed stage counts.'
      )

    self._compile_callbacks()
    self._compiled_signature = self._params_signature

    self._bounds = tuple(
        np.concatenate([
            np.broadcast_to(stage, (m.N, self._nh))
            if m.N
            else np.empty((0, self._nh)),
            np.pad(terminal, (0, self._nh - self._nh_terminal))[None],
        ])
        for stage, terminal in zip(
            m.stage_inequality_bounds, m.terminal_inequality_bounds
        )
    )

  def _compile_callbacks(self):
    m = self.model
    N, nx, nu, tx = m.N, m.nx, m.nu, m.nx_tangent

    def objective(u, x, scale, p):
      return scale * m.objective(x, u, p)

    def gradient(u, x, scale, p):
      gx, gu = m.objective_gradient(x, u, p)
      return scale * jp.concatenate(
          [jp.concatenate([gu, jp.zeros((1, nu))]), gx], axis=1
      )

    def dynamics_jacobian(x_next, u, x, p):
      A, B = m.dynamics_jacobian(x, u, x_next, p)
      return jp.concatenate([B, A], axis=-1).swapaxes(-1, -2)

    def constraint(kind, width, terminal_count, jacobian=False):
      suffix = '_jacobian' if jacobian else '_residual'
      stage_fn = getattr(m, 'stage_' + kind + suffix)
      terminal_fn = getattr(m, 'terminal_' + kind + suffix)

      def evaluate(u, x, p):
        stage = stage_fn(x[:-1], u, p)
        terminal = terminal_fn(x[-1], p)
        if jacobian:
          jx, ju = stage
          stage = jp.concatenate([ju, jx], axis=-1).swapaxes(-1, -2)
          terminal = jp.pad(terminal.T, ((nu, 0), (0, width - terminal_count)))
        else:
          terminal = jp.pad(terminal, (0, width - terminal_count))
        if not N:
          return terminal[None]
        return jp.concatenate([stage, terminal[None]], axis=0)

      return evaluate

    def hessian(scale, u, x, ld, le, li, initial, p):
      stage_eq = le[:-1] if N else jp.zeros((0, self._stage_ng))
      stage_ineq = li[:-1] if N else jp.zeros((0, self._stage_nh))
      hxx, hxu, huu = m.lagrangian_hessian(
          x,
          u,
          initial,
          ld,
          stage_eq,
          stage_ineq,
          le[-1, : self._ng_terminal],
          li[-1, : self._nh_terminal],
          scale,
          p,
      )
      huu = jp.concatenate([huu, jp.zeros((1, nu, nu))])
      hxu = jp.concatenate([hxu, jp.zeros((1, tx, nu))])
      return jp.concatenate(
          [
              jp.concatenate([huu, hxu.swapaxes(-1, -2)], axis=2),
              jp.concatenate([hxu, hxx], axis=2),
          ],
          axis=1,
      )

    dtype = m.evaluation_dtype
    x, u = jp.zeros((N + 1, nx), dtype), jp.zeros((N, nu), dtype)
    for i in m.quaternion_positions:
      x = x.at[:, i].set(1.0)
    scale = jp.asarray(1.0, dtype=dtype)

    self._callbacks = {}
    self._callback_dtypes = {}

    def add(name, fn, *args):
      self._callbacks[name] = jax.jit(fn).lower(*args, self._params).compile()
      self._callback_dtypes[name] = tuple(arg.dtype for arg in args)

    add('g_initial', m.initial_residual, x[0])
    add('Gt_initial', lambda x, p: m.initial_jacobian(x, p).T, x[0])
    add('L', objective, u, x, scale)
    add('rq', gradient, u, x, scale)
    add(
        'b',
        lambda xp, u, x, p: m.dynamics_residual(x, u, xp, p),
        x[1:],
        u,
        x[:-1],
    )
    add('BAt', dynamics_jacobian, x[1:], u, x[:-1])
    for kind, width, terminal_count, names in (
        ('equality', self._ng, self._ng_terminal, ('g', 'Gt')),
        ('inequality', self._nh, self._nh_terminal, ('gineq', 'Gt_ineq')),
    ):
      for jacobian, name in zip((False, True), names):
        add(name, constraint(kind, width, terminal_count, jacobian), u, x)
    add(
        'RSQt',
        hessian,
        scale,
        u,
        x,
        jp.zeros((N, tx), dtype),
        jp.zeros((N + 1, self._ng), dtype),
        jp.zeros((N + 1, self._nh), dtype),
        jp.zeros(tx, dtype),
    )
    add('retract_state', m.retract_state, x, jp.zeros((N+1, tx), dtype), scale)
    add('retract_control', m.retract_control, u, u, scale)

  def _evaluate(self, name, args):
    tic = time.perf_counter()
    args = tuple(
        jp.asarray(arg, dtype=dtype)
        for arg, dtype in zip(args, self._callback_dtypes[name])
    )
    jax.block_until_ready(args)
    self._conversion_time += time.perf_counter() - tic
    tic = time.perf_counter()
    result = self._callbacks[name](*args, self._params)
    jax.block_until_ready(result)
    self._callback_time += time.perf_counter() - tic
    tic = time.perf_counter()
    result = np.asarray(result, dtype=float)
    self._conversion_time += time.perf_counter() - tic
    return result

  def get_horizon_length(self):
    return self.model.N + 1

  def get_nx(self):
    return self.model.nx

  def get_nx_tangent(self):
    return self.model.nx_tangent

  def get_nu(self):
    return self.model.nu

  def get_ng(self):
    return self._ng

  def get_ng_terminal(self):
    return self._ng_terminal

  def get_ng_ineq(self):
    return self._nh

  def get_ng_ineq_terminal(self):
    return self._nh_terminal

  def batched_get_bounds(self):
    return self._bounds

  def batched_get_initial_xk(self):
    return self._xs_init

  def batched_get_initial_uk(self):
    return self._us_init

  def eval_g_initial(self, x0):
    return self._evaluate('g_initial', (x0,))

  def eval_Gt_initial(self, x0):
    return self._evaluate('Gt_initial', (x0,))

  def batched_eval_b(self, x_next, u, x):
    return self._evaluate('b', (x_next, u, x))

  def batched_eval_BAt(self, x_next, u, x):
    return self._evaluate('BAt', (x_next, u, x))

  def batched_eval_L(self, scale, u, x):
    return self._evaluate('L', (u, x, scale))

  def batched_eval_rq(self, scale, u, x):
    return self._evaluate('rq', (u, x, scale))

  def batched_eval_g(self, u, x):
    return self._evaluate('g', (u, x))

  def batched_eval_gineq(self, u, x):
    return self._evaluate('gineq', (u, x))

  def batched_eval_Gt(self, u, x):
    return self._evaluate('Gt', (u, x))

  def batched_eval_Gt_ineq(self, u, x):
    return self._evaluate('Gt_ineq', (u, x))

  def batched_eval_RSQt(
      self, scale, u, x, lam_dyn, lam_eq, lam_ineq, lam_initial
  ):
    return self._evaluate('RSQt', (scale, u, x, lam_dyn, lam_eq, lam_ineq, lam_initial))

  def batched_apply_retraction_xk(self, x, delta_x, alpha):
    return self._evaluate('retract_state', (x, delta_x, alpha))

  def batched_apply_retraction_uk(self, u, delta_u, alpha):
    return self._evaluate('retract_control', (u, delta_u, alpha))

  def intermediate(self, iteration, x, u, stats):
    if self.update_params is None:
      return
    tic = time.perf_counter()
    params = self.update_params(dict(stats, iter=iteration, xs=x, us=u), self._params)
    signature = self._parameter_signature(params)
    if signature != self._params_signature:
      raise ValueError(
          'update_params must preserve parameter tree structure, shapes, and dtypes.'
      )
    jax.block_until_ready(params)
    self._params = params
    self._conversion_time += time.perf_counter() - tic
