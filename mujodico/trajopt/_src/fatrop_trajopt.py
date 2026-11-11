"""JAX trajectory optimization through cyfatrop's batched interface."""

import importlib
import time
from collections.abc import Callable

import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.types import (
  Bounds,
  Params,
  TerminalCost,
  TerminalEqualityConstraint,
  TerminalInequalityConstraint,
)

StageFunction = Callable[[jax.Array, jax.Array, jax.Array, Params], jax.Array]


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
      terminal_cost: TerminalCost,
      stage_equality_constraint: StageFunction | None = None,
      stage_inequality_constraint: StageFunction | None = None,
      terminal_equality_constraint: TerminalEqualityConstraint | None = None,
      terminal_inequality_constraint: TerminalInequalityConstraint | None = None,
      x_bounds: Bounds | None = None,
      u_bounds: Bounds | None = None,
      gauss_newton: bool = True,
      params_trace: Params | None = None,
      update_params: Callable[[dict, Params], Params] | None = None,
  ):
    try:
      self._cyfatrop = importlib.import_module('cyfatrop')
    except (ImportError, OSError) as e:
      raise ImportError(
          'FatropTrajOpt requires the optional cyfatrop package. '
          'Install it with:\n\n'
          '    pip install git+https://github.com/wei-chen-li/cyfatrop.git'
      ) from e

    self.N, self.nx, self.nu = int(N), int(nx), int(nu)
    self.update_params = update_params
    self._initial_state = initial_state
    self._dynamics = dynamics
    self._stage_cost = stage_cost
    self._terminal_cost = terminal_cost
    self._stage_eq = stage_equality_constraint
    self._stage_ineq = stage_inequality_constraint
    self._terminal_eq = terminal_equality_constraint
    self._terminal_ineq = terminal_inequality_constraint
    self._x_bounds = self._check_bounds(x_bounds, nx, 'x_bounds')
    self._u_bounds = self._check_bounds(u_bounds, nu, 'u_bounds')
    self._gauss_newton = gauss_newton
    self._signature = None
    self._callback_time = self._conversion_time = 0.0

    # Keep cyfatrop optional at module import time, while still satisfying
    # its requirement that the problem inherit exactly one public ABC.
    class Problem(_FatropProblem, self._cyfatrop.TrajOptAbstract):
      pass

    self._problem = Problem(self)
    if params_trace is not None:
      self._set_params(params_trace)

  @staticmethod
  def _check_bounds(bounds, size, name):
    if bounds is None:
      return np.full(size, -np.inf), np.full(size, np.inf)
    lower, upper = (np.asarray(v, dtype=float).copy() for v in bounds)
    if lower.shape != (size,) or upper.shape != (size,):
      raise ValueError(f'{name} must contain arrays of shape ({size},).')
    if (
        np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower >= upper)
    ):
      raise ValueError(
          f'{name} must have lower < upper and no NaN entries; '
          'express fixed variables through equality constraints.'
      )
    return lower, upper

  def _set_params(self, params):
    x0 = np.asarray(self._initial_state(params), dtype=float)
    if x0.shape != (self.nx,) or not np.all(np.isfinite(x0)):
      raise ValueError(
          f'initial_state(params) must return a finite ({self.nx},) array.'
      )
    leaves, treedef = jax.tree_util.tree_flatten(params)
    signature = treedef, tuple(
        (np.shape(v), np.asarray(v).dtype.str) for v in leaves
    )
    self._params, self._x0 = params, x0.copy()
    if signature != self._signature:
      self._compile(params)
      self._signature = signature

  def _compile(self, params_trace):
    N, nx, nu = self.N, self.nx, self.nu
    x_trace, u_trace, k_trace = jp.zeros(nx), jp.zeros(nu), jp.array(0)
    x_indices = np.flatnonzero(
        np.isfinite(self._x_bounds[0]) | np.isfinite(self._x_bounds[1])
    )
    u_indices = np.flatnonzero(
        np.isfinite(self._u_bounds[0]) | np.isfinite(self._u_bounds[1])
    )

    def count(fn, args, name):
      if fn is None:
        return 0
      shape = jax.eval_shape(fn, *args).shape
      if len(shape) != 1:
        raise ValueError(f'{name} must return a one-dimensional array.')
      return shape[0]

    stage_args = x_trace, u_trace, k_trace, params_trace
    terminal_args = x_trace, params_trace
    ng = count(self._stage_eq, stage_args, 'stage equality') if N else 0
    nh = count(self._stage_ineq, stage_args, 'stage inequality') if N else 0
    ng_terminal = count(self._terminal_eq, terminal_args, 'terminal equality')
    nh_terminal = count(
        self._terminal_ineq, terminal_args, 'terminal inequality'
    )
    ng_ineq = nh + x_indices.size + u_indices.size
    ng_ineq_terminal = nh_terminal + x_indices.size
    if not N:
      ng, ng_ineq = ng_terminal, ng_ineq_terminal
    if ng_terminal > ng or ng_ineq_terminal > ng_ineq:
      raise ValueError(
          'cyfatrop.TrajOptAbstract requires terminal constraint counts '
          'not to exceed stage counts (including variable bounds).'
      )
    if (N and ng > nu) or ng_terminal > (nx if N else 0):
      raise ValueError(
          'Too many equalities for cyfatrop.TrajOptAbstract: stage '
          'equalities plus the fixed initial state must fit stage variables; '
          'terminal equalities must fit terminal state variables.'
      )
    if jax.eval_shape(self._terminal_cost, *terminal_args).shape != ():
      raise ValueError('terminal_cost must return a scalar.')
    if N:
      if jax.eval_shape(self._stage_cost, *stage_args).shape != ():
        raise ValueError('stage_cost must return a scalar.')
      if jax.eval_shape(self._dynamics, *stage_args).shape != (nx,):
        raise ValueError(f'dynamics must return an array of shape ({nx},).')

    self._ng, self._ng_terminal = ng, ng_terminal
    self._ng_ineq, self._ng_ineq_terminal = ng_ineq, ng_ineq_terminal
    stage_lower = np.concatenate([
        np.zeros(nh),
        self._x_bounds[0][x_indices],
        self._u_bounds[0][u_indices],
    ])
    stage_upper = np.concatenate([
        np.full(nh, np.inf),
        self._x_bounds[1][x_indices],
        self._u_bounds[1][u_indices],
    ])
    terminal_lower = np.concatenate(
        [np.zeros(nh_terminal), self._x_bounds[0][x_indices]]
    )
    terminal_upper = np.concatenate(
        [np.full(nh_terminal, np.inf), self._x_bounds[1][x_indices]]
    )
    self._bounds = tuple(
        np.concatenate([
            np.broadcast_to(stage, (N, ng_ineq))
            if N
            else np.empty((0, ng_ineq)),
            np.pad(terminal, (0, ng_ineq - ng_ineq_terminal))[None],
        ])
        for stage, terminal in (
            (stage_lower, terminal_lower),
            (stage_upper, terminal_upper),
        )
    )

    def stage_cost(y, k, p):
      return self._stage_cost(y[nu:], y[:nu], k, p)

    def terminal_cost(y, k, p):
      return self._terminal_cost(y[nu:], p)

    def dynamics(y, k, p):
      return self._dynamics(y[nu:], y[:nu], k, p)

    def stage_eq(y, k, p):
      return (
          jp.zeros(0)
          if self._stage_eq is None
          else self._stage_eq(y[nu:], y[:nu], k, p)
      )

    def terminal_eq(y, k, p):
      value = (
          jp.zeros(0)
          if self._terminal_eq is None
          else self._terminal_eq(y[nu:], p)
      )
      return jp.pad(value, (0, ng - ng_terminal))

    def stage_ineq(y, k, p):
      value = (
          jp.zeros(0)
          if self._stage_ineq is None
          else self._stage_ineq(y[nu:], y[:nu], k, p)
      )
      return jp.concatenate([value, y[nu:][x_indices], y[:nu][u_indices]])

    def terminal_ineq(y, k, p):
      value = (
          jp.zeros(0)
          if self._terminal_ineq is None
          else self._terminal_ineq(y[nu:], p)
      )
      value = jp.concatenate([value, y[nu:][x_indices]])
      return jp.pad(value, (0, ng_ineq - ng_ineq_terminal))

    ks = jp.arange(N)

    def batch(stage, terminal):
      def evaluate(u, x, p):
        ys = jp.concatenate([u, x[:-1]], axis=1)
        y_terminal = jp.concatenate([jp.zeros(nu), x[-1]])
        last = terminal(y_terminal, jp.array(N), p)
        first = (
            jax.vmap(stage, in_axes=(0, 0, None))(ys, ks, p)
            if N
            else jp.zeros((0, *last.shape))
        )
        return jp.concatenate([first, last[None]], axis=0)

      return evaluate

    def jacobian_transpose(fn):
      return lambda *args: jax.jacfwd(fn)(*args).T

    def batched_b(x_next, u, x, p):
      if not N:
        return jp.zeros((0, nx))
      ys = jp.concatenate([u, x], axis=1)
      return jax.vmap(dynamics, in_axes=(0, 0, None))(ys, ks, p) - x_next

    def batched_BAt(x_next, u, x, p):
      if not N:
        return jp.zeros((0, nu + nx, nx))
      ys = jp.concatenate([u, x], axis=1)
      return jax.vmap(jacobian_transpose(dynamics), in_axes=(0, 0, None))(
          ys, ks, p
      )

    def stage_lagrangian(y, ld, le, li, k, scale, p):
      value = scale * stage_cost(y, k, p)
      if not self._gauss_newton:
        value += jp.vdot(ld, dynamics(y, k, p))
        value += jp.vdot(le, stage_eq(y, k, p))
        value += jp.vdot(li, stage_ineq(y, k, p))
      return value

    def terminal_lagrangian(y, le, li, scale, p):
      value = scale * terminal_cost(y, jp.array(N), p)
      if not self._gauss_newton:
        value += jp.vdot(le, terminal_eq(y, jp.array(N), p))
        value += jp.vdot(li, terminal_ineq(y, jp.array(N), p))
      return value

    def batched_RSQt(u, x, scale, ld, le, li, p):
      ys = jp.concatenate([u, x[:-1]], axis=1)
      y_terminal = jp.concatenate([jp.zeros(nu), x[-1]])
      last = jax.hessian(terminal_lagrangian)(
          y_terminal, le[-1], li[-1], scale, p
      )
      first = (
          jax.vmap(
              jax.hessian(stage_lagrangian), in_axes=(0, 0, 0, 0, 0, None, None)
          )(
              ys,
              ld,
              le[:-1],
              li[:-1],
              ks,
              scale,
              p,
          )
          if N
          else jp.zeros((0, nu + nx, nu + nx))
      )
      return jp.concatenate([first, last[None]], axis=0)

    u_batch, x_batch = jp.zeros((N, nu)), jp.zeros((N + 1, nx))
    self._callbacks = {}
    for name, fn in (
        ('L', batch(stage_cost, terminal_cost)),
        ('rq', batch(jax.grad(stage_cost), jax.grad(terminal_cost))),
        ('g', batch(stage_eq, terminal_eq)),
        (
            'Gt',
            batch(
                jacobian_transpose(stage_eq), jacobian_transpose(terminal_eq)
            ),
        ),
        ('gineq', batch(stage_ineq, terminal_ineq)),
        (
            'Gt_ineq',
            batch(
                jacobian_transpose(stage_ineq),
                jacobian_transpose(terminal_ineq),
            ),
        ),
    ):
      self._callbacks[name] = (
          jax.jit(fn).lower(u_batch, x_batch, params_trace).compile()
      )
    for name, fn in (('b', batched_b), ('BAt', batched_BAt)):
      self._callbacks[name] = (
          jax.jit(fn)
          .lower(x_batch[1:], u_batch, x_batch[:-1], params_trace)
          .compile()
      )
    self._callbacks['RSQt'] = (
        jax.jit(batched_RSQt)
        .lower(
            u_batch,
            x_batch,
            jp.array(1.0),
            jp.zeros((N, nx)),
            jp.zeros((N + 1, ng)),
            jp.zeros((N + 1, ng_ineq)),
            params_trace,
        )
        .compile()
    )

  def _evaluate(self, name, *args):
    tic = time.perf_counter()
    jax_args = tuple(jp.asarray(arg) for arg in args)
    jax.block_until_ready(jax_args)
    self._conversion_time += time.perf_counter() - tic
    tic = time.perf_counter()
    result = self._callbacks[name](*jax_args, self._params)
    result.block_until_ready()
    self._callback_time += time.perf_counter() - tic
    tic = time.perf_counter()
    result = np.asarray(result, dtype=float)
    self._conversion_time += time.perf_counter() - tic
    return result

  def solve(
      self,
      params: Params,
      xs_init: jax.Array | np.ndarray,
      us_init: jax.Array | np.ndarray,
      **options,
  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (xs, us, info), forwarding options to cyfatrop.Solver.

    Shapes are (N+1, nx) for xs and (N, nu) for us. Compilation is excluded
    from total_time, as in the other trajectory optimizer wrappers.
    """
    xs_init, us_init = np.asarray(xs_init, dtype=float), np.asarray(
        us_init, dtype=float
    )
    if xs_init.shape != (self.N + 1, self.nx):
      raise ValueError(f'xs_init must have shape ({self.N + 1}, {self.nx}).')
    if us_init.shape != (self.N, self.nu):
      raise ValueError(f'us_init must have shape ({self.N}, {self.nu}).')
    self._set_params(params)
    self._xs_init = xs_init.copy()
    self._us_init = us_init.copy()
    self._callback_time = self._conversion_time = 0.0
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
        callback_time=self._callback_time,
        conversion_time=self._conversion_time,
    )
    return np.asarray(result.x), np.asarray(result.u), info


class _FatropProblem:
  """Implementation mixed with TrajOptAbstract after the optional import."""

  def __init__(self, optimizer):
    self.opt = optimizer

  def get_horizon_length(self):
    return self.opt.N + 1

  def get_nx(self):
    return self.opt.nx

  def get_nu(self):
    return self.opt.nu

  def get_ng(self):
    return self.opt._ng

  def get_ng_terminal(self):
    return self.opt._ng_terminal

  def get_ng_ineq(self):
    return self.opt._ng_ineq

  def get_ng_ineq_terminal(self):
    return self.opt._ng_ineq_terminal

  def batched_get_bounds(self):
    return self.opt._bounds

  def batched_get_initial_xk(self):
    return self.opt._xs_init

  def batched_get_initial_uk(self):
    return self.opt._us_init

  def eval_g_initial(self, x0):
    return np.asarray(x0) - self.opt._x0

  def eval_Gt_initial(self, x0):
    return np.eye(self.opt.nx)

  def batched_eval_b(self, x_next, u, x):
    return self.opt._evaluate('b', x_next, u, x)

  def batched_eval_BAt(self, x_next, u, x):
    return self.opt._evaluate('BAt', x_next, u, x)

  def batched_eval_L(self, scale, u, x):
    return scale * self.opt._evaluate('L', u, x)

  def batched_eval_rq(self, scale, u, x):
    return scale * self.opt._evaluate('rq', u, x)

  def batched_eval_g(self, u, x):
    return self.opt._evaluate('g', u, x)

  def batched_eval_Gt(self, u, x):
    return self.opt._evaluate('Gt', u, x)

  def batched_eval_gineq(self, u, x):
    return self.opt._evaluate('gineq', u, x)

  def batched_eval_Gt_ineq(self, u, x):
    return self.opt._evaluate('Gt_ineq', u, x)

  def batched_eval_RSQt(
      self, scale, u, x, lam_dyn, lam_eq, lam_ineq, lam_initial
  ):
    return self.opt._evaluate('RSQt', u, x, scale, lam_dyn, lam_eq, lam_ineq)

  def intermediate(self, iteration, x, u, stats):
    if self.opt.update_params is None:
      return
    stats = dict(stats, iter=iteration, xs=x, us=u)
    params = self.opt.update_params(stats, self.opt._params)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    signature = treedef, tuple(
        (np.shape(v), np.asarray(v).dtype.str) for v in leaves
    )
    if signature != self.opt._signature:
      raise ValueError(
          'update_params must preserve parameter tree structure, shapes, and'
          ' dtypes.'
      )
    self.opt._set_params(params)
