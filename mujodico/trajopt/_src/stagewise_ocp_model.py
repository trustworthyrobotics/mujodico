from collections.abc import Callable
from numbers import Integral

import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.types import Bounds, Params, StageFunction, StateFunction


class StagewiseOcpModel:
  """An explicit OCP, independent of a solver or solver callback layout.

  Inequality evaluators return raw bounded values. Their effective bounds
  include bounded state variables, bounded controls (non-termina only), then
  user inequalities. Jacobians use output-by-input layout and separate x/u blocks.
  """

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
      gauss_newton: bool = True,
      params_trace: Params | None = None,
  ):
    for name, value, minimum in (('N', N, 0), ('nx', nx, 1), ('nu', nu, 0)):
      if (
          not isinstance(value, Integral)
          or isinstance(value, bool)
          or value < minimum
      ):
        raise ValueError(f'{name} must be an integer >= {minimum}.')

    self.N, self.nx, self.nu = int(N), int(nx), int(nu)
    self.x_bounds = self._bounds(x_bounds, nx, 'x_bounds')
    self.u_bounds = self._bounds(u_bounds, nu, 'u_bounds')

    self._user_initial_state_fn = initial_state
    self._user_dynamics_fn = dynamics
    self._user_stage_cost_fn = stage_cost
    self._user_terminal_cost_fn = terminal_cost
    self._user_stage_eq_fn = stage_equality_constraint
    self._user_stage_ineq_fn = stage_inequality_constraint
    self._user_terminal_eq_fn = terminal_equality_constraint
    self._user_terminal_ineq_fn = terminal_inequality_constraint
    self._user_stage_bounds = stage_inequality_bounds
    self._user_terminal_bounds = terminal_inequality_bounds
    self._gauss_newton = gauss_newton

    self._initialized = False
    self._evaluators = {}

    if params_trace is not None:
      self.initialize(params_trace)

  def initialize(self, params_trace: Params) -> None:
    if self._initialized:
      return
    evaluators, shapes, stage_bounds, terminal_bounds, sizes = (
        self._build_evaluators(params_trace)
    )
    self._evaluators, self._input_specs = evaluators, shapes
    self._stage_bounds, self._terminal_bounds = stage_bounds, terminal_bounds
    self._constraint_sizes = sizes
    self._initialized = True

  def _require_initialized(self):
    if not self._initialized:
      raise RuntimeError(
          'Pass params_trace or call initialize(params_trace) before accessing metadata.'
      )

  @staticmethod
  def _bounds(bounds, size, name, *, inequality=False):
    if bounds is None:
      lower = np.zeros(size) if inequality else np.full(size, -np.inf)
      return lower, np.full(size, np.inf)
    if len(bounds) != 2:
      raise ValueError(f'{name} must contain lower and upper arrays.')
    lower, upper = (np.asarray(v, dtype=float).copy() for v in bounds)
    if lower.shape != (size,) or upper.shape != (size,):
      raise ValueError(f'{name} must contain arrays of shape ({size},).')
    if (
        np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower >= upper)
    ):
      raise ValueError(f'{name} requires lower < upper and no NaN entries.')
    return lower, upper

  @property
  def evaluation_dtype(self):
    self._require_initialized()
    return self._input_specs['objective'][0][1]

  @property
  def constraint_sizes(self) -> tuple[int, int, int, int]:
    """Stage equality/inequality and terminal equality/inequality counts."""
    self._require_initialized()
    return self._constraint_sizes

  @property
  def stage_inequality_bounds(self) -> Bounds:
    self._require_initialized()
    return tuple(v.copy() for v in self._stage_bounds)

  @property
  def terminal_inequality_bounds(self) -> Bounds:
    self._require_initialized()
    return tuple(v.copy() for v in self._terminal_bounds)

  def _build_evaluators(self, params):
    N, nx, nu = self.N, self.nx, self.nu
    x, u, k = jp.zeros(nx), jp.zeros(nu), jp.array(0)

    def check(fn, args, expected, name):
      shape = jax.eval_shape(fn, *args).shape
      if expected is None:
        if len(shape) == 1:
          return shape[0]
      elif shape == expected:
        return
      raise ValueError(
          f'{name} returned shape {shape}; expected'
          f' {expected if expected is not None else "a vector"}.'
      )

    check(self._user_initial_state_fn, (params,), (nx,), 'initial_state')
    check(self._user_terminal_cost_fn, (x, params), (), 'terminal_cost')
    if N:
      check(self._user_dynamics_fn, (x, u, k, params), (nx,), 'dynamics')
      check(self._user_stage_cost_fn, (x, u, k, params), (), 'stage_cost')

    def empty_stage(x, u, k, p):
      return jp.zeros(0)

    def empty_state(x, p):
      return jp.zeros(0)

    stage_eq = (
        self._user_stage_eq_fn
        if self._user_stage_eq_fn is not None
        else empty_stage
    )
    user_stage_ineq = (
        self._user_stage_ineq_fn
        if self._user_stage_ineq_fn is not None
        else empty_stage
    )
    terminal_eq = (
        self._user_terminal_eq_fn
        if self._user_terminal_eq_fn is not None
        else empty_state
    )
    user_terminal_ineq = (
        self._user_terminal_ineq_fn
        if self._user_terminal_ineq_fn is not None
        else empty_state
    )

    ng = check(stage_eq, (x, u, k, params), None, 'stage equality')
    user_nh = check(
        user_stage_ineq, (x, u, k, params), None, 'stage inequality'
    )
    ng_terminal = check(terminal_eq, (x, params), None, 'terminal equality')
    user_nh_terminal = check(
        user_terminal_ineq, (x, params), None, 'terminal inequality'
    )

    user_stage_bounds = self._bounds(
        self._user_stage_bounds,
        user_nh,
        'stage_inequality_bounds',
        inequality=True,
    )
    user_terminal_bounds = self._bounds(
        self._user_terminal_bounds,
        user_nh_terminal,
        'terminal_inequality_bounds',
        inequality=True,
    )

    xi = np.flatnonzero(
        np.isfinite(self.x_bounds[0]) | np.isfinite(self.x_bounds[1])
    )
    ui = np.flatnonzero(
        np.isfinite(self.u_bounds[0]) | np.isfinite(self.u_bounds[1])
    )

    stage_bounds = tuple(
        np.concatenate(
            [self.x_bounds[i][xi], self.u_bounds[i][ui], user_stage_bounds[i]]
        )
        for i in (0, 1)
    )
    terminal_bounds = tuple(
        np.concatenate([self.x_bounds[i][xi], user_terminal_bounds[i]])
        for i in (0, 1)
    )
    nh, nh_terminal = stage_bounds[0].size, terminal_bounds[0].size

    def stage_ineq(x, u, k, p):
      return jp.concatenate([x[xi], u[ui], user_stage_ineq(x, u, k, p)])

    def terminal_ineq(x, p):
      return jp.concatenate([x[xi], user_terminal_ineq(x, p)])

    ks = jp.arange(N)

    def batch(fn, shape):
      def evaluate(xs, us, p):
        if not N:
          if shape and isinstance(shape[0], tuple):
            return tuple(jp.zeros((0, *block)) for block in shape)
          return jp.zeros((0, *shape))
        return jax.vmap(fn, in_axes=(0, 0, 0, None))(xs, us, ks, p)

      return evaluate

    stage_costs = batch(self._user_stage_cost_fn, ())

    def objective(xs, us, p):
      return jp.concatenate([
          stage_costs(xs[:-1], us, p),
          jp.asarray(self._user_terminal_cost_fn(xs[-1], p))[None],
      ])

    def objective_gradient(xs, us, p):
      return jax.grad(
          lambda xs, us: jp.sum(objective(xs, us, p)), argnums=(0, 1)
      )(xs, us)

    def dynamics_residual(xs, us, xs_next, p):
      return batch(self._user_dynamics_fn, (nx,))(xs, us, p) - xs_next

    def dynamics_jacobian(xs, us, xs_next, p):
      return batch(
          jax.jacfwd(self._user_dynamics_fn, argnums=(0, 1)),
          ((nx, nx), (nx, nu)),
      )(xs, us, p)

    def initial_residual(x, p):
      return x - self._user_initial_state_fn(p)

    def stage_lagrangian(x, u, xnext, k, ld, le, li, scale, p):
      value = scale * self._user_stage_cost_fn(x, u, k, p)
      if not self._gauss_newton:
        value += jp.vdot(ld, self._user_dynamics_fn(x, u, k, p) - xnext)
        value += jp.vdot(le, stage_eq(x, u, k, p))
        value += jp.vdot(li, stage_ineq(x, u, k, p))
      return value

    def terminal_lagrangian(x, le, li, scale, p):
      value = scale * self._user_terminal_cost_fn(x, p)
      if not self._gauss_newton:
        value += jp.vdot(le, terminal_eq(x, p))
        value += jp.vdot(li, terminal_ineq(x, p))
      return value

    def lagrangian_hessian(xs, us, lam_init, ld, le, li, lte, lti, scale, p):
      if N:
        blocks = jax.vmap(
            jax.hessian(stage_lagrangian, argnums=(0, 1)),
            in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
        )(xs[:-1], us, xs[1:], ks, ld, le, li, scale, p)
        hxx, hxu, huu = blocks[0][0], blocks[0][1], blocks[1][1]
      else:
        hxx, hxu, huu = (
            jp.zeros((0, nx, nx)),
            jp.zeros((0, nx, nu)),
            jp.zeros((0, nu, nu)),
        )
      terminal = jax.hessian(terminal_lagrangian)(xs[-1], lte, lti, scale, p)
      hxx = jp.concatenate([hxx, terminal[None]])
      if not self._gauss_newton:
        initial = jax.hessian(
            lambda x: jp.vdot(lam_init, initial_residual(x, p))
        )(xs[0])
        hxx = hxx.at[0].add(initial)
      return hxx, hxu, huu

    xs, us = jp.zeros((N + 1, nx)), jp.zeros((N, nu))
    shapes = {}
    functions = {}

    def add(name, fn, *args):
      shapes[name] = tuple((arg.shape, arg.dtype) for arg in args)
      functions[name] = fn

    add('objective', objective, xs, us)
    add('objective_gradient', objective_gradient, xs, us)
    add('dynamics_residual', dynamics_residual, xs[:-1], us, xs[1:])
    add(
        'dynamics_jacobian',
        dynamics_jacobian,
        xs[:-1],
        us,
        xs[1:],
    )
    for name, fn, size in (
        ('stage_equality', stage_eq, ng),
        ('stage_inequality', stage_ineq, nh),
    ):
      add(name + '_residual', batch(fn, (size,)), xs[:-1], us)
      add(
          name + '_jacobian',
          batch(jax.jacfwd(fn, argnums=(0, 1)), ((size, nx), (size, nu))),
          xs[:-1],
          us,
      )
    for name, fn in (
        ('terminal_equality', terminal_eq),
        ('terminal_inequality', terminal_ineq),
    ):
      add(name + '_residual', fn, x)
      add(name + '_jacobian', jax.jacfwd(fn), x)
    add('initial_state', self._user_initial_state_fn)
    add('initial_residual', initial_residual, x)
    add('initial_jacobian', jax.jacfwd(initial_residual), x)
    add(
        'lagrangian_hessian',
        lagrangian_hessian,
        xs,
        us,
        jp.zeros((nx,)),
        jp.zeros((N, nx)),
        jp.zeros((N, ng)),
        jp.zeros((N, nh)),
        jp.zeros(ng_terminal),
        jp.zeros(nh_terminal),
        jp.asarray(1.0),
    )
    return (
        functions,
        shapes,
        stage_bounds,
        terminal_bounds,
        (ng, nh, ng_terminal, nh_terminal),
    )

  def _evaluate(self, name, *args):
    params, args = args[-1], args[:-1]
    self.initialize(params)
    # Use the trace dtypes, including for integer-valued array inputs.
    specs = self._input_specs[name]
    args = tuple(
        jp.asarray(v, dtype=dtype) for v, (_, dtype) in zip(args, specs)
    )
    for value, (shape, _) in zip(args, specs):
      if value.shape != shape:
        raise ValueError(f'{name}: expected shape {shape}, got {value.shape}.')
    return self._evaluators[name](*args, params)

  def objective(self, xs, us, params):
    """Return stage and terminal costs, shape (N+1,)."""
    return self._evaluate('objective', xs, us, params)

  def objective_gradient(self, xs, us, params):
    """Return (gx, gu) with shapes (N+1, nx) and (N, nu)."""
    return self._evaluate('objective_gradient', xs, us, params)

  def lagrangian_hessian(
      self,
      xs,
      us,
      lam_init,
      lam_dyn,
      lam_stage_eq,
      lam_stage_ineq,
      lam_terminal_eq,
      lam_terminal_ineq,
      objective_factor,
      params,
  ):
    """Return (Hxx, Hxu, Huu), retaining the terminal block only in Hxx.

    Multiplier shapes are (nx,), (N, nx), (N, ng), (N, nh), (ng_terminal,),
    and (nh_terminal,). Inequality counts include bounded variables. Blocks
    have shapes (N+1, nx, nx), (N, nx, nu), and (N, nu, nu).
    """
    return self._evaluate(
        'lagrangian_hessian',
        xs,
        us,
        lam_init,
        lam_dyn,
        lam_stage_eq,
        lam_stage_ineq,
        lam_terminal_eq,
        lam_terminal_ineq,
        objective_factor,
        params,
    )

  def dynamics_residual(self, xs, us, xs_next, params):
    return self._evaluate('dynamics_residual', xs, us, xs_next, params)

  def dynamics_jacobian(self, xs, us, xs_next, params):
    """Return residual derivatives (A, B) shaped (N, nx, nx), (N, nx, nu)."""
    return self._evaluate('dynamics_jacobian', xs, us, xs_next, params)

  def stage_equality_residual(self, xs, us, params):
    return self._evaluate('stage_equality_residual', xs, us, params)

  def stage_equality_jacobian(self, xs, us, params):
    return self._evaluate('stage_equality_jacobian', xs, us, params)

  def stage_inequality_residual(self, xs, us, params):
    return self._evaluate('stage_inequality_residual', xs, us, params)

  def stage_inequality_jacobian(self, xs, us, params):
    return self._evaluate('stage_inequality_jacobian', xs, us, params)

  def terminal_equality_residual(self, x, params):
    return self._evaluate('terminal_equality_residual', x, params)

  def terminal_equality_jacobian(self, x, params):
    return self._evaluate('terminal_equality_jacobian', x, params)

  def terminal_inequality_residual(self, x, params):
    return self._evaluate('terminal_inequality_residual', x, params)

  def terminal_inequality_jacobian(self, x, params):
    return self._evaluate('terminal_inequality_jacobian', x, params)

  def initial_state(self, params):
    return self._evaluate('initial_state', params)

  def initial_residual(self, x0, params):
    return self._evaluate('initial_residual', x0, params)

  def initial_jacobian(self, x0, params):
    return self._evaluate('initial_jacobian', x0, params)

  def retract_state(self, x, dx, alpha, params):
    return x + alpha * dx

  def retract_control(self, u, du, alpha, params):
    return u + alpha * du
