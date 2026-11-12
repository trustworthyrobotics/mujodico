from collections.abc import Callable
from numbers import Integral

import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src import manifold
from mujodico.trajopt._src.types import Bounds, Params, StageFunction, StateFunction


class StagewiseOcpModel:
  """An explicit OCP, independent of a solver or solver callback layout.

  Inequality evaluators return raw bounded values. Their effective bounds
  include bounded state variables, bounded controls (non-terminal only), then
  user inequalities. Derivatives use tangent coordinates, output-by-input
  layout, and separate state/control blocks. Quaternion inputs and dynamics
  outputs must be unit quaternions in scalar-first order.

  By default dynamics Jacobians hold the next-state chart fixed. Set
  full_dynamics_jacobian=True to differentiate the next-state retraction too
  and solve the resulting implicit linearization.
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
      quaternion_positions: tuple[int, ...] = (),
      gauss_newton: bool = True,
      full_dynamics_jacobian: bool = False,
      params_trace: Params | None = None,
  ):
    for name, value, minimum in (('N', N, 0), ('nx', nx, 1), ('nu', nu, 0)):
      if (
          not isinstance(value, Integral)
          or isinstance(value, bool)
          or value < minimum
      ):
        raise ValueError(f'{name} must be an integer >= {minimum}.')

    if any(
        not isinstance(i, Integral)
        or isinstance(i, bool)
        or i < 0
        or i + 4 > nx
        for i in quaternion_positions
    ):
      raise ValueError(
          'quaternion_positions must contain valid quaternion start indices.'
      )

    quaternion_positions = tuple(sorted(int(i) for i in quaternion_positions))
    if any(b < a + 4 for a, b in zip(quaternion_positions, quaternion_positions[1:])):
      raise ValueError(
          'quaternion_positions must not contain duplicate or overlapping blocks.'
      )

    self.N, self.nx, self.nu = int(N), int(nx), int(nu)
    self.nx_tangent = self.nx - len(quaternion_positions)
    self.quaternion_positions = quaternion_positions
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
    self._full_dynamics_jacobian = full_dynamics_jacobian

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
    N, nx, nu, tx = self.N, self.nx, self.nu, self.nx_tangent

    x, u, k = jp.zeros(nx), jp.zeros(nu), jp.array(0)
    for i in self.quaternion_positions:
      x = x.at[i].set(1.0)

    def stage_derivative(fn, differentiate):
      def evaluate(x, u, *args):
        p = args[-1]

        def local(dx, du):
          return fn(
              self.retract_state(x, dx, 1.0, p),
              self.retract_control(u, du, 1.0, p),
              *args,
          )

        return differentiate(local, argnums=(0, 1))(
            jp.zeros(tx, dtype=x.dtype), jp.zeros_like(u)
        )

      return evaluate

    def state_derivative(fn, differentiate):
      def evaluate(x, *args):
        return differentiate(
            lambda dx: fn(self.retract_state(x, dx, 1.0, args[-1]), *args)
        )(jp.zeros(tx, dtype=x.dtype))

      return evaluate

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
      gx, gu = batch(
          stage_derivative(self._user_stage_cost_fn, jax.grad), ((tx,), (nu,))
      )(xs[:-1], us, p)
      terminal = state_derivative(self._user_terminal_cost_fn, jax.grad)(xs[-1], p)
      return jp.concatenate([gx, terminal[None]]), gu

    def dynamics_local(x, u, xnext, k, p):
      return self._state_difference(self._user_dynamics_fn(x, u, k, p), xnext)

    def dynamics_residual(x, u, xnext, p):
      if not N:
        return jp.zeros((0, tx), dtype=x.dtype)
      return jax.vmap(dynamics_local, in_axes=(0, 0, 0, 0, None))(x, u, xnext, ks, p)

    def dynamics_jacobian(x, u, xnext, p):
      if not N:
        return jp.zeros((0, tx, tx)), jp.zeros((0, tx, nu))

      if not self._full_dynamics_jacobian:
        return jax.vmap(
            stage_derivative(dynamics_local, jax.jacfwd),
            in_axes=(0, 0, 0, 0, None),
        )(x, u, xnext, ks, p)

      def linearize(x, u, xnext, k):
        def residual(dx, du, dxnext):
          return dynamics_local(
              self.retract_state(x, dx, 1.0, p),
              self.retract_control(u, du, 1.0, p),
              self.retract_state(xnext, dxnext, 1.0, p),
              k,
              p,
          )

        jx, ju, jnext = jax.jacfwd(residual, argnums=(0, 1, 2))(
            jp.zeros(tx, dtype=x.dtype),
            jp.zeros_like(u),
            jp.zeros(tx, dtype=xnext.dtype),
        )
        ab = jp.concatenate([jx, ju], axis=1)
        for count, i in enumerate(self.quaternion_positions):
          rows = slice(i - count, i - count + 3)
          ab = ab.at[rows].set(-jp.linalg.solve(jnext[rows, rows], ab[rows]))
        return ab[:, :tx], ab[:, tx:]

      return jax.vmap(linearize, in_axes=(0, 0, 0, 0))(x, u, xnext, ks)

    def initial_residual(x, p):
      return self._state_difference(self._user_initial_state_fn(p), x)

    def stage_lagrangian(x, u, xnext, k, ld, le, li, scale, p):
      value = scale * self._user_stage_cost_fn(x, u, k, p)
      if not self._gauss_newton:
        value += jp.vdot(ld, dynamics_local(x, u, xnext, k, p))
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
            stage_derivative(stage_lagrangian, jax.hessian),
            in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
        )(xs[:-1], us, xs[1:], ks, ld, le, li, scale, p)
        hxx, hxu, huu = blocks[0][0], blocks[0][1], blocks[1][1]
      else:
        hxx, hxu, huu = (
            jp.zeros((0, tx, tx)),
            jp.zeros((0, tx, nu)),
            jp.zeros((0, nu, nu)),
        )
      terminal = state_derivative(terminal_lagrangian, jax.hessian)(
          xs[-1], lte, lti, scale, p
      )
      hxx = jp.concatenate([hxx, terminal[None]])
      if not self._gauss_newton:
        initial = state_derivative(
            lambda x, p: jp.vdot(lam_init, initial_residual(x, p)), jax.hessian
        )(xs[0], p)
        hxx = hxx.at[0].add(initial)
      return hxx, hxu, huu

    xs, us = jp.broadcast_to(x, (N + 1, nx)), jp.zeros((N, nu))
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
          batch(stage_derivative(fn, jax.jacfwd), ((size, tx), (size, nu))),
          xs[:-1],
          us,
      )
    for name, fn in (
        ('terminal_equality', terminal_eq),
        ('terminal_inequality', terminal_ineq),
    ):
      add(name + '_residual', fn, x)
      add(name + '_jacobian', state_derivative(fn, jax.jacfwd), x)
    add('initial_state', self._user_initial_state_fn)
    add('initial_residual', initial_residual, x)
    add('initial_jacobian', state_derivative(initial_residual, jax.jacfwd), x)
    add(
        'lagrangian_hessian',
        lagrangian_hessian,
        xs,
        us,
        jp.zeros((tx,)),
        jp.zeros((N, tx)),
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
    """Return gradients, shapes (N+1, nx_tangent), (N, nu)."""
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

    Multiplier shapes are (tx,), (N, tx), (N, ng), (N, nh), (ng_terminal,),
    and (nh_terminal,). Inequality counts include bounded variables. Blocks
    have shapes (N+1, tx, tx), (N, tx, nu), and (N, nu, nu), where tx is
    nx_tangent. Dynamics curvature uses the fixed next-state chart.
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
    """Return differences from xs_next to predicted states, shape (N, tx)."""
    return self._evaluate('dynamics_residual', xs, us, xs_next, params)

  def dynamics_jacobian(self, xs, us, xs_next, params):
    """Return (A, B), shapes (N, tx, tx), (N, tx, nu), with tx=nx_tangent.

    When full_dynamics_jacobian=False, for quaternions
        q̅ₖ₊₁ Exp(δθₖ₊₁) = f(Retr(x̅ₖ,δxₖ), Retr(u̅ₖ,δuₖ))
        Exp(δθₖ₊₁) = q̅ₖ₊₁⁻¹ f(Retr(x̅ₖ,δxₖ), Retr(u̅ₖ,δuₖ))
        Log(q̅ₖ₊₁⁻¹ f(Retr(x̅ₖ,δxₖ), Retr(u̅ₖ,δuₖ))) - δθₖ₊₁ = 0
    or in general
        Retr(x̅ₖ₊₁,δxₖ₊₁) = f(Retr(x̅ₖ,δxₖ), Retr(u̅ₖ,δuₖ))
        Difference(f(Retr(x̅ₖ,δxₖ), Retr(u̅ₖ,δuₖ)), x̅ₖ₊₁) - δxₖ₊₁ = 0
    differentiating at zero increments gives
        rₖ + Aₖ δxₖ + Bₖ δuₖ - δxₖ₊₁ = 0
    where
        rₖ = Difference(f(x̅ₖ,u̅ₖ), x̅ₖ₊₁)
        Aₖ = ∂Difference(f(Retr(x̅ₖ,δxₖ),u̅ₖ), x̅ₖ₊₁) / ∂δxₖ
        Bₖ = ∂Difference(f(x̅ₖ,Retr(u̅ₖ,δuₖ)), x̅ₖ₊₁) / ∂δuₖ
    The next-state Jacobian is -I by construction.

    When full_dynamics_jacobian=True, differentiate the original dynamics
    residual with respect to all tangent increments, including the next state.
    For quaternions
        Log((q̅ₖ₊₁ Exp(δθₖ₊₁))⁻¹ f(Retr(x̅ₖ,δxₖ), Retr(u̅ₖ,δuₖ))) = 0
    Or in general
        Difference(f(Retr(x̅ₖ,δxₖ), Retr(u̅ₖ,δuₖ)), Retr(x̅ₖ₊₁,δxₖ₊₁)) = 0
    Differentiating at zero increments gives
        rₖ + Jₓ δxₖ + Jᵤ δuₖ + Jₙ δxₖ₊₁ = 0
    Since Jₙ is generally not -I for quaternion states, multiply the
    linearized equation by -Jₙ⁻¹:
        -Jₙ⁻¹ rₖ - Jₙ⁻¹ Jₓ δxₖ - Jₙ⁻¹ Jᵤ δuₖ - δxₖ₊₁ = 0
    with -Jₙ⁻¹ rₖ = Jₗ(rₖ) rₖ = rₖ.

    Hence, both approaches give the same residual, but generally different
    A and B matrices away from dynamics feasibility.
    """
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
    x, dx = jp.asarray(x), jp.asarray(dx)
    assert x.ndim in (1, 2) and x.shape[-1] == self.nx
    assert dx.shape == (*x.shape[:-1], self.nx_tangent)

    if x.ndim == 2:
      return jax.vmap(self.retract_state, in_axes=(0, 0, None, None))(
          x, dx, alpha, params
      )

    blocks, previous = [], 0
    for count, i in enumerate(self.quaternion_positions):
      blocks.append(x[previous:i] + alpha * dx[previous - count : i - count])
      blocks.append(
          manifold.quat_multiply(
              x[i : i + 4],
              manifold.quat_exp(alpha * dx[i - count : i - count + 3]),
          )
      )
      previous = i + 4
    blocks.append(x[previous:] + alpha * dx[previous-len(self.quaternion_positions):])
    return jp.concatenate(blocks)

  def _state_difference(self, xnext_pred, xnext):
    blocks, previous = [], 0
    for i in self.quaternion_positions:
      blocks.append(xnext_pred[previous:i] - xnext[previous:i])
      blocks.append(
          manifold.quat_log(
              manifold.quat_multiply(
                  manifold.quat_inverse(xnext[i : i + 4]), xnext_pred[i : i + 4]
              )
          )
      )
      previous = i + 4
    blocks.append(xnext_pred[previous:] - xnext[previous:])
    return jp.concatenate(blocks)

  def retract_control(self, u, du, alpha, params):
    u, du = jp.asarray(u), jp.asarray(du)
    assert u.ndim in (1, 2) and u.shape[-1] == self.nu
    assert du.shape == u.shape
    return u + alpha * du
