"""Structured OCP callbacks and exact retraction-coordinate derivatives.

The Newton-only next-state copy makes a general manifold edge Lagrangian
stage-local.  No next-state curvature or physical multiplier transformation
is discarded.  Eliminating the copy reproduces the original tangent KKT.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .manifold import Euclidean


class Primal(NamedTuple):
  xs: jax.Array
  us: jax.Array
  s: jax.Array
  st: jax.Array
  ut: jax.Array


class Evaluation(NamedTuple):
  objective: jax.Array
  eq: jax.Array
  eqT: jax.Array


def _empty_stage(x, u, params):
  del u, params
  return jnp.zeros((0,), x.dtype)


def _empty_terminal(x, params):
  del params
  return jnp.zeros((0,), x.dtype)


def _box(bounds, size):
  if bounds is None:
    return np.full(size, -np.inf), np.full(size, np.inf)
  lo = np.broadcast_to(np.asarray(bounds[0], dtype=float), (size,)).copy()
  hi = np.broadcast_to(np.asarray(bounds[1], dtype=float), (size,)).copy()
  if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
    raise ValueError('Bounds must have no NaNs and satisfy lower <= upper')
  return lo, hi


def _native_box(bounds, mapping):
  lo, hi = bounds
  native = np.any(np.asarray(mapping) != 0, axis=1)
  fixed = np.isfinite(lo) & (lo == hi)
  routed = ~native & (np.isfinite(lo) | np.isfinite(hi)) & ~fixed
  keep = native & ~fixed
  return (
      np.where(keep, lo, -np.inf),
      np.where(keep, hi, np.inf),
      np.flatnonzero(fixed),
      np.flatnonzero(routed),
  )


def _constraint_spec(bounds, lower, upper):
  if bounds is not None:
    if lower is not None or upper is not None:
      raise ValueError(
          'Specify either inequality_bounds or lower/upper aliases'
      )
    lower, upper = bounds
  if lower is None and upper is None:
    lower, upper = 0.0, np.inf
  else:
    lower = -np.inf if lower is None else lower
    upper = np.inf if upper is None else upper
  lo, hi = np.asarray(lower, float), np.asarray(upper, float)
  if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
    raise ValueError('Invalid nonlinear constraint bounds')
  return lo, hi


def _partition(value, spec):
  value = jnp.asarray(value).reshape(-1)
  lo = np.broadcast_to(spec[0], value.shape)
  hi = np.broadcast_to(spec[1], value.shape)
  fixed = np.isfinite(lo) & (lo == hi)
  # Completely unbounded rows carry no information and need no slack.
  keep = ~fixed & (np.isfinite(lo) | np.isfinite(hi))
  i, f = np.flatnonzero(keep), np.flatnonzero(fixed)
  return value[i], value[f] - jnp.asarray(lo[f], value.dtype), lo[i], hi[i]


class Problem:
  """Uniform-stage nonlinear OCP with dynamic parameter PyTrees.

  Bounds have fixed shapes and finite-side topology. Costs and physical
  callbacks receive ``params`` on every call. Construction captures only
  structure, never the values of the parameter PyTree.

  ``terminal_*_with_control`` and terminal controls are useful to construct
  structured auxiliary NLPs, such as elastic feasibility restoration.
  """

  def __init__(
      self,
      *,
      horizon,
      num_states,
      num_controls,
      stage_cost,
      terminal_cost,
      initial_state,
      dynamics,
      stage_equality=None,
      stage_inequality=None,
      terminal_equality=None,
      terminal_inequality=None,
      x_bounds=None,
      u_bounds=None,
      stage_inequality_bounds=None,
      terminal_inequality_bounds=None,
      ineq_lb=None,
      ineq_ub=None,
      terminal_ineq_lb=None,
      terminal_ineq_ub=None,
      state_manifold=None,
      control_manifold=None,
      terminal_control_dim=0,
      terminal_control_bounds=None,
      terminal_cost_with_control=None,
      terminal_equality_with_control=None,
      terminal_inequality_with_control=None,
  ):
    if horizon < 1:
      raise ValueError('horizon must be at least one')
    if num_states < 1 or num_controls < 0 or terminal_control_dim < 0:
      raise ValueError('Invalid state or control dimension')
    self.horizon = self.N = int(horizon)
    self.num_states = self.nx = int(num_states)
    self.num_controls = self.nu = int(num_controls)
    self.state_manifold = state_manifold or Euclidean(self.nx)
    self.control_manifold = control_manifold or Euclidean(self.nu)
    if self.state_manifold.storage_dim != self.nx:
      raise ValueError(
          'state_manifold storage dimension does not match num_states'
      )
    if self.control_manifold.storage_dim != self.nu:
      raise ValueError(
          'control_manifold storage dimension does not match num_controls'
      )
    self.nxt = self.state_manifold.tangent_dim
    self.nut = self.control_manifold.tangent_dim
    self.n_terminal_controls = self.nt = int(terminal_control_dim)
    self.state_map = jnp.asarray(self.state_manifold.native_map)
    self.control_map = jnp.asarray(self.control_manifold.native_map)
    self.initial_state = initial_state
    self._dynamics = dynamics
    self._stage_cost = stage_cost
    self._terminal_cost = terminal_cost
    self._stage_h = stage_equality or _empty_stage
    self._stage_g = stage_inequality or _empty_stage
    self._terminal_h = terminal_equality or _empty_terminal
    self._terminal_g = terminal_inequality or _empty_terminal
    self._terminal_cost_control = terminal_cost_with_control
    self._terminal_h_control = terminal_equality_with_control
    self._terminal_g_control = terminal_inequality_with_control
    self.original_x_bounds = _box(x_bounds, self.nx)
    self.original_u_bounds = _box(u_bounds, self.nu)
    self.original_ut_bounds = _box(terminal_control_bounds, self.nt)
    self._xb = _native_box(
        self.original_x_bounds, self.state_manifold.native_map
    )
    self._ub = _native_box(
        self.original_u_bounds, self.control_manifold.native_map
    )
    self._utb = _native_box(self.original_ut_bounds, np.eye(self.nt))
    self._gb = _constraint_spec(stage_inequality_bounds, ineq_lb, ineq_ub)
    self._gtb = _constraint_spec(
        terminal_inequality_bounds, terminal_ineq_lb, terminal_ineq_ub
    )

  def stage_objective(self, x, u, params):
    return jnp.asarray(self._stage_cost(x, u, params), dtype=x.dtype).reshape(
        ()
    )

  def dynamics(self, x, u, params):
    return jnp.asarray(self._dynamics(x, u, params), dtype=x.dtype)

  def terminal_objective(self, x, ut, params):
    if self._terminal_cost_control is not None:
      return jnp.asarray(
          self._terminal_cost_control(x, ut, params), dtype=x.dtype
      ).reshape(())
    return jnp.asarray(self._terminal_cost(x, params), dtype=x.dtype).reshape(
        ()
    )

  def stage_h(self, x, u, params):
    _, fixed_g, _, _ = _partition(
        jnp.asarray(self._stage_g(x, u, params), x.dtype), self._gb
    )
    return jnp.concatenate((
        jnp.asarray(self._stage_h(x, u, params), x.dtype).reshape(-1),
        fixed_g,
        x[self._xb[2]]
        - jnp.asarray(self.original_x_bounds[0][self._xb[2]], x.dtype),
        u[self._ub[2]]
        - jnp.asarray(self.original_u_bounds[0][self._ub[2]], x.dtype),
    ))

  def stage_g(self, x, u, params):
    g, _, _, _ = _partition(
        jnp.asarray(self._stage_g(x, u, params), x.dtype), self._gb
    )
    return jnp.concatenate((g, x[self._xb[3]], u[self._ub[3]]))

  def terminal_h(self, x, ut, params):
    hg = (
        self._terminal_h(x, params)
        if self._terminal_h_control is None
        else self._terminal_h_control(x, ut, params)
    )
    gg = (
        self._terminal_g(x, params)
        if self._terminal_g_control is None
        else self._terminal_g_control(x, ut, params)
    )
    hg, gg = jnp.asarray(hg, x.dtype), jnp.asarray(gg, x.dtype)
    _, fixed_g, _, _ = _partition(gg, self._gtb)
    return jnp.concatenate((
        jnp.asarray(hg).reshape(-1),
        fixed_g,
        x[self._xb[2]]
        - jnp.asarray(self.original_x_bounds[0][self._xb[2]], x.dtype),
        ut[self._utb[2]]
        - jnp.asarray(self.original_ut_bounds[0][self._utb[2]], x.dtype),
    ))

  def terminal_g(self, x, ut, params):
    gg = (
        self._terminal_g(x, params)
        if self._terminal_g_control is None
        else self._terminal_g_control(x, ut, params)
    )
    gg = jnp.asarray(gg, x.dtype)
    g, _, _, _ = _partition(gg, self._gtb)
    return jnp.concatenate((g, x[self._xb[3]]))

  def _stage_g_bounds(self, x, u, params):
    _, _, lo, hi = _partition(
        jnp.asarray(self._stage_g(x, u, params), x.dtype), self._gb
    )
    return (
        jnp.asarray(
            np.concatenate((
                lo,
                self.original_x_bounds[0][self._xb[3]],
                self.original_u_bounds[0][self._ub[3]],
            )),
            x.dtype,
        ),
        jnp.asarray(
            np.concatenate((
                hi,
                self.original_x_bounds[1][self._xb[3]],
                self.original_u_bounds[1][self._ub[3]],
            )),
            x.dtype,
        ),
    )

  def _terminal_g_bounds(self, x, ut, params):
    gg = (
        self._terminal_g(x, params)
        if self._terminal_g_control is None
        else self._terminal_g_control(x, ut, params)
    )
    gg = jnp.asarray(gg, x.dtype)
    _, _, lo, hi = _partition(gg, self._gtb)
    return (
        jnp.asarray(
            np.concatenate((lo, self.original_x_bounds[0][self._xb[3]])),
            x.dtype,
        ),
        jnp.asarray(
            np.concatenate((hi, self.original_x_bounds[1][self._xb[3]])),
            x.dtype,
        ),
    )

  def initialize(self, params, xs=None, us=None, ut=None):
    x0 = jnp.asarray(self.initial_state(params))
    if x0.shape != (self.nx,):
      raise ValueError('initial_state must return a stored state vector')
    if us is None:
      us = jnp.broadcast_to(
          self.control_manifold.identity(x0.dtype), (self.N, self.nu)
      )
    else:
      us = jnp.asarray(us, x0.dtype)
    if us.shape != (self.N, self.nu):
      raise ValueError('Initial control trajectory has an incompatible shape')
    us = jax.vmap(
        lambda u: self.control_manifold.retract(
            u, jnp.zeros(self.nut, x0.dtype), params
        )
    )(us)
    if xs is None:

      def step(x, u):
        xn = self.dynamics(x, u, params)
        return xn, xn

      _, tail = jax.lax.scan(step, x0, us)
      xs = jnp.concatenate((x0[None], tail), axis=0)
    else:
      xs = jnp.asarray(xs, x0.dtype)
      if xs.shape != (self.N + 1, self.nx):
        raise ValueError('Initial state trajectory has an incompatible shape')
      xs = xs.at[0].set(x0)
    xs = (
        jax.vmap(
            lambda x: self.state_manifold.retract(
                x, jnp.zeros(self.nxt, x0.dtype), params
            )
        )(xs)
        .at[0]
        .set(x0)
    )
    if ut is None:
      ut = jnp.zeros(self.nt, x0.dtype)
    else:
      ut = jnp.asarray(ut, x0.dtype)
    if ut.shape != (self.nt,):
      raise ValueError('Initial terminal control has an incompatible shape')
    s = jax.vmap(lambda x, u: self.stage_g(x, u, params))(xs[:-1], us)
    st = self.terminal_g(xs[-1], ut, params)
    return Primal(xs, us, s, st, ut)

  def bounds(self, primal, params):
    dtype = primal.xs.dtype
    gl, gu = self._stage_g_bounds(primal.xs[0], primal.us[0], params)
    tl, tu = self._terminal_g_bounds(primal.xs[-1], primal.ut, params)
    xl = jnp.broadcast_to(jnp.asarray(self._xb[0], dtype), primal.xs.shape)
    xu = jnp.broadcast_to(jnp.asarray(self._xb[1], dtype), primal.xs.shape)
    # x0 is prescribed and has no barrier variable.
    xl, xu = xl.at[0].set(-jnp.inf), xu.at[0].set(jnp.inf)
    lower = Primal(
        xl,
        jnp.broadcast_to(jnp.asarray(self._ub[0], dtype), primal.us.shape),
        jnp.broadcast_to(gl, primal.s.shape),
        tl,
        jnp.asarray(self._utb[0], dtype),
    )
    upper = Primal(
        xu,
        jnp.broadcast_to(jnp.asarray(self._ub[1], dtype), primal.us.shape),
        jnp.broadcast_to(gu, primal.s.shape),
        tu,
        jnp.asarray(self._utb[1], dtype),
    )
    return lower, upper

  def evaluate(self, primal, params):
    def stage(x, u, s, xn):
      c = jnp.concatenate((
          self.state_manifold.local(xn, self.dynamics(x, u, params), params),
          self.stage_h(x, u, params),
          self.stage_g(x, u, params) - s,
      ))
      return self.stage_objective(x, u, params), c

    objectives, eq = jax.vmap(stage)(
        primal.xs[:-1], primal.us, primal.s, primal.xs[1:]
    )
    eqT = jnp.concatenate((
        self.terminal_h(primal.xs[-1], primal.ut, params),
        self.terminal_g(primal.xs[-1], primal.ut, params) - primal.st,
    ))
    objective = jnp.sum(objectives) + self.terminal_objective(
        primal.xs[-1], primal.ut, params
    )
    return Evaluation(objective, eq, eqT)

  def linearize(
      self,
      primal,
      eq,
      eqT,
      params,
      objective_scale=1.0,
      hessian_mode='exact',
      stagewise=False,
  ):
    from .linear_solver import RHS, System

    nx, nu, ng = self.nxt, self.nut, primal.s.shape[1]
    ny = nu + ng + nx
    dtype = primal.xs.dtype
    zero = jnp.zeros(nx + ny, dtype)

    def edge(x, u, s, xn, multipliers):
      def functions(delta):
        xx = self.state_manifold.retract(x, delta[:nx], params)
        uu = self.control_manifold.retract(u, delta[nx : nx + nu], params)
        ss = s + delta[nx + nu : nx + nu + ng]
        xp = self.state_manifold.retract(xn, delta[-nx:], params)
        c = jnp.concatenate((
            self.state_manifold.local(
                xp, self.dynamics(xx, uu, params), params
            ),
            self.stage_h(xx, uu, params),
            self.stage_g(xx, uu, params) - ss,
        ))
        return self.stage_objective(xx, uu, params), c

      def lag(delta):
        obj, c = functions(delta)
        return objective_scale * obj + jnp.dot(multipliers, c)

      gradient = jax.grad(lag)(zero)
      jac = jax.jacfwd(lambda d: functions(d)[1])(zero)
      if hessian_mode in ('exact', 'exact_lagrangian'):
        H = jax.hessian(lag)(zero)
      elif hessian_mode in ('cost', 'cost_only'):
        H = jax.hessian(lambda d: objective_scale * functions(d)[0])(zero)
      elif hessian_mode == 'zero':
        H = jnp.zeros((zero.size, zero.size), dtype)
      else:
        raise ValueError('Unknown hessian_mode')
      return H, gradient, jac, functions(zero)[1]

    inputs = (primal.xs[:-1], primal.us, primal.s, primal.xs[1:], eq)
    if stagewise:
      H, gradient, J, c = jax.lax.map(lambda values: edge(*values), inputs)
    else:
      H, gradient, J, c = jax.vmap(edge)(*inputs)
    nt, ngt = self.nt, primal.st.size
    zerot = jnp.zeros(nx + nt + ngt, dtype)

    def terminal(delta):
      xx = self.state_manifold.retract(primal.xs[-1], delta[:nx], params)
      uu = primal.ut + delta[nx : nx + nt]
      ss = primal.st + delta[nx + nt :]
      return self.terminal_objective(xx, uu, params), jnp.concatenate((
          self.terminal_h(xx, uu, params),
          self.terminal_g(xx, uu, params) - ss,
      ))

    def terminal_lag(delta):
      obj, con = terminal(delta)
      return objective_scale * obj + jnp.dot(eqT, con)

    gradt = jax.grad(terminal_lag)(zerot)
    jt = jax.jacfwd(lambda d: terminal(d)[1])(zerot)
    if hessian_mode in ('exact', 'exact_lagrangian'):
      ht = jax.hessian(terminal_lag)(zerot)
    elif hessian_mode in ('cost', 'cost_only'):
      ht = jax.hessian(lambda d: objective_scale * terminal(d)[0])(zerot)
    else:
      ht = jnp.zeros((zerot.size, zerot.size), dtype)
    system = System(
        H[:, :nx, :nx],
        H[:, :nx, nx:],
        H[:, nx:, nx:],
        jnp.zeros((self.N, nx, nx), dtype),
        jnp.broadcast_to(
            jnp.concatenate(
                (jnp.zeros((nx, nu + ng), dtype), jnp.eye(nx, dtype=dtype)),
                axis=1,
            ),
            (self.N, nx, ny),
        ),
        J[:, :, :nx],
        J[:, :, nx:],
        jnp.zeros(c.shape, dtype),
        ht[:nx, :nx],
        ht[:nx, nx:],
        ht[nx:, nx:],
        jt[:, :nx],
        jt[:, nx:],
        jnp.zeros(eqT.shape, dtype),
    )
    rhs = RHS(
        gradient[:, :nx],
        gradient[:, nx:],
        c,
        jnp.zeros((self.N, nx), dtype),
        gradt[:nx],
        gradt[nx:],
        terminal(zerot)[1],
        jnp.zeros(nx, dtype),
    )
    return system, rhs

  def retract(self, primal, direction, alpha, params):
    nu, ng = self.nut, primal.s.shape[1]
    xs = (
        jax.vmap(
            lambda x, dx: self.state_manifold.retract(x, alpha * dx, params)
        )(primal.xs, direction.dx)
        .at[0]
        .set(self.initial_state(params))
    )
    us = jax.vmap(
        lambda u, du: self.control_manifold.retract(u, alpha * du, params)
    )(primal.us, direction.dy[:, :nu])
    return Primal(
        xs,
        us,
        primal.s + alpha * direction.dy[:, nu : nu + ng],
        primal.st + alpha * direction.dyT[self.nt :],
        primal.ut + alpha * direction.dyT[: self.nt],
    )

  def valid(self, primal, params):
    state_ok = jnp.all(
        jax.vmap(lambda x: self.state_manifold.valid(x, params))(primal.xs)
    )
    control_ok = jnp.all(
        jax.vmap(lambda u: self.control_manifold.valid(u, params))(primal.us)
    )
    charts_ok = jnp.all(
        jax.vmap(
            lambda x, u, xn: self.state_manifold.chart_valid(
                xn, self.dynamics(x, u, params), params
            )
        )(primal.xs[:-1], primal.us, primal.xs[1:])
    )
    return (
        state_ok
        & control_ok
        & charts_ok
        & jnp.all(jnp.isfinite(primal.s))
        & jnp.all(jnp.isfinite(primal.st))
        & jnp.all(jnp.isfinite(primal.ut))
    )


OcpProblem = Problem
