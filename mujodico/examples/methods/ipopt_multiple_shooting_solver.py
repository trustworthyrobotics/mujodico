from collections.abc import Callable
import time
from typing import Generic, TypeVar, overload

import jax
import jax.numpy as jp
import numpy as np

from mujodico.examples.methods import support


ParamsT = TypeVar('ParamsT')
StageCost = Callable[[jax.Array, jax.Array, ParamsT], jax.Array | np.ndarray]
TerminalCost = Callable[[jax.Array, ParamsT], jax.Array | np.ndarray]
InitialEqualityConstraint = Callable[[jax.Array, ParamsT], jax.Array | np.ndarray]
StageEqualityConstraint = Callable[
  [jax.Array, jax.Array, jax.Array, ParamsT], jax.Array | np.ndarray
]
StageInequalityConstraint = Callable[
  [jax.Array, jax.Array, ParamsT], jax.Array | np.ndarray
]
TerminalConstraint = Callable[[jax.Array, ParamsT], jax.Array | np.ndarray]
Bounds = tuple[jax.Array | np.ndarray, jax.Array | np.ndarray]


class IpoptMultipleShootingSolver:
  """Sparse Ipopt solver for finite-horizon problems with local constraints.

  Constraint callbacks return vectors; inequality callbacks use g(z) <= 0.
  Omitted constraints are treated as empty. Bounds are (lower, upper) pairs.
  Params are passed to solve and may change between solves.
  """

  def __init__(
      self,
      N: int,
      nx: int,
      nu: int,
      *,
      stage_cost: StageCost,
      terminal_cost: TerminalCost,
      initial_equality_constraint: InitialEqualityConstraint | None = None,
      stage_equality_constraint: StageEqualityConstraint | None = None,
      stage_inequality_constraint: StageInequalityConstraint | None = None,
      terminal_equality_constraint: TerminalConstraint | None = None,
      terminal_inequality_constraint: TerminalConstraint | None = None,
      x_bounds: Bounds | None = None,
      u_bounds: Bounds | None = None,
      params_trace: ParamsT | None = None,
      gauss_newton: bool = True,
  ):
    self._cyipopt = support.load_cyipopt()
    self.N = N
    self.nx = nx
    self.nu = nu
    self.stage_cost = stage_cost
    self.terminal_cost = terminal_cost
    self.x_bounds = (
      x_bounds if x_bounds is not None
      else (-np.inf * np.ones(nx), np.inf * np.ones(nx))
    )
    self.u_bounds = (
      u_bounds if u_bounds is not None
      else (-np.inf * np.ones(nu), np.inf * np.ones(nu))
    )
    self._gauss_newton = gauss_newton
    self._params_signature = None
    self._constraint_sizes: tuple[int, int, int, int, int] | None = None
    self._eval_time = 0.0

    def empty_state_constraint(x: jax.Array, params: ParamsT) -> jax.Array:
      del x, params
      return jp.zeros((0,))

    def empty_stage_equality(
        x: jax.Array, u: jax.Array, xnext: jax.Array, params: ParamsT
    ) -> jax.Array:
      del x, u, xnext, params
      return jp.zeros((0,))

    def empty_stage_inequality(
        x: jax.Array, u: jax.Array, params: ParamsT
    ) -> jax.Array:
      del x, u, params
      return jp.zeros((0,))

    self.initial_equality_constraint = (
        initial_equality_constraint
        if initial_equality_constraint is not None
        else empty_state_constraint
    )
    self.stage_equality_constraint = (
        stage_equality_constraint
        if stage_equality_constraint is not None
        else empty_stage_equality
    )
    self.stage_inequality_constraint = (
        stage_inequality_constraint
        if stage_inequality_constraint is not None
        else empty_stage_inequality
    )
    self.terminal_equality_constraint = (
        terminal_equality_constraint
        if terminal_equality_constraint is not None
        else empty_state_constraint
    )
    self.terminal_inequality_constraint = (
        terminal_inequality_constraint
        if terminal_inequality_constraint is not None
        else empty_state_constraint
    )
    if params_trace is not None:
      self._compile_if_needed(params_trace)

  def solve(
      self,
      params: ParamsT,
      xs_init: jax.Array | np.ndarray,
      us_init: jax.Array | np.ndarray,
      **kwargs,
  ) -> tuple[np.ndarray, np.ndarray, dict]:
    self._compile_if_needed(params)

    vars_init = self._pack(np.asarray(xs_init), np.asarray(us_init))

    vars_lb, vars_ub = self._variable_bounds()
    constraint_lb, constraint_ub = self._constraint_bounds()

    problem = self._cyipopt.Problem(
        n=vars_lb.size,
        m=constraint_lb.size,
        problem_obj=self._NLP(params),
        lb=vars_lb,
        ub=vars_ub,
        cl=constraint_lb,
        cu=constraint_ub,
    )

    for k, v in kwargs.items():
      problem.add_option(k, v)

    self._eval_time = 0.0
    tic = time.perf_counter()

    vars_opt, info = problem.solve(vars_init)

    total_time = time.perf_counter() - tic
    eval_time = self._eval_time

    del info['x']
    info['total_time_sec'] = total_time
    info['eval_time_sec'] = eval_time

    xs_opt, us_opt = self._unpack(vars_opt)
    return np.asarray(xs_opt), np.asarray(us_opt), info

  def _get_params_signature(self, params: ParamsT) -> tuple:
    leaves, treedef = jax.tree_util.tree_flatten(params)
    leaf_signatures = []
    for leaf in leaves:
      value = np.asarray(leaf)
      leaf_signatures.append((value.shape, value.dtype.str))
    return treedef, tuple(leaf_signatures)

  def _compile_if_needed(self, params: ParamsT) -> None:
    signature = self._get_params_signature(params)
    if signature != self._params_signature:
      self._compile(params)
      self._params_signature = signature

  def _compile(self, params_trace: ParamsT) -> None:
    N, nx, nu = self.N, self.nx, self.nu
    vars_trace = self._pack(jp.zeros((N + 1, nx)), jp.zeros((N, nu)))

    initial_trace = self.initial_equality_constraint(jp.zeros(nx), params_trace)
    stage_eq_trace = self.stage_equality_constraint(
        jp.zeros(nx), jp.zeros(nu), jp.zeros(nx), params_trace
    )
    stage_ineq_trace = self.stage_inequality_constraint(
        jp.zeros(nx), jp.zeros(nu), params_trace
    )
    terminal_eq_trace = self.terminal_equality_constraint(jp.zeros(nx), params_trace)
    terminal_ineq_trace = self.terminal_inequality_constraint(jp.zeros(nx), params_trace)
    n_initial_eq = initial_trace.size
    n_stage_eq = stage_eq_trace.size
    n_stage_ineq = stage_ineq_trace.size
    n_terminal_eq = terminal_eq_trace.size
    n_terminal_ineq = terminal_ineq_trace.size
    self._constraint_sizes = (
        n_initial_eq,
        n_stage_eq,
        n_stage_ineq,
        n_terminal_eq,
        n_terminal_ineq,
    )
    n_stage_constraints = n_stage_eq + n_stage_ineq

    def objective(vars: jax.Array, params: ParamsT) -> jax.Array:
      xs, us = self._unpack(vars)
      stage_costs = jax.vmap(self.stage_cost, in_axes=(0, 0, None))(
          xs[:-1], us, params
      )
      return jp.sum(stage_costs) + self.terminal_cost(xs[-1], params)

    def constraints(vars: jax.Array, params: ParamsT) -> jax.Array:
      xs, us = self._unpack(vars)
      initial_eq = self.initial_equality_constraint(xs[0], params)
      stage_eq = jax.vmap(
          self.stage_equality_constraint, in_axes=(0, 0, 0, None)
      )(xs[:-1], us, xs[1:], params)
      stage_ineq = jax.vmap(
          self.stage_inequality_constraint, in_axes=(0, 0, None)
      )(xs[:-1], us, params)
      terminal_eq = self.terminal_equality_constraint(xs[-1], params)
      terminal_ineq = self.terminal_inequality_constraint(xs[-1], params)
      return jp.concatenate([
          initial_eq.ravel(),
          jp.concatenate([stage_eq, stage_ineq], axis=1).ravel(),
          terminal_eq.ravel(),
          terminal_ineq.ravel(),
      ])

    def constraints_jacobian_entries(
        vars: jax.Array, params: ParamsT
    ) -> jax.Array:
      xs, us = self._unpack(vars)
      initial_jac = jax.jacfwd(self.initial_equality_constraint)(xs[0], params)

      def stage_jac(x: jax.Array, u: jax.Array, xnext: jax.Array) -> jax.Array:
        local = jp.concatenate([x, u, xnext])

        def local_constraints(var: jax.Array) -> jax.Array:
          x = var[:nx]
          u = var[nx:nx + nu]
          xnext = var[nx + nu:]
          return jp.concatenate([
              self.stage_equality_constraint(x, u, xnext, params),
              self.stage_inequality_constraint(x, u, params),
          ])

        return jax.jacfwd(local_constraints)(local).ravel()

      stage_jacs = jax.vmap(stage_jac)(xs[:-1], us, xs[1:])
      terminal_jac = jax.jacfwd(lambda x: jp.concatenate([
          self.terminal_equality_constraint(x, params),
          self.terminal_inequality_constraint(x, params),
      ]))(xs[-1])
      return jp.concatenate([
          initial_jac.ravel(), stage_jacs.ravel(), terminal_jac.ravel()
      ])

    def stage_lagrangian(
        var: jax.Array,
        initial_lam: jax.Array,
        eq_lam: jax.Array,
        ineq_lam: jax.Array,
        obj_factor: jax.Array,
        params: ParamsT,
    ) -> jax.Array:
      x = var[:nx]
      u = var[nx:nx + nu]
      xnext = var[nx + nu:]
      value = obj_factor * self.stage_cost(x, u, params)
      if not self._gauss_newton :
        if initial_lam.size:
          value += jp.vdot(initial_lam, self.initial_equality_constraint(x, params))
        if eq_lam.size:
          value += jp.vdot(eq_lam, self.stage_equality_constraint(x, u, xnext, params))
        if ineq_lam.size:
          value += jp.vdot(ineq_lam, self.stage_inequality_constraint(x, u, params))
      return value

    def terminal_lagrangian(
        x: jax.Array,
        eq_lam: jax.Array,
        ineq_lam: jax.Array,
        obj_factor: jax.Array,
        params: ParamsT,
    ) -> jax.Array:
      value = obj_factor * self.terminal_cost(x, params)
      if not self._gauss_newton :
        if eq_lam.size:
          value += jp.vdot(eq_lam, self.terminal_equality_constraint(x, params))
        if ineq_lam.size:
          value += jp.vdot(ineq_lam, self.terminal_inequality_constraint(x, params))
      return value

    def lagrangian_hessian_entries(
        vars: jax.Array,
        lagrange: jax.Array,
        obj_factor: jax.Array,
        params: ParamsT,
    ) -> jax.Array:
      xs, us = self._unpack(vars)
      initial_lam = lagrange[:n_initial_eq]
      stage_lam = lagrange[n_initial_eq:n_initial_eq + N * n_stage_constraints]
      stage_lam = stage_lam.reshape((N, n_stage_constraints))
      stage_eq_lam = stage_lam[:, :n_stage_eq]
      stage_ineq_lam = stage_lam[:, n_stage_eq:]
      terminal_lam = lagrange[n_initial_eq + N * n_stage_constraints:]
      terminal_eq_lam = terminal_lam[:n_terminal_eq]
      terminal_ineq_lam = terminal_lam[n_terminal_eq:]

      initial_lams = jp.zeros((N, n_initial_eq)).at[0].set(initial_lam)

      def stage_hessian(
          x: jax.Array,
          u: jax.Array,
          xnext: jax.Array,
          init_lam: jax.Array,
          eq_lam: jax.Array,
          ineq_lam: jax.Array,
      ) -> jax.Array:
        local = jp.concatenate([x, u, xnext])
        return jax.hessian(stage_lagrangian)(
            local, init_lam, eq_lam, ineq_lam, obj_factor, params
        )

      stage_hessians = jax.vmap(stage_hessian)(
          xs[:-1], us, xs[1:], initial_lams, stage_eq_lam, stage_ineq_lam
      )
      stage_hessians = stage_hessians.at[1:, :nx, :nx].add(
          stage_hessians[:-1, nx + nu:, nx + nu:]
      )
      terminal_hessian = jax.hessian(terminal_lagrangian)(
          xs[-1], terminal_eq_lam, terminal_ineq_lam, obj_factor, params
      )
      terminal_hessian += stage_hessians[-1, nx + nu:, nx + nu:]
      stage_i, stage_j = np.tril_indices(2 * nx + nu)
      stage_mask = (stage_i < nx + nu) | (stage_j < nx + nu)
      return jp.concatenate([
          stage_hessians[:, stage_i[stage_mask], stage_j[stage_mask]].ravel(),
          terminal_hessian[jp.tril_indices(nx)],
      ])

    objective_grad = jax.jit(jax.grad(objective)).lower(
        vars_trace,
        params_trace,
    ).compile()
    objective = jax.jit(objective).lower(
        vars_trace,
        params_trace,
    ).compile()
    constraints = jax.jit(constraints).lower(
        vars_trace,
        params_trace,
    ).compile()
    constraints_jacobian_entries = jax.jit(constraints_jacobian_entries).lower(
        vars_trace,
        params_trace,
    ).compile()
    constraints_jacobian_structure = self._make_constraints_jacobian_structure(
        n_initial_eq,
        n_stage_constraints,
        n_terminal_eq + n_terminal_ineq,
    )
    lagrangian_hessian_entries = jax.jit(lagrangian_hessian_entries).lower(
        vars_trace,
        jp.ones(n_initial_eq + N * n_stage_constraints + n_terminal_eq + n_terminal_ineq),
        jp.array(1.0),
        params_trace,
    ).compile()
    lagrangian_hessian_structure = self._make_lagrangian_hessian_structure()

    def timed(fn: Callable) -> Callable:
      def wrapped(*args, **kwargs):
        tic = time.perf_counter()
        out = fn(*args, **kwargs)
        self._eval_time += time.perf_counter() - tic
        return out
      return wrapped

    class NLP:
      def __init__(self, params: ParamsT):
        self.params = params

      @timed
      def objective(self, vars: np.ndarray) -> float:
        return float(objective(jp.asarray(vars), self.params))

      @timed
      def gradient(self, vars: np.ndarray) -> np.ndarray:
        return np.asarray(objective_grad(jp.asarray(vars), self.params))

      @timed
      def constraints(self, vars: np.ndarray) -> np.ndarray:
        return np.asarray(constraints(jp.asarray(vars), self.params))

      @timed
      def jacobian(self, vars: np.ndarray) -> np.ndarray:
        return np.asarray(constraints_jacobian_entries(jp.asarray(vars), self.params))

      def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        return constraints_jacobian_structure

      @timed
      def hessian(self, vars: np.ndarray, lagrange: np.ndarray, objective_factor: float) -> np.ndarray:
        return np.asarray(lagrangian_hessian_entries(
            jp.asarray(vars),
            jp.asarray(lagrange),
            jp.asarray(objective_factor),
            self.params,
        ))

      def hessianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        return lagrangian_hessian_structure

    self._NLP = NLP

  def _variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    x_lb, x_ub = self.x_bounds
    u_lb, u_ub = self.u_bounds
    xs_lb = np.repeat(np.asarray(x_lb)[None, :], self.N + 1, axis=0)
    xs_ub = np.repeat(np.asarray(x_ub)[None, :], self.N + 1, axis=0)
    us_lb = np.repeat(np.asarray(u_lb)[None, :], self.N, axis=0)
    us_ub = np.repeat(np.asarray(u_ub)[None, :], self.N, axis=0)
    return self._pack(xs_lb, us_lb), self._pack(xs_ub, us_ub)

  def _constraint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    constraint_sizes = self._constraint_sizes
    assert constraint_sizes is not None
    n_initial_eq, n_stage_eq, n_stage_ineq, n_terminal_eq, n_terminal_ineq = constraint_sizes
    stage_constraints = np.concatenate([
        np.zeros(n_stage_eq),
        np.full(n_stage_ineq, -np.inf),
    ])
    lower = np.concatenate([
        np.zeros(n_initial_eq),
        np.tile(stage_constraints, self.N),
        np.zeros(n_terminal_eq),
        np.full(n_terminal_ineq, -np.inf),
    ])
    upper = np.zeros(lower.size)
    return lower, upper

  def _make_constraints_jacobian_structure(
      self,
      n_initial_eq: int,
      n_stage: int,
      n_terminal: int,
  ) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    cols = []
    if n_initial_eq:
      init_rows, init_cols = np.indices((n_initial_eq, self.nx))
      rows.append(init_rows.ravel())
      cols.append(init_cols.ravel())
    if n_stage:
      local_rows, local_cols = np.indices((n_stage, 2 * self.nx + self.nu))
      for stage in range(self.N):
        rows.append(local_rows.ravel() + n_initial_eq + stage * n_stage)
        cols.append(local_cols.ravel() + stage * (self.nx + self.nu))
    if n_terminal:
      terminal_rows, terminal_cols = np.indices((n_terminal, self.nx))
      rows.append(terminal_rows.ravel() + n_initial_eq + self.N * n_stage)
      cols.append(terminal_cols.ravel() + self.N * (self.nx + self.nu))
    if not rows:
      return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return np.concatenate(rows).astype(np.int32), np.concatenate(cols).astype(np.int32)

  def _make_lagrangian_hessian_structure(self) -> tuple[np.ndarray, np.ndarray]:
    stage_i, stage_j = np.tril_indices(2 * self.nx + self.nu)
    stage_mask = (stage_i < self.nx + self.nu) | (stage_j < self.nx + self.nu)
    stage_i, stage_j = stage_i[stage_mask], stage_j[stage_mask]
    terminal_i, terminal_j = np.tril_indices(self.nx)
    rows, cols = [], []
    for stage in range(self.N):
      offset = stage * (self.nx + self.nu)
      rows.append(stage_i + offset)
      cols.append(stage_j + offset)
    offset = self.N * (self.nx + self.nu)
    rows.append(terminal_i + offset)
    cols.append(terminal_j + offset)
    return np.concatenate(rows).astype(np.int32), np.concatenate(cols).astype(np.int32)

  @overload
  def _pack(self, xs: jax.Array, us: jax.Array) -> jax.Array: ...

  @overload
  def _pack(self, xs: np.ndarray, us: np.ndarray) -> np.ndarray: ...

  def _pack(
      self,
      xs: jax.Array | np.ndarray,
      us: jax.Array | np.ndarray,
  ) -> jax.Array | np.ndarray:
    assert xs.shape == (self.N + 1, self.nx)
    assert us.shape == (self.N, self.nu)
    cp = jp if isinstance(xs, jax.Array) else np
    return cp.concatenate([cp.hstack((xs[:-1], us)).ravel(), xs[-1]])

  @overload
  def _unpack(self, vars: jax.Array) -> tuple[jax.Array, jax.Array]: ...

  @overload
  def _unpack(self, vars: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

  def _unpack(
      self,
      vars: jax.Array | np.ndarray,
  ) -> tuple[jax.Array | np.ndarray, jax.Array | np.ndarray]:
    assert vars.shape == ((self.N + 1) * self.nx + self.N * self.nu,)
    cp = jp if isinstance(vars, jax.Array) else np
    stage = vars[:-self.nx].reshape(self.N, self.nx + self.nu)
    xs = cp.vstack((stage[:, :self.nx], vars[-self.nx:]))
    return xs, stage[:, self.nx:]
