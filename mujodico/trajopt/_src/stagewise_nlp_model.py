import time
from typing import overload
from typing_extensions import override

import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.abstract_nlp_model import AbstractNLPModel
from mujodico.trajopt._src.types import (
  Bounds,
  Callable,
  InitialEqualityConstraint,
  LocalEqualityConstraint,
  LocalInequalityConstraint,
  Params,
  StageCost,
  TerminalCost,
  TerminalEqualityConstraint,
  TerminalInequalityConstraint,
)


class StagewiseNLPModel(AbstractNLPModel):
  """Represents the stagewise multiple-shooting nonlinear program.

    minimizeₓ,ᵤ   ∑ₖ₌₀ᴺ⁻¹ stage_cost(xₖ, uₖ, params) + terminal_cost(x_N, params)
    subject to    initial_equality_constraint(x₀, params) = 0
                  local_equality_constraint(xₖ, uₖ, xₖ₊₁, params) = 0    for k=0,...,N-1
                  local_inequality_constraint(xₖ, uₖ, xₖ₊₁, params) ≥ 0  for k=0,...,N-1
                  terminal_equality_constraint(x_N, params) = 0
                  terminal_inequality_constraint(x_N, params) ≥ 0
                  x_lb ≤ xₖ ≤ x_ub                                      for k=0,...,N
                  u_lb ≤ uₖ ≤ u_ub                                      for k=0,...,N-1
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
      local_equality_constraint: LocalEqualityConstraint | None = None,
      local_inequality_constraint: LocalInequalityConstraint | None = None,
      terminal_equality_constraint: TerminalEqualityConstraint | None = None,
      terminal_inequality_constraint: TerminalInequalityConstraint | None = None,
      x_bounds: Bounds | None = None,
      u_bounds: Bounds | None = None,
      gauss_newton: bool = True,
      params: Params | None = None,
      update_params: Callable[[dict, Params], Params] | None = None,
  ):
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
    self._callback_time = 0.0
    self._conversion_time = 0.0

    def empty_state_constraint(x: jax.Array, params: Params) -> jax.Array:
      del x, params
      return jp.zeros((0,))

    def empty_local_constraint(
        x: jax.Array, u: jax.Array, xnext: jax.Array, params: Params
    ) -> jax.Array:
      del x, u, xnext, params
      return jp.zeros((0,))

    self.initial_equality_constraint = (
        initial_equality_constraint
        if initial_equality_constraint is not None
        else empty_state_constraint
    )
    self.local_equality_constraint = (
        local_equality_constraint
        if local_equality_constraint is not None
        else empty_local_constraint
    )
    self.local_inequality_constraint = (
        local_inequality_constraint
        if local_inequality_constraint is not None
        else empty_local_constraint
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
    self.update_params = update_params
    if params is not None:
      self.set_params(params)

  @override
  @property
  def num_vars(self) -> int:
    return (self.N + 1) * self.nx + self.N * self.nu

  @override
  @property
  def var_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    return self._variable_bounds()

  @override
  @property
  def num_constraints(self) -> int:
    return self.constraint_bounds[0].size

  @override
  @property
  def constraint_sizes(self) -> tuple[int, int, int, int, int]:
    """Counts of initial, local equality/inequality, and terminal constraints."""
    self._require_compiled()
    assert self._constraint_sizes is not None
    return self._constraint_sizes

  @override
  @property
  def constraint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    self._require_compiled()
    return self._constraint_bounds()

  @override
  def objective(self, x: np.ndarray) -> float:
    return float(self._evaluate(self._objective, x))

  @override
  def gradient(self, x: np.ndarray) -> np.ndarray:
    return self._evaluate(self._objective_grad, x)

  @override
  def constraints(self, x: np.ndarray) -> np.ndarray:
    return self._evaluate(self._constraints, x)

  @override
  def jacobian_structure(self) -> list[tuple[int, int]]:
    self._require_compiled()
    rows, cols = self._jacobian_structure
    return list(zip(rows.tolist(), cols.tolist()))

  @override
  def jacobian_entries(self, x: np.ndarray) -> np.ndarray:
    return self._evaluate(self._jacobian_entries, x)

  @override
  def jacobian(self, x: np.ndarray) -> np.ndarray:
    return self.jacobian_entries(x)

  @override
  def hessian_structure(self) -> list[tuple[int, int]]:
    self._require_compiled()
    rows, cols = self._hessian_structure
    return list(zip(rows.tolist(), cols.tolist()))

  @override
  def hessian_entries(
      self,
      x: np.ndarray,
      lagrange: np.ndarray,
      objective_factor: float,
  ) -> np.ndarray:
    return self._evaluate(
        self._hessian_entries,
        x,
        lagrange,
        objective_factor,
    )

  @override
  def hessian(self, x, lagrange, objective_factor):
    return self.hessian_entries(x, lagrange, objective_factor)

  @override
  def intermediate(self, stats: dict) -> None:
    if self.update_params is not None:
      self._params = self.update_params(stats, self._params)

  def _require_compiled(self):
    if self._params_signature is None or not hasattr(self, '_objective'):
      raise RuntimeError('Call set_params(params) before evaluating the NLP.')
    return self

  def _evaluate(self, fn, *args) -> np.ndarray:
    self._require_compiled()
    tic = time.perf_counter()
    jax_args = tuple(jp.asarray(arg) for arg in args)
    for arg in jax_args:
      arg.block_until_ready()
    self._conversion_time += time.perf_counter() - tic

    tic = time.perf_counter()
    result = fn(*jax_args, self._params)
    if isinstance(result, jax.Array):
      result.block_until_ready()
    self._callback_time += time.perf_counter() - tic

    tic = time.perf_counter()
    result = np.asarray(result)
    self._conversion_time += time.perf_counter() - tic
    return result

  @property
  def stats(self):
    r = {
        'callback_time': self._callback_time,
        'conversion_time': self._conversion_time,
    }
    self._callback_time = 0.0
    self._conversion_time = 0.0
    return r

  @staticmethod
  def _get_params_signature(params: Params) -> tuple:
    leaves, treedef = jax.tree_util.tree_flatten(params)
    leaf_signatures = []
    for leaf in leaves:
      value = np.asarray(leaf)
      leaf_signatures.append((value.shape, value.dtype.str))
    return treedef, tuple(leaf_signatures)

  def set_params(self, params: Params) -> None:
    self._params = params
    signature = self._get_params_signature(params)
    if signature != self._params_signature:
      self._compile(params)
      self._params_signature = signature

  def _compile(self, params_trace: Params) -> None:
    N, nx, nu = self.N, self.nx, self.nu
    vars_trace = self._pack(jp.zeros((N + 1, nx)), jp.zeros((N, nu)))

    initial_trace = self.initial_equality_constraint(jp.zeros(nx), params_trace)
    local_eq_trace = self.local_equality_constraint(
        jp.zeros(nx), jp.zeros(nu), jp.zeros(nx), params_trace
    )
    local_ineq_trace = self.local_inequality_constraint(
        jp.zeros(nx), jp.zeros(nu), jp.zeros(nx), params_trace
    )
    terminal_eq_trace = self.terminal_equality_constraint(jp.zeros(nx), params_trace)
    terminal_ineq_trace = self.terminal_inequality_constraint(jp.zeros(nx), params_trace)
    n_initial_eq = initial_trace.size
    n_local_eq = local_eq_trace.size
    n_local_ineq = local_ineq_trace.size
    n_terminal_eq = terminal_eq_trace.size
    n_terminal_ineq = terminal_ineq_trace.size
    self._constraint_sizes = (
        n_initial_eq,
        n_local_eq,
        n_local_ineq,
        n_terminal_eq,
        n_terminal_ineq,
    )
    n_local_constraints = n_local_eq + n_local_ineq

    def objective(vars: jax.Array, params: Params) -> jax.Array:
      xs, us = self._unpack(vars)
      stage_costs = jax.vmap(self.stage_cost, in_axes=(0, 0, None))(
          xs[:-1], us, params
      )
      return jp.sum(stage_costs) + self.terminal_cost(xs[-1], params)

    def constraints(vars: jax.Array, params: Params) -> jax.Array:
      xs, us = self._unpack(vars)
      initial_eq = self.initial_equality_constraint(xs[0], params)
      local_eq = jax.vmap(
          self.local_equality_constraint, in_axes=(0, 0, 0, None)
      )(xs[:-1], us, xs[1:], params)
      local_ineq = jax.vmap(
          self.local_inequality_constraint, in_axes=(0, 0, 0, None)
      )(xs[:-1], us, xs[1:], params)
      terminal_eq = self.terminal_equality_constraint(xs[-1], params)
      terminal_ineq = self.terminal_inequality_constraint(xs[-1], params)
      return jp.concatenate([
          initial_eq.ravel(),
          jp.concatenate([local_eq, local_ineq], axis=1).ravel(),
          terminal_eq.ravel(),
          terminal_ineq.ravel(),
      ])

    def constraints_jacobian_entries(
        vars: jax.Array, params: Params
    ) -> jax.Array:
      xs, us = self._unpack(vars)
      initial_jac = jax.jacfwd(self.initial_equality_constraint)(xs[0], params)

      def local_jac(x: jax.Array, u: jax.Array, xnext: jax.Array) -> jax.Array:
        local = jp.concatenate([x, u, xnext])

        def local_constraints(var: jax.Array) -> jax.Array:
          x = var[:nx]
          u = var[nx:nx + nu]
          xnext = var[nx + nu:]
          return jp.concatenate([
              self.local_equality_constraint(x, u, xnext, params),
              self.local_inequality_constraint(x, u, xnext, params),
          ])

        return jax.jacfwd(local_constraints)(local).ravel()

      local_jacs = jax.vmap(local_jac)(xs[:-1], us, xs[1:])
      terminal_jac = jax.jacfwd(lambda x: jp.concatenate([
          self.terminal_equality_constraint(x, params),
          self.terminal_inequality_constraint(x, params),
      ]))(xs[-1])
      return jp.concatenate([
          initial_jac.ravel(), local_jacs.ravel(), terminal_jac.ravel()
      ])

    def local_lagrangian(
        var: jax.Array,
        initial_lam: jax.Array,
        eq_lam: jax.Array,
        ineq_lam: jax.Array,
        obj_factor: jax.Array,
        params: Params,
    ) -> jax.Array:
      x = var[:nx]
      u = var[nx:nx + nu]
      xnext = var[nx + nu:]
      value = obj_factor * self.stage_cost(x, u, params)
      if not self._gauss_newton :
        if initial_lam.size:
          value += jp.vdot(initial_lam, self.initial_equality_constraint(x, params))
        if eq_lam.size:
          value += jp.vdot(eq_lam, self.local_equality_constraint(x, u, xnext, params))
        if ineq_lam.size:
          value += jp.vdot(ineq_lam, self.local_inequality_constraint(x, u, xnext, params))
      return value

    def terminal_lagrangian(
        x: jax.Array,
        eq_lam: jax.Array,
        ineq_lam: jax.Array,
        obj_factor: jax.Array,
        params: Params,
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
        params: Params,
    ) -> jax.Array:
      xs, us = self._unpack(vars)
      initial_lam = lagrange[:n_initial_eq]
      local_lam = lagrange[n_initial_eq:n_initial_eq + N * n_local_constraints]
      local_lam = local_lam.reshape((N, n_local_constraints))
      local_eq_lam = local_lam[:, :n_local_eq]
      local_ineq_lam = local_lam[:, n_local_eq:]
      terminal_lam = lagrange[n_initial_eq + N * n_local_constraints:]
      terminal_eq_lam = terminal_lam[:n_terminal_eq]
      terminal_ineq_lam = terminal_lam[n_terminal_eq:]

      initial_lams = jp.zeros((N, n_initial_eq)).at[0].set(initial_lam)

      def local_hessian(
          x: jax.Array,
          u: jax.Array,
          xnext: jax.Array,
          init_lam: jax.Array,
          eq_lam: jax.Array,
          ineq_lam: jax.Array,
      ) -> jax.Array:
        local = jp.concatenate([x, u, xnext])
        return jax.hessian(local_lagrangian)(
            local, init_lam, eq_lam, ineq_lam, obj_factor, params
        )

      local_hessians = jax.vmap(local_hessian)(
          xs[:-1], us, xs[1:], initial_lams, local_eq_lam, local_ineq_lam
      )
      local_hessians = local_hessians.at[1:, :nx, :nx].add(
          local_hessians[:-1, nx + nu:, nx + nu:]
      )
      terminal_hessian = jax.hessian(terminal_lagrangian)(
          xs[-1], terminal_eq_lam, terminal_ineq_lam, obj_factor, params
      )
      terminal_hessian += local_hessians[-1, nx + nu:, nx + nu:]
      local_i, local_j = np.tril_indices(2 * nx + nu)
      local_mask = (local_i < nx + nu) | (local_j < nx + nu)
      return jp.concatenate([
          local_hessians[:, local_i[local_mask], local_j[local_mask]].ravel(),
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
    constraints_jacobian_structure = self._make_constraints_jacobian_structure()
    lagrangian_hessian_entries = jax.jit(lagrangian_hessian_entries).lower(
        vars_trace,
        jp.ones(n_initial_eq + N * n_local_constraints + n_terminal_eq + n_terminal_ineq),
        jp.array(1.0),
        params_trace,
    ).compile()
    lagrangian_hessian_structure = self._make_lagrangian_hessian_structure()

    self._objective = objective
    self._objective_grad = objective_grad
    self._constraints = constraints
    self._jacobian_entries = constraints_jacobian_entries
    self._hessian_entries = lagrangian_hessian_entries
    self._jacobian_structure = constraints_jacobian_structure
    self._hessian_structure = lagrangian_hessian_structure

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
    n_initial_eq, n_local_eq, n_local_ineq, n_terminal_eq, n_terminal_ineq = constraint_sizes

    upper = np.concatenate([
        np.zeros(n_initial_eq),
        np.tile(np.concatenate([
            np.zeros(n_local_eq),
            np.full(n_local_ineq, np.inf),
        ]), self.N),
        np.zeros(n_terminal_eq),
        np.full(n_terminal_ineq, np.inf),
    ])
    lower = np.zeros(upper.size)
    return lower, upper

  def _make_constraints_jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
    constraint_sizes = self._constraint_sizes
    assert constraint_sizes is not None
    n_initial_eq, n_local_eq, n_local_ineq, n_terminal_eq, n_terminal_ineq = constraint_sizes
    n_local = n_local_eq + n_local_ineq
    n_terminal = n_terminal_eq + n_terminal_ineq

    rows = []
    cols = []
    if n_initial_eq:
      init_rows, init_cols = np.indices((n_initial_eq, self.nx))
      rows.append(init_rows.ravel())
      cols.append(init_cols.ravel())
    if n_local:
      local_rows, local_cols = np.indices((n_local, 2 * self.nx + self.nu))
      for k in range(self.N):
        rows.append(local_rows.ravel() + n_initial_eq + k * n_local)
        cols.append(local_cols.ravel() + k * (self.nx + self.nu))
    if n_terminal:
      terminal_rows, terminal_cols = np.indices((n_terminal, self.nx))
      rows.append(terminal_rows.ravel() + n_initial_eq + self.N * n_local)
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
