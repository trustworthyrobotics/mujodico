"""End-to-end regressions against analytic OCP solutions and KKT conditions."""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mujodico.trajopt._src.fatrop_ref import InitialGuess, IppasSolver, Options, Status

jax.config.update('jax_enable_x64', True)


def quadratic_solver(options=None, **extra):
  return IppasSolver(
      horizon=2,
      num_states=1,
      num_controls=1,
      initial_state=lambda p: p['x0'],
      dynamics=lambda x, u, p: x + u,
      stage_cost=lambda x, u, p: 0.5 * jnp.sum(u * u),
      terminal_cost=lambda x, p: 0.5 * jnp.sum((x - p['goal']) ** 2),
      options=options
      or Options(max_iter=60, tol=1e-8, enable_restoration=False),
      **extra
  )


def parameters(x0=0.0, goal=3.0):
  return {'x0': jnp.array([x0]), 'goal': jnp.array([goal])}


def guess():
  return InitialGuess(jnp.zeros((3, 1)), jnp.zeros((2, 1)))


class SolverTests(unittest.TestCase):

  def assert_converged(self, result, tolerance=3e-6):
    self.assertIn(int(result.status), (Status.SUCCESS, Status.ACCEPTABLE))
    self.assertLess(float(result.primal_error), tolerance)
    self.assertLess(float(result.dual_error), tolerance)
    self.assertLess(float(result.complementarity), tolerance)
    self.assertTrue(np.all(np.isfinite(np.asarray(result.xs))))

  def test_jitted_two_stage_quadratic_matches_analytic_solution(self):
    solver = quadratic_solver()
    result = jax.jit(solver.solve)(parameters(), guess())
    self.assert_converged(result)
    np.testing.assert_allclose(result.us, [[1.0], [1.0]], atol=2e-6)
    np.testing.assert_allclose(result.xs, [[0.0], [1.0], [2.0]], atol=2e-6)
    np.testing.assert_allclose(result.objective, 1.5, atol=2e-6)
    self.assertEqual(
        int(jnp.sum(result.trace.valid)), int(result.iterations) + 1
    )
    self.assertTrue(
        bool(
            jnp.all(
                jnp.isfinite(
                    jnp.where(result.trace.valid, result.trace.error, 0.0)
                )
            )
        )
    )

  def test_dynamic_initial_state_and_goal_reuse_one_compilation(self):
    solver = quadratic_solver()
    traces = []

    def solve(params, initial):
      traces.append(None)
      return solver.solve(params, initial)

    compiled = jax.jit(solve)
    first = compiled(parameters(0.0, 3.0), guess())
    second = compiled(parameters(1.0, 2.5), guess())
    self.assert_converged(first)
    self.assert_converged(second)
    self.assertEqual(len(traces), 1)
    np.testing.assert_allclose(second.us, [[0.5], [0.5]], atol=2e-6)
    np.testing.assert_allclose(second.xs, [[1.0], [1.5], [2.0]], atol=2e-6)

  def test_serial_and_parallel_solutions_and_traces_agree(self):
    common = Options(max_iter=60, tol=1e-8, enable_restoration=False)
    serial = quadratic_solver(
        dataclasses.replace(common, linear_backend='serial')
    )
    parallel = quadratic_solver(
        dataclasses.replace(common, linear_backend='parallel')
    )
    a = jax.jit(serial.solve)(parameters(), guess())
    b = jax.jit(parallel.solve)(parameters(), guess())
    self.assert_converged(a)
    self.assert_converged(b)
    self.assertEqual(int(a.iterations), int(b.iterations))
    for left, right in (
        (a.xs, b.xs),
        (a.us, b.us),
        (a.trace.objective, b.trace.objective),
        (a.trace.error, b.trace.error),
    ):
      np.testing.assert_allclose(left, right, atol=2e-8, rtol=2e-8)

  def test_native_boxes_and_nonlinear_path_bound_reach_active_solution(self):
    solver = quadratic_solver(
        u_bounds=(0.0, 0.5),
        x_bounds=(-1.0, 2.0),
        stage_inequality=lambda x, u, p: u * u,
        stage_inequality_bounds=(-np.inf, 0.04),
    )
    result = jax.jit(solver.solve)(parameters(), guess())
    self.assert_converged(result)
    np.testing.assert_allclose(result.us, [[0.2], [0.2]], atol=2e-5)
    self.assertTrue(np.all(np.asarray(result.us) ** 2 <= 0.04 + 2e-6))
    self.assertTrue(np.all(np.asarray(result.zl.us) >= 0.0))
    self.assertTrue(np.all(np.asarray(result.zu.us) >= 0.0))

  def test_state_only_running_and_terminal_equalities_with_terminal_inequality(
      self,
  ):
    solver = quadratic_solver(
        u_bounds=(-0.6, 0.6),
        stage_equality=lambda x, u, p: x,
        terminal_equality=lambda x, p: x - 0.5,
        stage_inequality=lambda x, u, p: u * u,
        stage_inequality_bounds=(-np.inf, 1.0),
        terminal_inequality=lambda x, p: x * x,
        terminal_inequality_bounds=(-np.inf, 1.0),
    )
    result = jax.jit(solver.solve)(parameters(), guess())
    self.assert_converged(result)
    np.testing.assert_allclose(result.xs, [[0.0], [0.0], [0.5]], atol=3e-6)
    np.testing.assert_allclose(result.us, [[0.0], [0.5]], atol=3e-6)
    residual = solver.evaluate(result.primal, parameters())
    self.assertLess(float(jnp.max(jnp.abs(residual.eq), initial=0.0)), 3e-6)
    self.assertLess(float(jnp.max(jnp.abs(residual.eqT), initial=0.0)), 3e-6)

  def test_fixed_initial_state_outside_native_bounds_reports_infeasible(self):
    solver = quadratic_solver(x_bounds=(-1.0, 1.0))
    result = jax.jit(solver.solve)(parameters(2.0, 3.0), guess())
    self.assertEqual(int(result.status), Status.INFEASIBLE)
    self.assertEqual(int(result.iterations), 0)

  def test_nonfinite_initial_state_reports_numerical_error_without_iterations(
      self,
  ):
    solver = quadratic_solver()
    result = jax.jit(solver.solve)(parameters(np.nan, 3.0), guess())
    self.assertEqual(int(result.status), Status.NUMERICAL_ERROR)
    self.assertEqual(int(result.iterations), 0)


if __name__ == '__main__':
  unittest.main()
