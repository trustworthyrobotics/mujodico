from absl.testing import absltest
from absl.testing import parameterized
from functools import partial
from itertools import product
import jax
import jax.numpy as jp
from mujodico import trajopt
import numpy as np


class ProxDDPTest(parameterized.TestCase):

  @parameterized.parameters(
      *product(
        [False, True],
        [False, True],
        [False, True],
      )
  )
  def test_double_integrator(self, ineq_constraint, lifted_dynamics, parallel):
    dt = 0.02
    a_max = 10.0
    horizon = 100

    x0 = jp.array([-1.0, 0.0])

    Q = jp.diag(jp.array([1.0, 0.0]))
    R = jp.diag(jp.array([1e-5, 0.0] if lifted_dynamics else [1e-5]))

    def implicit_dynamics1(x, u, xnext, params=None):
      q = x[0]
      v = x[1]
      a = u[0]
      return jp.array([
          xnext[1] - (v + dt * a),
          xnext[0] - (q + dt * v + 0.5 * dt**2 * a)
      ])

    def implicit_dynamics2(x, u, xnext, params=None):
      q = x[0]
      v = x[1]
      a = u[0]
      vnext = u[1]
      return jp.array([
          vnext - (v + dt * a),
          xnext[1] - vnext,
          xnext[0] - (q + dt * vnext)
      ])

    implicit_dynamics = implicit_dynamics2 if lifted_dynamics else implicit_dynamics1

    params = (x0,)

    def initial_state_constraint(x, params):
      x0, = params
      return x - x0

    def stage_cost(x, u, params=None):
      return x.dot(Q @ x) + u.dot(R @ u)

    def terminal_cost(x, params=None):
      return horizon * x.dot(Q @ x)

    def stage_constraint(x, u, params=None):
      a = u[0]
      return jp.array([
          a - a_max,
          -a - a_max,
      ])

    def terminal_constraint(x, params=None):
      q = x[0]
      return jp.array([
          q - 1e-3,
          -q - 1e-3,
      ])

    solver = trajopt.ProxDDP(
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        implicit_dynamics=implicit_dynamics,
        initial_state_constraint=initial_state_constraint,
        stage_constraint=stage_constraint if ineq_constraint else None,
        terminal_constraint=terminal_constraint if ineq_constraint else None,
        horizon=horizon,
        num_states=Q.shape[0],
        num_controls=R.shape[0],
        params_example=params,
    )

    options = trajopt.ProxDDP.Options(
        parallel_associative_scan=parallel,
    )

    solve = jax.jit(solver.solve, static_argnames=('options',))
    x, u = solve(params, options=options)

    tol = 1e-4

    np.testing.assert_allclose(
        initial_state_constraint(x[0], params),
        0.0,
        rtol=tol, atol=tol,
    )
    np.testing.assert_allclose(
        jax.vmap(implicit_dynamics)(x[:-1], u, x[1:]),
        0.0,
        rtol=tol, atol=tol,
    )
    if ineq_constraint:
      np.testing.assert_array_less(
          jax.vmap(stage_constraint)(x[:-1], u),
          tol,
      )
      np.testing.assert_array_less(
          terminal_constraint(x[-1]),
          tol,
      )


if __name__ == "__main__":
  absltest.main()
