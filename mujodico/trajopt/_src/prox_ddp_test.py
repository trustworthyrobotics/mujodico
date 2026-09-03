from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
from mujodico import trajopt
import numpy as np


class ProxDDPTest(parameterized.TestCase):

  @parameterized.parameters([
      (False, False),
      (True, True),
  ])
  def test_double_integrator(self, has_stage_constraint, has_terminal_constraint):
    dt = 0.02
    A = jp.array([[1, dt], [0, 1]])
    B = jp.array([[0.5 * dt**2], [dt]])

    Q = jp.diag(jp.array([1.0, 0.0]))
    R = jp.diag(jp.array([1e-5]))

    u_max = 10.0
    horizon = 100

    x0 = jp.array([-1.0, 0.0])
    params = (x0,)

    def implicit_dynamics(x, u, xnext, params=None):
      return xnext - (A @ x + B @ u)

    def initial_state_constraint(x, params):
      x0, = params
      return x - x0

    def stage_cost(x, u, params=None):
      return x.dot(Q @ x) + u.dot(R @ u)

    def terminal_cost(x, params=None):
      return horizon * x.dot(Q @ x)

    def stage_constraint(x, u, params=None):
      return jp.array([
          u[0] - u_max,
          -u[0] - u_max,
      ])

    def terminal_constraint(x, params=None):
      return jp.array([
          x[0] - 1e-3,
          -x[0] - 1e-3,
      ])

    solver = trajopt.ProxDDP(
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        implicit_dynamics=implicit_dynamics,
        initial_state_constraint=initial_state_constraint,
        stage_constraint=stage_constraint if has_stage_constraint else None,
        terminal_constraint=terminal_constraint if has_terminal_constraint else None,
        horizon=horizon,
        num_states=2,
        num_controls=1,
        params_example=params,
    )

    x, u = jax.jit(solver.solve)(params)

    tol = 1.2e-3

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
    if has_stage_constraint:
      np.testing.assert_array_less(
          jax.vmap(stage_constraint)(x[:-1], u),
          tol,
      )
    if has_terminal_constraint:
      np.testing.assert_array_less(
          terminal_constraint(x[-1]),
          tol,
      )


if __name__ == "__main__":
  absltest.main()
