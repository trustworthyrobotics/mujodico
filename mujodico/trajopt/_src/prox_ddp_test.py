from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
from mujodico import trajopt


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

    def implicit_dynamics(x, u, xnext):
      return xnext - (A @ x + B @ u)

    def stage_cost(x, u):
      return x.dot(Q @ x) + u.dot(R @ u)

    def terminal_cost(x):
      return horizon * x.dot(Q @ x)

    def stage_constraint(x, u):
      return jp.array([
          u[0] - u_max,
          -u[0] - u_max,
      ])

    def terminal_constraint(x):
      return jp.array([
          x[0] - 1e-3,
          -x[0] - 1e-3,
      ])

    solver = trajopt.ProxDDP(
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        implicit_dynamics=implicit_dynamics,
        stage_constraint=stage_constraint if has_stage_constraint else None,
        terminal_constraint=terminal_constraint if has_terminal_constraint else None,
        horizon=horizon,
        num_states=2,
        num_controls=1,
    )

    x, u = jax.jit(solver.solve)(x0)

    print('x =\n', x)
    print('u =\n', u)


if __name__ == "__main__":
  absltest.main()
