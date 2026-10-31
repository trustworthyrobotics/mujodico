"""Complete compiled quaternion optimization and globalization regression."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mujodico.trajopt._src.fatrop_ref.convergence import Status
from mujodico.trajopt._src.fatrop_ref.manifold import Quaternion, quat_exp, quat_mul
from mujodico.trajopt._src.fatrop_ref.options import Options
from mujodico.trajopt._src.fatrop_ref.solver import InitialGuess, IppasSolver

jax.config.update('jax_enable_x64', True)


class ManifoldSolverTest(unittest.TestCase):

  def test_float32_callbacks_cannot_widen_compiled_solver_state(self):
    # Global x64 is enabled, so these literal arrays and static dynamics
    # intentionally have float64 outputs before the Problem precision cast.
    solver = IppasSolver(
        horizon=1,
        num_states=1,
        num_controls=1,
        options=Options(
            max_iter=30, tol=2e-5, constr_viol_tol=2e-5, linear_tol=1e-6
        ),
        initial_state=lambda p: p['x0'],
        dynamics=lambda x, u, p: jnp.asarray([[1.0]]) @ x
        + jnp.asarray([1.0]) * u[0],
        stage_cost=lambda x, u, p: (u[0] - p['target']) ** 2 + jnp.asarray(0.0),
        terminal_cost=lambda x, p: 0.0,
        stage_equality=lambda x, u, p: p['row_scale']
        * (jnp.asarray([1.0]) * u[0] - jnp.asarray([0.3])),
        stage_inequality=lambda x, u, p: jnp.asarray([1.0]),
        stage_inequality_bounds=(0.0, 2.0),
        terminal_equality=lambda x, p: jnp.zeros(0),
        terminal_inequality=lambda x, p: jnp.asarray([1.0]),
        terminal_inequality_bounds=(0.0, 2.0),
        x_bounds=(-5.0, 5.0),
        u_bounds=(-2.0, 2.0),
    )
    params = {
        'x0': jnp.asarray([0.0], jnp.float32),
        'target': jnp.asarray(0.3, jnp.float32),
        'row_scale': jnp.asarray(1.0, jnp.float32),
    }
    compiled = jax.jit(solver.solve)
    for scale in (1.0, 0.0):
      with self.subTest(equality_scale=scale):
        result = compiled(
            dict(params, row_scale=jnp.asarray(scale, jnp.float32))
        )
        self.assertIn(
            int(result.status),
            (Status.SUCCESS, Status.ACCEPTABLE),
            (result.status, result.error),
        )
        for leaf in jax.tree.leaves(result):
          if jnp.issubdtype(leaf.dtype, jnp.floating):
            self.assertEqual(leaf.dtype, jnp.float32)

  def test_compiled_rotation_trajectory_and_dynamic_params(self):
    manifold = Quaternion()
    solver = IppasSolver(
        horizon=2,
        num_states=4,
        num_controls=3,
        state_manifold=manifold,
        options=Options(
            max_iter=60,
            tol=1e-6,
            constr_viol_tol=1e-6,
            enable_restoration=False,
            enable_watchdog=False,
        ),
        initial_state=lambda params: params[0],
        dynamics=lambda q, u, params: quat_mul(q, quat_exp(u)),
        stage_cost=lambda q, u, params: 0.05 * jnp.dot(u, u),
        terminal_cost=lambda q, params: 2.5
        * jnp.dot(manifold.local(params[1], q), manifold.local(params[1], q)),
    )
    traces = []

    def run(params):
      traces.append(1)
      return solver.solve(params)

    compiled = jax.jit(run)
    q0 = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    params = jnp.stack([q0, quat_exp(jnp.asarray([0.3, -0.1, 0.2]))])
    result = compiled(params)
    self.assertIn(
        int(result.status),
        (Status.SUCCESS, Status.ACCEPTABLE),
        (result.status, result.error, result.iterations),
    )
    self.assertLess(float(result.error), 2e-6)
    np.testing.assert_allclose(
        jnp.linalg.norm(result.xs, axis=1), 1.0, atol=1e-13
    )
    self.assertLess(float(result.primal_error), 1e-6)
    new_q0 = quat_exp(jnp.asarray([0.04, 0.02, -0.01]))
    new_goal = quat_mul(new_q0, quat_exp(jnp.asarray([-0.2, 0.15, 0.1])))
    updated = compiled(jnp.stack([new_q0, new_goal]))
    self.assertEqual(len(traces), 1)
    self.assertIn(int(updated.status), (Status.SUCCESS, Status.ACCEPTABLE))
    np.testing.assert_allclose(updated.xs[0], new_q0, atol=1e-14)
    self.assertFalse(np.allclose(updated.us, result.us))
    np.testing.assert_allclose(
        jnp.linalg.norm(updated.xs, axis=1), 1.0, atol=1e-13
    )

  def test_soc_and_hard_restoration_use_manifold_updates(self):
    m = Quaternion()
    for hard in (False, True):
      with self.subTest(hard_restoration=hard):
        options = Options(
            max_iter=1,
            max_soc=0 if hard else 4,
            max_ls=1 if hard else 40,
            max_resto_iter=20,
            max_soft_resto_iters=0,
            enable_watchdog=False,
            enable_restoration=hard,
        )
        solver = IppasSolver(
            horizon=1,
            num_states=4,
            num_controls=2,
            state_manifold=m,
            options=options,
            initial_state=lambda p: jnp.asarray([1.0, 0.0, 0.0, 0.0]),
            dynamics=lambda q, u, p: quat_mul(
                q, quat_exp(jnp.asarray([u[0], 0.0, 0.0]))
            ),
            stage_cost=lambda q, u, p: u[0] + 10 * (jnp.dot(u, u) - 1),
            terminal_cost=lambda q, p: jnp.asarray(0.0, q.dtype),
            stage_equality=lambda q, u, p: jnp.atleast_1d(jnp.dot(u, u) - 1),
        )
        radius2 = 0.96 - 1e-4 if hard else 0.96
        guess = InitialGuess(us=jnp.asarray([[-jnp.sqrt(radius2), 0.2]]))
        result = jax.jit(solver.solve)(None, guess)
        self.assertEqual(int(result.trace.step_type[1]), 4 if hard else 2)
        if hard:
          self.assertGreater(int(result.trace.restoration_iterations[1]), 0)
          self.assertLess(
              float(result.trace.primal[1]), float(result.trace.primal[0])
          )
        else:
          self.assertGreater(int(result.trace.soc_steps[1]), 0)
        np.testing.assert_allclose(
            jnp.linalg.norm(result.xs, axis=1), 1.0, atol=1e-13
        )
        defect = m.local(
            result.xs[1], solver.dynamics(result.xs[0], result.us[0], None)
        )
        self.assertLess(float(jnp.linalg.norm(defect)), 1e-9)


if __name__ == '__main__':
  unittest.main()
