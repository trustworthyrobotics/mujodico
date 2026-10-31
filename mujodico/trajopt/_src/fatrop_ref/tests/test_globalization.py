"""Fixtures that execute SOC, watchdog, soft and hard restoration in solve."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mujodico.trajopt._src.fatrop_ref.options import Options
from mujodico.trajopt._src.fatrop_ref.solver import InitialGuess, IppasSolver

jax.config.update('jax_enable_x64', True)


def circle_solver(options):
  # On the circle the objective is just a. Away from it, the radial term
  # makes an uncorrected full Newton step increase the objective: Maratos.
  return IppasSolver(
      horizon=1,
      num_states=1,
      num_controls=2,
      options=options,
      initial_state=lambda p: jnp.zeros(1),
      dynamics=lambda x, u, p: x,
      stage_cost=lambda x, u, p: u[0] + 10 * (jnp.dot(u, u) - 1),
      terminal_cost=lambda x, p: jnp.asarray(0.0, x.dtype),
      stage_equality=lambda x, u, p: jnp.atleast_1d(jnp.dot(u, u) - 1),
  )


class GlobalizationTest(unittest.TestCase):

  def test_actual_solver_accepts_second_order_correction(self):
    solver = circle_solver(
        Options(
            max_iter=1,
            enable_restoration=False,
            max_soft_resto_iters=0,
            enable_watchdog=False,
        )
    )
    guess = InitialGuess(us=jnp.asarray([[-jnp.sqrt(0.96), 0.2]]))
    result = jax.jit(solver.solve)(None, guess)
    self.assertEqual(int(result.trace.step_type[1]), 2)
    self.assertGreater(int(result.trace.soc_steps[1]), 0)
    self.assertEqual(float(result.trace.alpha_primal[1]), 1.0)
    self.assertLess(float(result.objective), float(result.trace.objective[0]))

  def test_actual_solver_watchdog_rolls_back_checkpoint(self):
    solver = circle_solver(
        Options(
            max_iter=4,
            max_soc=0,
            enable_restoration=False,
            max_soft_resto_iters=0,
            watchdog_trigger=1,
            watchdog_max=1,
        )
    )
    guess = InitialGuess(us=jnp.asarray([[-jnp.sqrt(0.96), 0.2]]))
    result = jax.jit(solver.solve)(None, guess)
    kinds = np.asarray(result.trace.step_type)
    rolls = np.flatnonzero(kinds == 6)
    self.assertGreater(rolls.size, 0, kinds)
    for i in rolls:
      self.assertGreater(i, 0)
      self.assertAlmostEqual(
          float(result.trace.objective[i]),
          float(result.trace.objective[i - 1]),
          places=12,
      )
      self.assertAlmostEqual(
          float(result.trace.mu[i]), float(result.trace.mu[i - 1]), places=12
      )

  def test_actual_solver_uses_soft_restoration(self):
    solver = circle_solver(
        Options(
            max_iter=1,
            max_soc=0,
            max_ls=1,
            enable_restoration=False,
            enable_watchdog=False,
            max_soft_resto_iters=2,
        )
    )
    guess = InitialGuess(us=jnp.asarray([[-jnp.sqrt(0.96), 0.2]]))
    result = jax.jit(solver.solve)(None, guess)
    self.assertEqual(int(result.trace.step_type[1]), 3)
    self.assertLess(float(result.trace.error[1]), float(result.trace.error[0]))

  def test_actual_solver_enters_hard_restoration(self):
    solver = circle_solver(
        Options(
            max_iter=1,
            max_soc=0,
            max_ls=1,
            max_resto_iter=15,
            enable_restoration=True,
            enable_watchdog=False,
            max_soft_resto_iters=0,
        )
    )
    # Start slightly infeasible. The full objective step is still rejected,
    # while the elastic NLP can reduce the original violation.
    guess = InitialGuess(us=jnp.asarray([[-jnp.sqrt(0.96 - 1e-4), 0.2]]))
    result = jax.jit(solver.solve)(None, guess)
    self.assertEqual(int(result.trace.step_type[1]), 4)
    self.assertGreater(int(result.trace.restoration_iterations[1]), 0)
    self.assertLess(
        float(result.trace.primal[1]), float(result.trace.primal[0])
    )


if __name__ == '__main__':
  unittest.main()
