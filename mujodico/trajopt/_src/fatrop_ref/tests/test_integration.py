"""Real example adapter and solver, with an analytically soluble transcription."""

import contextlib
import io
from types import SimpleNamespace
from typing import NamedTuple
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from mujodico.examples.methods import ippas_lifted_multiple_shooting as adapter

jax.config.update('jax_enable_x64', True)


class Params(NamedTuple):
  x_0: object
  goal: object


class AnalyticTranscription:

  def __init__(self, task):
    del task
    self.N, self.nx, self.nw = 2, 1, 2
    self.params = Params(jnp.array([1.0]), jnp.array([4.0]))
    self.x_bounds = ([-10.0], [10.0])
    self.w_bounds = ([-2.0, -4.0], [2.0, 4.0])
    self.stage_cost = lambda x, w, p: 0.5 * w[0] ** 2
    self.terminal_cost = lambda x, p: 0.5 * jnp.sum((x - p.goal) ** 2)
    self.dynamics = lambda x, w: x + w[:1]
    self.stage_equality = lambda x, w: jnp.atleast_1d(w[1] - 2 * w[0])
    self.stage_inequality = lambda x, w: jnp.zeros(0, x.dtype)


class IntegrationTest(unittest.TestCase):

  def test_lifted_adapter_returns_forces_and_reuses_parameterized_solver(self):
    with mock.patch.object(
        adapter, 'LiftedMultipleShootingTranscription', AnalyticTranscription
    ):
      method = adapter.IppasLiftedMultipleShooting(
          SimpleNamespace(quaternion_w_idxs=[])
      )
    guess = {'xs': np.zeros((3, 1)), 'us': np.zeros((2, 1))}
    with contextlib.redirect_stdout(io.StringIO()):
      first = method.solve(guess, iterations=50)
      cached = next(iter(method._solver_cache.values()))
      method.transcription.params = Params(jnp.array([0.0]), jnp.array([2.0]))
      second = method.solve(guess, iterations=50)
    self.assertEqual(len(method._solver_cache), 1)
    self.assertIs(cached, next(iter(method._solver_cache.values())))
    np.testing.assert_allclose(first['us'], 1.0, atol=3e-5)
    np.testing.assert_allclose(first['cfs'], 2.0, atol=3e-5)
    np.testing.assert_allclose(second['us'], 2 / 3, atol=3e-5)
    np.testing.assert_allclose(second['cfs'], 4 / 3, atol=3e-5)
    np.testing.assert_array_equal(first['xs'][0], [1.0])
    np.testing.assert_array_equal(second['xs'][0], [0.0])


if __name__ == '__main__':
  unittest.main()
