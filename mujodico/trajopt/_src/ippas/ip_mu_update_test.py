from absl.testing import absltest
import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_mu_update as g
from mujodico.trajopt._src.ippas import IppasOptions


class IpMuUpdateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())
    self.options = IppasOptions()

  def test_barrier_floor(self):
    o = self.options
    candidate = jax.jit(lambda mu: g._barrier_candidate(mu, o))
    self.assertAlmostEqual(float(candidate(jp.array(0.1))), 0.02)
    floor = min(o.tol, o.compl_inf_tol) / (o.barrier_tol_factor + 1)
    self.assertAlmostEqual(float(candidate(jp.array(floor))), floor)
    self.assertLessEqual(float(candidate(jp.array(floor / 2))), floor / 2)

  def test_recomputes_error_for_each_target(self):
    # This iterate is central at mu=0.1. It permits a reduction to 0.02, then
    # 0.002828..., whose fresh error 0.09717 exceeds 10*mu and stops the loop.
    result = jax.jit(
        lambda mu: g.update(
            mu, lambda target: jp.abs(0.1 - target), self.options
        )
    )(jp.array(0.1))
    self.assertAlmostEqual(float(result), 0.02**1.5)

  def test_unsolved_subproblem_keeps_mu(self):
    result = jax.jit(
        lambda mu: g.update(mu, lambda target: jp.array(2.0), self.options)
    )(jp.array(0.1))
    self.assertEqual(float(result), 0.1)

  def test_solved_subproblem_reaches_floor(self):
    result = jax.jit(
        lambda mu: g.update(mu, lambda target: jp.array(0.0), self.options)
    )(jp.array(0.1))
    floor = min(self.options.tol, self.options.compl_inf_tol) / (
        self.options.barrier_tol_factor + 1
    )
    self.assertAlmostEqual(float(result), floor)


if __name__ == '__main__':
  absltest.main()
