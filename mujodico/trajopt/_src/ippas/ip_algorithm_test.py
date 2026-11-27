from dataclasses import replace

from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np
from scipy import optimize

from mujodico.trajopt._src import manifold
from mujodico.trajopt._src.ippas import IppasInitialGuess
from mujodico.trajopt._src.ippas import IppasOptions
from mujodico.trajopt._src.ippas import IppasSolver
from mujodico.trajopt._src.ippas import IppasStatus
from mujodico.trajopt._src.ippas.test_util import simple_model
from mujodico.trajopt._src.ippas_trajopt import IppasTrajOpt
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel


class IpAlgorithmTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())

  def test_jit_dynamic_parameters_and_vmap(self):
    solver = IppasSolver(simple_model())
    compiled = jax.jit(solver.solve)
    xs, us = jp.zeros((4, 1)), jp.zeros((3, 1))
    for p in (jp.array([0.0, 2.0]), jp.array([0.5, -1.0])):
      with jax.checking_leaks():
        r = compiled(p, IppasInitialGuess(xs, us))
      self.assertEqual(int(r.status), IppasStatus.SUCCESS)
      expected = (p[1] - p[0]) / 3.1
      np.testing.assert_allclose(r.us, expected, atol=1e-9)
      self.assertLess(float(r.error), 1e-8)
    ps = jp.array([[0.0, 2.0], [0.5, -1.0]])
    batch = jax.jit(
        jax.vmap(lambda p: solver.solve(p, IppasInitialGuess(xs, us)))
    )(ps)
    np.testing.assert_array_equal(
        batch.status, [IppasStatus.SUCCESS, IppasStatus.SUCCESS]
    )
    np.testing.assert_allclose(
        batch.us[:, 0, 0], (ps[:, 1] - ps[:, 0]) / 3.1, atol=1e-9
    )
    fresh = IppasSolver(simple_model())
    with jax.checking_leaks():
      cold_batch = jax.jit(
          jax.vmap(lambda p: fresh.solve(p, IppasInitialGuess(xs, us)))
      )(ps)
    np.testing.assert_allclose(cold_batch.us, batch.us, atol=1e-9)

  def test_active_boxes_and_terminal_inequalities(self):
    model = simple_model(
        u_bounds=(np.array([-0.3]), np.array([0.6])),
        terminal_inequality_constraint=lambda x, p: jp.concatenate((x, x)),
        terminal_inequality_bounds=(
            np.array([-np.inf, 0.0]),
            np.array([1.2, np.inf]),
        ),
    )
    r = jax.jit(IppasSolver(model, IppasOptions(bound_relax_factor=0.0)).solve)(
        jp.array([0.0, 2.0]),
        IppasInitialGuess(jp.zeros((4, 1)), jp.zeros((3, 1))),
    )
    self.assertEqual(int(r.status), IppasStatus.SUCCESS)
    np.testing.assert_allclose(r.us, 0.4, atol=1e-7)
    np.testing.assert_allclose(r.xs[-1], 1.2, atol=1e-7)
    self.assertTrue(
        np.all(np.asarray(r.iterate.s) > model.stage_inequality_bounds[0])
    )
    self.assertTrue(
        np.all(np.asarray(r.iterate.sf) < model.terminal_inequality_bounds[1])
    )
    self.assertLess(float(r.error), 1e-8)

  def test_primal_dual_warm_start(self):
    solver = IppasSolver(
        simple_model(u_bounds=(np.array([-1.0]), np.array([1.0])))
    )
    r = jax.jit(solver.solve)(
        jp.array([0.0, 2.0]),
        IppasInitialGuess(jp.zeros((4, 1)), jp.zeros((3, 1))),
    )
    warm = jax.jit(solver.solve)(
        jp.array([0.0, 2.0]), IppasInitialGuess(r.xs, r.us, r.iterate)
    )
    self.assertEqual(int(warm.status), IppasStatus.SUCCESS)
    self.assertEqual(int(warm.iterations), 0)
    invalid = replace(r.iterate, z_l=-jp.ones_like(r.iterate.z_l))
    bad = jax.jit(solver.solve)(
        jp.array([0.0, 2.0]), IppasInitialGuess(invalid.x, invalid.u, invalid)
    )
    self.assertEqual(int(bad.status), IppasStatus.INVALID_INITIAL_GUESS)

  def test_relaxed_kkt_and_original_bound_feasibility_are_distinct(self):
    model = simple_model(
        N=1,
        terminal_inequality_constraint=lambda x, p: x,
        terminal_inequality_bounds=(np.array([-np.inf]), np.array([2.0])),
    )
    solver = IppasSolver(model)
    result = jax.jit(solver.solve)(
        jp.array([0.0, 10.0]),
        IppasInitialGuess(jp.zeros((2, 1)), jp.zeros((1, 1))),
    )
    self.assertEqual(int(result.status), IppasStatus.SUCCESS)
    self.assertGreater(float(result.primal_error), solver.options.tol)
    self.assertLess(float(result.primal_error), solver.options.constr_viol_tol)
    self.assertLess(float(result.error), solver.options.tol)
    self.assertLess(float(result.xs[-1, 0]), 2.0 + 2e-8)

  def test_nonlinear_exact_hessian_against_scipy(self):
    solver = IppasSolver(
        simple_model(
            N=1,
            dynamics=lambda x, u, k, p: x + u + 0.1 * u**2,
            gauss_newton=False,
        )
    )
    r = jax.jit(solver.solve)(
        jp.array([0.2, 1.4]),
        IppasInitialGuess(jp.zeros((2, 1)), jp.zeros((1, 1))),
    )
    expected = optimize.minimize_scalar(
        lambda u: 0.5 * u * u + 5 * (0.2 + u + 0.1 * u * u - 1.4) ** 2
    )
    self.assertEqual(int(r.status), IppasStatus.SUCCESS)
    np.testing.assert_allclose(r.us[0, 0], expected.x, atol=1e-7)

  def test_quaternion_retractions(self):
    m = StagewiseOcpModel(
        2,
        4,
        3,
        initial_state=lambda p: jp.array([1.0, 0.0, 0.0, 0.0]),
        dynamics=lambda x, u, k, p: manifold.quat_multiply(
            x, manifold.quat_exp(u)
        ),
        stage_cost=lambda x, u, k, p: 0.5 * jp.sum(u * u),
        terminal_cost=lambda x, p: 5 * jp.sum((manifold.quat_log(x) - p) ** 2),
        quaternion_addrs=(0,),
        full_dynamics_jacobian=True,
    )
    xs = jp.tile(jp.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
    r = jax.jit(IppasSolver(m).solve)(
        jp.array([0.3, 0.0, 0.0]), IppasInitialGuess(xs, jp.zeros((2, 3)))
    )
    self.assertEqual(int(r.status), IppasStatus.SUCCESS)
    np.testing.assert_allclose(r.us[:, 0], 3 / 21, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(r.xs, axis=1), 1, atol=1e-12)

  def test_stage_equalities_and_state_only_constraints(self):
    m = simple_model(
        nu=2,
        dynamics=lambda x, u, k, p: x + u[:1],
        stage_equality_constraint=lambda x, u, k, p: u[1:] - 2 * u[:1],
    )
    r = jax.jit(IppasSolver(m).solve)(
        jp.array([0.0, 2.0]),
        IppasInitialGuess(jp.zeros((4, 1)), jp.zeros((3, 2))),
    )
    self.assertEqual(int(r.status), IppasStatus.SUCCESS)
    np.testing.assert_allclose(r.us[:, 1], 2 * r.us[:, 0], atol=1e-10)
    m = simple_model(stage_equality_constraint=lambda x, u, k, p: x)
    r = jax.jit(IppasSolver(m).solve)(
        jp.array([0.0, 2.0]),
        IppasInitialGuess(jp.zeros((4, 1)), jp.zeros((3, 1))),
    )
    self.assertEqual(int(r.status), IppasStatus.SUCCESS)
    np.testing.assert_allclose(r.xs[:-1], 0.0, atol=1e-8)
    np.testing.assert_allclose(r.us[-1], 20 / 11, atol=1e-7)

  def test_redundant_state_equalities_jit_vmap_and_warm_start(self):
    m = simple_model(
        stage_equality_constraint=lambda x, u, k, p: jp.concatenate((x, 2 * x))
    )
    solver = IppasSolver(m)
    guess = IppasInitialGuess(jp.zeros((4, 1)), jp.zeros((3, 1)))
    ps = jp.array([[0.0, 2.0], [0.0, -1.0]])
    results = jax.jit(jax.vmap(lambda p: solver.solve(p, guess)))(ps)
    np.testing.assert_array_equal(results.status, IppasStatus.SUCCESS)
    np.testing.assert_allclose(results.xs[:, :-1], 0.0, atol=1e-8)
    np.testing.assert_allclose(
        results.us[:, -1, 0], ps[:, 1] * 10 / 11, atol=1e-7
    )
    r = jax.tree.map(lambda v: v[0], results)
    warm = jax.jit(solver.solve)(
        ps[0], IppasInitialGuess(r.xs, r.us, r.iterate)
    )
    self.assertEqual(int(warm.status), IppasStatus.SUCCESS)
    self.assertEqual(int(warm.iterations), 0)

  def test_dual_shift_failure_preserves_iterate(self):
    solver = IppasSolver(
        simple_model(stage_equality_constraint=lambda x, u, k, p: x),
        IppasOptions(kappa_c=1e4, max_soc=0),
    )
    guess = IppasInitialGuess(jp.zeros((4, 1)), jp.zeros((3, 1)))
    result = jax.jit(solver.solve)(jp.array([0.0, 2.0]), guess)
    self.assertEqual(int(result.status), IppasStatus.NUMERICAL_ERROR)
    np.testing.assert_array_equal(result.xs, guess.xs)
    np.testing.assert_array_equal(result.us, guess.us)
    for name in ('delta_c_stripe', 'kappa_c'):
      for value in (0.0, -1.0, float('nan'), float('inf')):
        with self.assertRaises(ValueError):
          IppasOptions(**{name: value})

  def test_wrapper_jit_returns_arrays_and_warm_state(self):
    m = simple_model()
    facade = IppasTrajOpt(
        3,
        1,
        1,
        initial_state=m._user_initial_state_fn,
        dynamics=m._user_dynamics_fn,
        stage_cost=m._user_stage_cost_fn,
        terminal_cost=m._user_terminal_cost_fn,
    )
    compiled = jax.jit(facade.solve)
    xs, us, info = compiled(
        jp.array([0.0, 2.0]), jp.zeros((4, 1)), jp.zeros((3, 1))
    )
    self.assertTrue(bool(info['success']))
    self.assertIsInstance(xs, jax.Array)
    warm = jax.jit(lambda p, x, u, w: facade.solve(p, x, u, warm_start=w))
    _, _, second = warm(jp.array([0.0, 2.0]), xs, us, info['warm_start'])
    self.assertEqual(int(second['iter_count']), 0)

  def test_max_iter_and_rejections(self):
    solver = IppasSolver(simple_model(), IppasOptions(max_iter=0))
    r = jax.jit(solver.solve)(
        jp.array([0.0, 2.0]),
        IppasInitialGuess(jp.zeros((4, 1)), jp.zeros((3, 1))),
    )
    self.assertEqual(int(r.status), IppasStatus.MAX_ITER)
    with self.assertRaisesRegex(ValueError, 'terminal equalities'):
      IppasSolver(
          simple_model(terminal_equality_constraint=lambda x, p: x)
      ).solve(
          jp.array([0.0, 2.0]),
          IppasInitialGuess(jp.zeros((4, 1)), jp.zeros((3, 1))),
      )
    with self.assertRaisesRegex(ValueError, 'N >= 1'):
      IppasSolver(simple_model(N=0))
    with jax.enable_x64(False):
      with self.assertRaisesRegex(ValueError, 'float64'):
        solver.solve(
            jp.array([0.0, 2.0]),
            IppasInitialGuess(jp.zeros((4, 1)), jp.zeros((3, 1))),
        )

  def test_nonfinite_trial_failure_preserves_iterate(self):
    m = simple_model(
        N=1,
        stage_cost=lambda x, u, k, p: jp.where(
            u[0] == 0.0, (u[0] - 1) ** 2, jp.nan
        ),
        terminal_cost=lambda x, p: jp.array(0.0),
    )
    s = IppasSolver(
        m,
        IppasOptions(
            max_soc=0,
            restoration_enabled=False,
            soft_rest_pd_error_reduction_factor=0,
        ),
    )
    result = jax.jit(s.solve)(
        jp.array([0.0, 0.0]),
        IppasInitialGuess(jp.zeros((2, 1)), jp.zeros((1, 1))),
    )
    self.assertEqual(int(result.status), IppasStatus.LINE_SEARCH_FAILED)
    np.testing.assert_array_equal(result.us, 0.0)


if __name__ == '__main__':
  absltest.main()
