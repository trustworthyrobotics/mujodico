from dataclasses import replace

from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import ip_initializer
from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas import ip_search_dir
from mujodico.trajopt._src.ippas import IppasInitialGuess
from mujodico.trajopt._src.ippas import IppasOptions
from mujodico.trajopt._src.ippas import IppasSolver
from mujodico.trajopt._src.ippas import IppasStatus
from mujodico.trajopt._src.ippas.ip_primal_dual_system_test import _random_problem_data
from mujodico.trajopt._src.ippas.test_util import point_mass_model
from mujodico.trajopt._src.ippas.test_util import simple_model


class IpSearchDirTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())

  def test_stage_dual_shift(self):
    data = _random_problem_data(9, 3, 2, 1, 1, 2, 1, 'mixed', 'pd')
    data['Eu'] = jp.zeros_like(data['Eu'])
    system = pd.IpPrimalDualSystem.create(**data)
    options = IppasOptions()
    solved, good, _ = jax.jit(
        lambda p: ip_search_dir.compute(
            p, jp.array(0.1), jp.array(0.0), options
        )
    )(system)
    self.assertTrue(bool(good))
    self.assertAlmostEqual(float(solved.delta_c), 1e-6 * 0.1**0.25)
    full_rank = jp.ones_like(data['Eu'])
    self.assertEqual(
        float(
            ip_search_dir.dual_regularization(full_rank, jp.array(0.1), options)
        ),
        0.0,
    )

  def test_primal_and_dual_correction(self):
    data = _random_problem_data(19, 2, 2, 1, 1, 0, 0, 'mixed', 'pd')
    data['Eu'] = jp.zeros_like(data['Eu'])
    data['Huu'] = -jp.ones_like(data['Huu'])
    system = pd.IpPrimalDualSystem.create(**data)
    options = IppasOptions()
    solved, good, delta = jax.jit(
        lambda p: ip_search_dir.compute(
            p, jp.array(0.1), jp.array(0.0), options
        )
    )(system)
    self.assertTrue(bool(good))
    self.assertGreater(float(delta), 1.0)
    self.assertGreater(float(solved.delta_c), 0.0)

  def test_negative_curvature_is_regularized(self):
    m = simple_model(
        N=1,
        stage_cost=lambda x, u, k, p: jp.sum((u * u - 1) ** 2),
        terminal_cost=lambda x, p: jp.array(0.0),
        gauss_newton=False,
    )
    s = IppasSolver(m)
    p = jp.array([0.0, 0.0])
    m.initialize(p)
    it = ip_initializer.initialize(
        m,
        p,
        IppasInitialGuess(jp.zeros((2, 1)), jp.array([[0.2]])),
        ip_initializer.get_bounds(m, s.options),
        s.options,
    )
    sys, _ = pd.evaluate(
        m,
        it,
        p,
        jp.array(0.1),
        ip_initializer.get_bounds(m, s.options),
        s.options,
    )
    _, good, delta = jax.jit(
        lambda sys, mu, delta: ip_search_dir.compute(sys, mu, delta, s.options)
    )(sys, jp.array(0.1), jp.array(0.0))
    self.assertTrue(bool(good))
    self.assertGreater(float(delta), 3.5)
    result = jax.jit(s.solve)(p, IppasInitialGuess(it.x, it.u))
    self.assertEqual(int(result.status), IppasStatus.SUCCESS)
    np.testing.assert_allclose(result.us, 1.0, atol=1e-7)

  def _native_fixture(self):
    model, guess = point_mass_model()
    # Remove optional one-sided barrier damping when checking original-NLP
    # stationarity, as in the native test's zero dual-infeasibility assertions.
    options = IppasOptions(kappa_d=0.0, initialize_duals=False)
    bounds = ip_initializer.get_bounds(model, options)
    it = ip_initializer.initialize(model, None, guess, bounds, options)
    system, _ = pd.evaluate(model, it, None, jp.array(1.0), bounds, options)
    return model, bounds, it, system, options

  def test_solve_linear_system(self):
    _, _, _, system, options = self._native_fixture()
    direction, good, _ = jax.jit(
        lambda p: ip_search_dir.compute(
            p, jp.array(1.0), jp.array(0.0), options
        )
    )(system)
    self.assertTrue(bool(good))
    self.assertTrue(
        bool(
            ip_iterate.all_finite(
                tuple(getattr(direction, n) for n in pd._STEPS)
            )
        )
    )

  def test_update_iterate_and_check_infeasibility(self):
    model, bounds, it, system, options = self._native_fixture()
    direction, good, _ = jax.jit(
        lambda p: ip_search_dir.compute(
            p, jp.array(1.0), jp.array(0.0), options
        )
    )(system)
    self.assertTrue(bool(good))
    # Native test takes full primal and dual steps without line-search clipping.
    trial = replace(
        it,
        x=model.retract_state(it.x, direction.dx, 1.0, None),
        u=it.u + direction.du,
        s=it.s + direction.ds,
        sf=it.sf + direction.dsf,
        lmbda=it.lmbda + direction.dlmbda,
        beta=it.beta + direction.dbeta,
        y=it.y + direction.dy,
        yf=it.yf + direction.dyf,
        z_l=it.z_l + direction.dz_l,
        z_u=it.z_u + direction.dz_u,
        zf_l=it.zf_l + direction.dzf_l,
        zf_u=it.zf_u + direction.dzf_u,
    )
    _, data = pd.evaluate(
        model, trial, None, jp.array(1.0), bounds, options
    )
    self.assertLess(
        float(ip_iterate.max_abs(data.stationarity_x, data.stationarity_u)),
        1e-6,
    )
    self.assertLess(
        float(
            ip_iterate.max_abs(
                -trial.y - trial.z_l + trial.z_u,
                -trial.yf - trial.zf_l + trial.zf_u,
            )
        ),
        1e-6,
    )
    self.assertLess(
        float(
            ip_iterate.max_abs(data.r0, data.r_f, data.r_e, data.r_g, data.r_gf)
        ),
        1e-6,
    )
    # Complementarity is nonlinear: check its linearization at the saved iterate.
    residuals = pd.system_defect(direction)
    for name in ('r_c_l', 'r_c_u', 'r_cf_l', 'r_cf_u'):
      self.assertLess(
          float(ip_iterate.max_abs(residuals[name])), 1e-6, msg=name
      )


if __name__ == '__main__':
  absltest.main()
