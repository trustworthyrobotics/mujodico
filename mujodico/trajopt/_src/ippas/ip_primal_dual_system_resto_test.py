"""FATROP restoration solve/refinement assertions and independent dense KKT."""

from dataclasses import replace

from absl.testing import absltest
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import ip_initializer as init
from mujodico.trajopt._src.ippas import ip_initializer_resto as initializer
from mujodico.trajopt._src.ippas import ip_primal_dual_system as original
from mujodico.trajopt._src.ippas import ip_primal_dual_system_resto as pd
from mujodico.trajopt._src.ippas import IppasInitialGuess, IppasOptions
from mujodico.trajopt._src.ippas.ip_iterate import damping, max_abs
from mujodico.trajopt._src.ippas.test_util import simple_model


class RestorationSystemTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())

  def fixture(self):
    m = simple_model(
        N=2,
        gauss_newton=False,
        dynamics=lambda x, u, k, p: x + u + 0.1 * u * u,
        stage_equality_constraint=lambda x, u, k, p: x + 0.3 * u * u - 0.7,
        stage_inequality_constraint=lambda x, u, k, p: u,
        stage_inequality_bounds=(np.array([-2.0]), np.array([2.0])),
        terminal_inequality_constraint=lambda x, p: x * x,
        terminal_inequality_bounds=(np.array([-0.5]), np.array([2.0])),
    )
    p = jp.array([0.0, 2.0])
    m.initialize(p)
    o = IppasOptions()
    bounds = init.get_bounds(m, o)
    base = init.initialize(
        m,
        p,
        IppasInitialGuess(
            jp.array([[0.1], [0.3], [0.5]]), jp.array([[0.2], [0.4]])
        ),
        bounds,
        o,
    )
    _, data = original.evaluate(m, base, p, jp.array(0.1), bounds, o)
    it, mu = initializer.initialize(base, data, jp.array(0.1), o)
    it = replace(
        it,
        base=replace(
            it.base,
            lmbda=jp.array([[0.2], [0.1], [-0.3]]),
            beta=jp.array([[0.1], [0.2]]),
            y=jp.array([[0.2], [-0.1]]),
            yf=jp.array([0.1]),
        ),
    )
    system, data = pd.evaluate(m, it, p, mu, bounds, o, base, jp.sqrt(0.1))
    return m, p, o, bounds, base, it, mu, system, data

  def test_full_newton_against_independent_dense_kkt(self):
    m, params, o, bounds, ref, it, mu, a, _ = self.fixture()
    shift = 0.03
    dc = 0.02
    a = replace(
        a,
        base=replace(
            a.base,
            Hxx=a.base.Hxx + shift * jp.eye(1),
            Huu=a.base.Huu + shift * jp.eye(1),
            delta_c=jp.array(dc),
        ),
        Hs=a.Hs + shift,
        Hsf=a.Hsf + shift,
        delta_w=jp.array(shift),
    )
    v = (it.base.x, it.base.u, it.base.s, it.base.sf, it.p, it.n)
    lam = (it.base.lmbda, it.base.beta, it.base.y, it.base.yf)
    z = (it.base.z_l, it.base.z_u, it.base.zf_l, it.base.zf_u, it.zp, it.zn)
    flat, unpack = ravel_pytree((v, lam, z))
    nv = ravel_pytree(v)[0].size

    def residual(flat):
      v, lam, z = unpack(flat)

      def constraints(v):
        x, u, s, sf, p, n = v
        return (
            m.initial_residual(x[0], params),
            m.dynamics_residual(x[:-1], u, x[1:], params),
            m.stage_equality_residual(x[:-1], u, params) + p.e - n.e,
            m.stage_inequality_residual(x[:-1], u, params) - s + p.g - n.g,
            m.terminal_inequality_residual(x[-1], params) - sf + p.gf - n.gf,
        )

      def lagrangian(v):
        x, u, s, sf, p, n = v
        b = replace(it.base, x=x, u=u, s=s, sf=sf)
        cost = jp.sqrt(0.1) * pd.proximity(
            m, b, ref
        ) + o.resto_penalty_parameter * sum(jp.sum(t) for t in (*p, *n))
        cost += jp.sum(damping(*bounds[:2], mu, o) * s) + jp.sum(
            damping(*bounds[2:], mu, o) * sf
        )
        cost += o.kappa_d * mu * sum(jp.sum(t) for t in (*p, *n))
        cons = constraints(v)
        return cost + sum(
            jp.vdot(c, l)
            for c, l in zip(cons, (lam[0][0], lam[0][1:], *lam[1:]))
        )

      grad = list(jax.grad(lagrangian)(v))
      grad[2] = grad[2] - z[0] + z[1]
      grad[3] = grad[3] - z[2] + z[3]
      grad[4] = pd.tree(lambda x, z: x - z, grad[4], z[4])
      grad[5] = pd.tree(lambda x, z: x - z, grad[5], z[5])
      x, u, s, sf, p, n = v
      cons = constraints(v)
      comp = (
          (s - bounds[0]) * z[0] - mu,
          (bounds[1] - s) * z[1] - mu,
          (sf - bounds[2]) * z[2] - mu,
          (bounds[3] - sf) * z[3] - mu,
          pd.tree(lambda p, z: p * z - mu, p, z[4]),
          pd.tree(lambda n, z: n * z - mu, n, z[5]),
      )
      return ravel_pytree((tuple(grad), cons, comp))[0]

    matrix = jax.jacfwd(residual)(flat)
    matrix = matrix.at[jp.arange(nv), jp.arange(nv)].add(shift)
    betaidx = nv + it.base.lmbda.size + jp.arange(it.base.beta.size)
    matrix = matrix.at[betaidx, betaidx].add(-dc)
    expected = jp.linalg.solve(matrix, -residual(flat))
    solved, good = jax.jit(
        pd.iterative_refinement_solve, static_argnums=(1, 2)
    )(a, 1e-10, 3)
    b = solved.base
    actual = ravel_pytree((
        (b.dx, b.du, b.ds, b.dsf, solved.dp, solved.dn),
        (b.dlmbda, b.dbeta, b.dy, b.dyf),
        (b.dz_l, b.dz_u, b.dzf_l, b.dzf_u, solved.dzp, solved.dzn),
    ))[0]
    self.assertTrue(bool(good))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    self.assertLess(
        float(max_abs(*jax.tree.leaves(pd.system_defect(solved)))), 1e-9
    )

  def test_raw_solve_cached_rhs_and_refinement(self):
    *_, a, data = self.fixture()
    solved = jax.jit(pd.solve)(a)
    self.assertLess(
        float(max_abs(*jax.tree.leaves(pd.system_defect(solved)))), 1e-9
    )
    changed = solved.replace_rhs(r_e=a.base.r_e + 0.2, r_g=a.base.r_g - 0.1)
    cached = pd.solve(changed)
    fresh = pd.solve(replace(changed, _cache=None))
    for field in original._STEPS:
      np.testing.assert_allclose(
          getattr(cached.base, field), getattr(fresh.base, field), atol=1e-10
      )
    self.assertIs(cached._cache.reduced._cache, solved._cache.reduced._cache)
    perturbed = replace(
        solved,
        base=replace(solved.base, dx=solved.base.dx + 0.01),
        dp=pd.tree(lambda v: v + 0.02, solved.dp),
    )
    refined, good = pd.iterative_refinement_solve(perturbed, 1e-10, 3)
    self.assertTrue(bool(good))
    np.testing.assert_allclose(refined.base.dx, solved.base.dx, atol=1e-9)

  def test_one_sided_and_empty_blocks(self):
    for constrained in (False, True):
      kwargs = (
          {}
          if not constrained
          else dict(
              stage_inequality_constraint=lambda x, u, k, p: u,
              stage_inequality_bounds=(np.array([-np.inf]), np.array([2.0])),
              terminal_inequality_constraint=lambda x, p: x,
              terminal_inequality_bounds=(np.array([0.0]), np.array([np.inf])),
          )
      )
      m = simple_model(N=1, **kwargs)
      p = jp.array([0.0, 2.0])
      m.initialize(p)
      o = IppasOptions()
      bounds = init.get_bounds(m, o)
      base = init.initialize(
          m, p, IppasInitialGuess(jp.zeros((2, 1)), jp.zeros((1, 1))), bounds, o
      )
      _, data = original.evaluate(m, base, p, jp.array(0.1), bounds, o)
      it, mu = initializer.initialize(base, data, jp.array(0.1), o)
      system, _ = pd.evaluate(m, it, p, mu, bounds, o, base, jp.sqrt(0.1))
      solved, good = jax.jit(
          lambda s: pd.iterative_refinement_solve(s, 1e-10, 3)
      )(system)
      self.assertTrue(bool(good))
      self.assertLess(
          float(max_abs(*jax.tree.leaves(pd.system_defect(solved)))), 1e-9
      )
      if constrained:
        np.testing.assert_array_equal(solved.base.dz_l, 0.0)
        np.testing.assert_array_equal(solved.base.dzf_u, 0.0)

  def test_return_multiplier_estimate_matches_joint_dense_projection(self):
    m, params, o, bounds, entry, *_ = self.fixture()
    point = replace(
        entry,
        x=entry.x + 0.01,
        u=entry.u + 0.05,
        s=entry.s + 0.03,
        sf=entry.sf + 0.04,
    )
    returned = initializer.initialize_original_multipliers(
        m,
        entry,
        point,
        params,
        jp.array(0.1),
        bounds,
        replace(o, bound_mult_reset_threshold=0),
    )
    v, unpack = ravel_pytree((returned.x, returned.u, returned.s, returned.sf))

    def constraints(v):
      x, u, s, sf = unpack(v)
      return ravel_pytree((
          m.initial_residual(x[0], params),
          m.dynamics_residual(x[:-1], u, x[1:], params),
          m.stage_equality_residual(x[:-1], u, params),
          m.stage_inequality_residual(x[:-1], u, params) - s,
          m.terminal_inequality_residual(x[-1], params) - sf,
      ))[0]

    jac = jax.jacfwd(constraints)(v)
    gx, gu = m.objective_gradient(returned.x, returned.u, params)
    gs = -returned.z_l + returned.z_u + damping(*bounds[:2], jp.array(0.1), o)
    gsf = (
        -returned.zf_l + returned.zf_u + damping(*bounds[2:], jp.array(0.1), o)
    )
    grad = ravel_pytree((gx, gu, gs, gsf))[0]
    expected = jp.linalg.solve(jac @ jac.T, -jac @ grad)
    actual = ravel_pytree(
        (returned.lmbda, returned.beta, returned.y, returned.yf)
    )[0]
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-9)

  def test_elastic_initialization_is_positive_and_stable(self):
    c = jp.array([-1e100, -1.0, 0.0, 1.0, 1e100])
    p, n = jax.jit(initializer.elastic_pair)(c, jp.array(1e-12), 1000.0)
    self.assertTrue(bool(jp.all(p > 0) & jp.all(n > 0)))
    np.testing.assert_allclose(p - n, -c, rtol=1e-15, atol=1e-24)

  def test_zero_objective_factor_skips_undefined_cost_jit_vmap(self):
    m = simple_model(
        N=1,
        gauss_newton=False,
        stage_cost=lambda x, u, k, p: jp.log(u[0]),
        terminal_cost=lambda x, p: jp.log(x[0]),
        stage_equality_constraint=lambda x, u, k, p: u * u,
    )
    p = jp.array([0.0, 2.0])
    m.initialize(p)

    def evaluate(value):
      return m.lagrangian_hessian(
          jp.full((2, 1), value),
          jp.full((1, 1), value),
          jp.zeros(1),
          jp.zeros((1, 1)),
          jp.ones((1, 1)),
          jp.zeros((1, 0)),
          jp.zeros(0),
          jp.zeros(0),
          0.0,
          p,
      )

    xx, xu, uu = jax.jit(jax.vmap(evaluate))(jp.array([-1.0, -0.5]))
    np.testing.assert_array_equal(xx, 0.0)
    np.testing.assert_allclose(uu, 2.0, atol=1e-12)


if __name__ == '__main__':
  absltest.main()
