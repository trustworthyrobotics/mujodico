"""Elastic feasibility restoration and manifold proximity regressions."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mujodico.trajopt._src.fatrop_ref.filter import empty_filter
from mujodico.trajopt._src.fatrop_ref.initialization import initialize
from mujodico.trajopt._src.fatrop_ref.manifold import CustomManifold, Quaternion, quat_exp, quat_mul
from mujodico.trajopt._src.fatrop_ref.options import Options
from mujodico.trajopt._src.fatrop_ref.problem import Problem
from mujodico.trajopt._src.fatrop_ref.restoration import RestorationParams, RestorationProblem, elastic_seed
from mujodico.trajopt._src.fatrop_ref.restoration import perform_restoration as _hard_restore

jax.config.update('jax_enable_x64', True)


class RestorationTest(unittest.TestCase):

  def test_seed_is_positive_and_elastic_equalities_are_exact(self):
    problem = Problem(
        horizon=1,
        num_states=1,
        num_controls=1,
        initial_state=lambda p: jnp.asarray([0.2]),
        dynamics=lambda x, u, p: x + u,
        stage_cost=lambda x, u, p: 0.5 * jnp.dot(u, u),
        terminal_cost=lambda x, p: 0.5 * jnp.dot(x, x),
        stage_equality=lambda x, u, p: jnp.asarray([u[0] - 0.6]),
        stage_inequality=lambda x, u, p: jnp.asarray([x[0] + u[0]]),
        stage_inequality_bounds=([0.0], [1.0]),
        terminal_equality=lambda x, p: jnp.asarray([x[0] - 0.8]),
        terminal_inequality=lambda x, p: jnp.asarray([x[0] ** 2]),
        terminal_inequality_bounds=([0.1], [2.0]),
    )
    q = problem.initialize(None, us=jnp.asarray([[0.1]]))
    q = q._replace(s=q.s + 0.2, st=q.st - 0.15)
    opts = Options()

    def seeded(primal, mu):
      rp = RestorationProblem(problem, primal, None, opts)
      seed = rp.seed(primal, None, mu)
      ev = rp.evaluate(seed, RestorationParams(None, primal))
      lower, upper = rp.bounds(seed, RestorationParams(None, primal))
      return seed, ev, lower, upper, rp.project(seed)

    seed, ev, lower, upper, projected = jax.jit(seeded)(q, jnp.asarray(0.1))
    np.testing.assert_allclose(ev.eq, 0.0, atol=1e-14)
    np.testing.assert_allclose(ev.eqT, 0.0, atol=1e-14)
    self.assertTrue(np.all(np.asarray(seed.us[:, 1:]) > 0.0))
    self.assertTrue(np.all(np.asarray(seed.ut) > 0.0))
    np.testing.assert_allclose(lower.us[:, 1:], 0.0)
    for a, b in zip(projected, q):
      np.testing.assert_allclose(a, b)
    p, n = elastic_seed(jnp.asarray([-100.0, 0.0, 100.0]), 0.1, 1000.0)
    self.assertTrue(bool(jnp.all((p > 0) & (n > 0))))
    np.testing.assert_allclose(
        jnp.asarray([-100.0, 0.0, 100.0]) + p - n, 0.0, atol=1e-14
    )

  def test_manifold_proximity_gradient_and_hessian(self):
    m = Quaternion()
    problem = Problem(
        horizon=1,
        num_states=4,
        num_controls=3,
        state_manifold=m,
        initial_state=lambda p: jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        dynamics=lambda q, u, p: quat_mul(q, quat_exp(0.2 * u)),
        stage_cost=lambda x, u, p: 0.5 * jnp.dot(u, u),
        terminal_cost=lambda x, p: jnp.asarray(0.0, x.dtype),
    )
    ref = problem.initialize(None, us=jnp.asarray([[0.1, -0.2, 0.1]]))
    current = ref._replace(
        xs=ref.xs.at[-1].set(quat_exp(jnp.asarray([0.4, -0.1, 0.2]))),
        us=ref.us + jnp.asarray([[0.2, -0.1, 0.1]]),
    )
    rp = RestorationProblem(problem, current, None, Options(zeta=2.3))
    p = RestorationParams(None, ref)
    e = rp.evaluate(current, p)
    system, rhs = jax.jit(rp.linearize)(
        current, jnp.zeros_like(e.eq), jnp.zeros_like(e.eqT), p
    )

    def objective(delta):
      q = current._replace(
          xs=current.xs.at[-1].set(m.retract(current.xs[-1], delta))
      )
      return rp.evaluate(q, p).objective

    zero = jnp.zeros(3)
    np.testing.assert_allclose(rhs.gTx, jax.grad(objective)(zero), atol=1e-12)
    np.testing.assert_allclose(
        system.QT, jax.hessian(objective)(zero), atol=1e-12
    )
    direction = jnp.asarray([0.3, -0.5, 0.2])
    eps = 2e-4
    f0, fp, fm = (
        objective(zero),
        objective(eps * direction),
        objective(-eps * direction),
    )
    np.testing.assert_allclose(
        (fp - fm) / (2 * eps), rhs.gTx @ direction, atol=1e-8
    )
    np.testing.assert_allclose(
        (fp - 2 * f0 + fm) / (eps * eps),
        direction @ system.QT @ direction,
        atol=1e-7,
    )
    np.testing.assert_allclose(system.R[0, :3, :3], 2.3 * np.eye(3), atol=1e-13)
    np.testing.assert_allclose(
        rhs.gy[0, :3], 2.3 * (current.us - ref.us)[0], atol=1e-13
    )
    cut = current._replace(
        xs=current.xs.at[-1].set(
            m.retract(ref.xs[-1], jnp.asarray([0.0, jnp.pi, 0.0]))
        ),
        us=jnp.asarray([[0.1, 0.7, 0.1]]),
    )
    self.assertTrue(bool(Problem.valid(rp, cut, p)))
    self.assertFalse(bool(jax.jit(rp.valid)(cut, p)))

  def test_user_manifold_callbacks_receive_original_params(self):
    manifold = CustomManifold(
        1,
        1,
        lambda x, d, p: x + p['scale'] * d,
        lambda a, b, p: (b - a) / p['scale'],
    )
    problem = Problem(
        horizon=1,
        num_states=1,
        num_controls=1,
        state_manifold=manifold,
        initial_state=lambda p: p['x0'],
        dynamics=lambda x, u, p: x + u,
        stage_cost=lambda x, u, p: jnp.dot(u, u),
        terminal_cost=lambda x, p: 0.0,
    )
    params = {'x0': jnp.zeros(1), 'scale': jnp.asarray(2.0)}
    ref = problem.initialize(params, us=jnp.asarray([[0.1]]))
    rp = RestorationProblem(problem, ref, params, Options())
    rparams = RestorationParams(params, ref)

    def evaluate(primal, params):
      e = rp.evaluate(primal, params)
      system, rhs = rp.linearize(
          primal, jnp.zeros_like(e.eq), jnp.zeros_like(e.eqT), params
      )
      return e.eq, rp.valid(primal, params), system.QT, rhs.gTx

    constraints, valid, _, _ = jax.jit(evaluate)(
        rp.seed(ref, params, 0.1), rparams
    )
    self.assertTrue(bool(valid))
    np.testing.assert_allclose(constraints, 0.0, atol=1e-14)

  def test_hard_restoration_improves_feasible_problem_and_rejects_infeasible(
      self,
  ):
    problem = Problem(
        horizon=1,
        num_states=1,
        num_controls=1,
        initial_state=lambda p: jnp.zeros(1),
        dynamics=lambda x, u, p: x + u,
        stage_cost=lambda x, u, p: 0.5 * jnp.dot(u, u),
        terminal_cost=lambda x, p: 0.5 * jnp.dot(x, x),
        stage_equality=lambda x, u, p: jnp.atleast_1d(p[0] * u[0] - p[1]),
    )
    opts = Options(
        max_iter=10,
        max_resto_iter=20,
        initialize_dual=False,
        max_soc=0,
        max_soft_resto_iters=0,
        tol=1e-6,
        constr_viol_tol=1e-6,
    )

    def restore(params):
      q = problem.initialize(params, us=jnp.asarray([[0.1]]))
      it, _, _, _, _ = initialize(problem, params, q, opts)
      restored, okay, iterations, status = _hard_restore(
          problem,
          it,
          params,
          jnp.asarray(0.1),
          empty_filter(11, jnp.float64),
          opts,
      )
      return (
          okay,
          iterations,
          status,
          problem.evaluate(restored.primal, params).eq,
      )

    compiled = jax.jit(restore)
    okay, iterations, status, constraints = compiled(jnp.asarray([1.0, 0.3]))
    self.assertTrue(bool(okay), (iterations, status, constraints))
    self.assertGreater(int(iterations), 0)
    self.assertLess(float(jnp.sum(jnp.abs(constraints))), 0.18)
    okay, iterations, status, constraints = compiled(jnp.asarray([0.0, 1.0]))
    self.assertFalse(bool(okay))
    self.assertGreater(float(jnp.max(jnp.abs(constraints))), 0.9)


if __name__ == '__main__':
  unittest.main()
