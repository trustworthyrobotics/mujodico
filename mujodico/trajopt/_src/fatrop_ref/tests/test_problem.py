"""Independent derivative and manifold-lifting regression tests."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mujodico.trajopt._src.fatrop_ref.manifold import Euclidean, Quaternion, quat_exp, quat_mul
from mujodico.trajopt._src.fatrop_ref.problem import Primal, Problem

jax.config.update('jax_enable_x64', True)


def make_euclidean(**kwargs):
  args = dict(
      horizon=2,
      num_states=2,
      num_controls=1,
      initial_state=lambda p: p['x0'],
      stage_cost=lambda x, u, p: 0.5 * jnp.dot(x, x) + p['r'] * jnp.dot(u, u),
      terminal_cost=lambda x, p: 0.5
      * jnp.dot(x - p['target'], x - p['target']),
      dynamics=lambda x, u, p: jnp.asarray([x[0] + x[1], x[1] + u[0]]),
  )
  args.update(kwargs)
  return Problem(**args)


class ProblemTest(unittest.TestCase):

  def setUp(self):
    self.params = dict(
        x0=jnp.asarray([0.5, -0.1]),
        target=jnp.asarray([0.0, 0.0]),
        r=jnp.asarray(0.2),
    )

  def test_dynamic_parameters_and_dimensions(self):
    p = make_euclidean(
        stage_equality=lambda x, u, p: jnp.asarray([x[0] + u[0]]),
        stage_inequality=lambda x, u, p: jnp.asarray([u[0] ** 2]),
        stage_inequality_bounds=([-1.0], [2.0]),
        terminal_inequality=lambda x, p: jnp.asarray([x[0]]),
        terminal_inequality_bounds=([-3.0], [4.0]),
    )
    initial = p.initialize(self.params)
    ev = p.evaluate(initial, self.params)
    self.assertEqual(initial.s.shape, (2, 1))
    self.assertEqual(ev.eq.shape, (2, 4))
    self.assertEqual(ev.eqT.shape, (1,))
    sys, rhs = jax.jit(p.linearize)(
        initial, jnp.zeros_like(ev.eq), jnp.zeros_like(ev.eqT), self.params
    )
    self.assertEqual(sys.R.shape, (2, 4, 4))  # u, s, next-state tangent
    self.assertEqual(rhs.gTy.shape, (1,))
    np.testing.assert_allclose(rhs.c[:, :2], 0.0, atol=1e-14)
    mapped = jax.jit(
        lambda q, a, b, params: p.linearize(q, a, b, params, stagewise=True)
    )(initial, jnp.zeros_like(ev.eq), jnp.zeros_like(ev.eqT), self.params)
    for actual, expected in zip(
        jax.tree.leaves(mapped), jax.tree.leaves((sys, rhs))
    ):
      np.testing.assert_allclose(actual, expected, atol=1e-13)
    compile_count = []

    def f(params):
      compile_count.append(1)
      q = p.initialize(params)
      return q.xs, p.evaluate(q, params).objective

    compiled = jax.jit(f)
    a = compiled(self.params)
    b = compiled(
        dict(self.params, x0=jnp.asarray([0.7, 0.2]), r=jnp.asarray(0.3))
    )
    self.assertEqual(len(compile_count), 1)
    self.assertFalse(np.allclose(a[0], b[0]))
    self.assertNotEqual(float(a[1]), float(b[1]))

  def test_coincident_bounds_and_masks(self):
    p = make_euclidean(
        x_bounds=([-0.2, -np.inf], [0.8, np.inf]),
        u_bounds=([0.3], [0.3]),
        stage_inequality=lambda x, u, p: jnp.asarray([x[1], u[0]]),
        stage_inequality_bounds=([0.1, -np.inf], [0.1, np.inf]),
    )
    q = p.initialize(self.params)
    ev = p.evaluate(q, self.params)
    self.assertEqual(q.s.shape, (2, 0))
    self.assertEqual(ev.eq.shape[1], 4)  # dynamics + fixed g + fixed u
    lo, hi = p.bounds(q, self.params)
    self.assertTrue(np.all(np.isneginf(lo.xs[0])))
    self.assertTrue(np.all(np.isposinf(hi.xs[0])))
    self.assertTrue(np.all(np.isneginf(lo.us)))
    np.testing.assert_allclose(lo.xs[1:, 0], -0.2)

  def test_quaternion_storage_bounds_are_nonlinear(self):
    p = Problem(
        horizon=1,
        num_states=4,
        num_controls=0,
        state_manifold=Quaternion(),
        initial_state=lambda p: jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        stage_cost=lambda x, u, p: jnp.asarray(0.0, x.dtype),
        terminal_cost=lambda x, p: jnp.asarray(0.0, x.dtype),
        dynamics=lambda x, u, p: x,
        x_bounds=(
            [-0.9, -np.inf, -np.inf, -np.inf],
            [1.1, np.inf, np.inf, np.inf],
        ),
    )
    q = p.initialize(None)
    self.assertEqual(q.s.shape, (1, 1))
    self.assertEqual(q.st.shape, (1,))
    lo, hi = p.bounds(q, None)
    self.assertTrue(np.all(np.isneginf(lo.xs)))
    self.assertTrue(np.all(np.isposinf(hi.xs)))
    np.testing.assert_allclose(lo.s, -0.9)
    np.testing.assert_allclose(hi.st, 1.1)

  def test_terminal_auxiliary_controls(self):
    p = make_euclidean(
        terminal_control_dim=2,
        terminal_control_bounds=([0.0, 0.0], [np.inf, np.inf]),
        terminal_cost_with_control=lambda x, ut, params: jnp.dot(x, x)
        + jnp.sum(ut),
        terminal_equality_with_control=lambda x, ut, params: jnp.asarray(
            [x[0] + ut[0] - ut[1]]
        ),
    )
    q = p.initialize(self.params, ut=jnp.asarray([0.4, 0.3]))
    e = p.evaluate(q, self.params)
    s, r = p.linearize(
        q, jnp.zeros_like(e.eq), jnp.zeros_like(e.eqT), self.params
    )
    self.assertEqual(s.RT.shape, (2, 2))
    self.assertEqual(s.FT.shape, (1, 2))
    np.testing.assert_allclose(s.FT, [[1.0, -1.0]])
    np.testing.assert_allclose(r.gTy, [1.0, 1.0])

  def test_exact_manifold_lifting_matches_global_pullback(self):
    m = Quaternion()
    params = dict(x0=quat_exp(jnp.asarray([0.1, 0.2, -0.1])))
    p = Problem(
        horizon=2,
        num_states=4,
        num_controls=3,
        state_manifold=m,
        initial_state=lambda p: p['x0'],
        stage_cost=lambda q, u, p: 0.2 * jnp.dot(u, u)
        + 0.3 * jnp.dot(m.local(p['x0'], q), m.local(p['x0'], q)),
        terminal_cost=lambda q, p: 0.5
        * jnp.dot(m.local(p['x0'], q), m.local(p['x0'], q)),
        dynamics=lambda q, u, p: quat_mul(q, quat_exp(0.3 * u)),
    )
    xs = jnp.stack([
        params['x0'],
        quat_exp(jnp.asarray([0.5, -0.1, 0.2])),
        quat_exp(jnp.asarray([0.2, 0.4, 0.3])),
    ])
    us = jnp.asarray([[0.2, -0.1, 0.3], [-0.3, 0.2, 0.1]])
    q = p.initialize(params, xs, us)
    e = p.evaluate(q, params)
    mult = jnp.asarray([[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2]])
    system, rhs = p.linearize(q, mult, jnp.zeros_like(e.eqT), params)
    size = 3 * 3 + 2 * 3

    def lag(delta):
      x = jax.vmap(m.retract)(q.xs, delta[:9].reshape(3, 3))
      u = q.us + delta[9:].reshape(2, 3)
      value = p.evaluate(q._replace(xs=x, us=u), params)
      return value.objective + jnp.sum(value.eq * mult)

    dense_h = np.asarray(jax.hessian(lag)(jnp.zeros(size)))
    dense_g = np.asarray(jax.grad(lag)(jnp.zeros(size)))
    lifted_h = np.zeros_like(dense_h)
    lifted_g = np.zeros_like(dense_g)
    for k in range(2):
      ids = (
          list(range(3 * k, 3 * k + 3))
          + list(range(9 + 3 * k, 12 + 3 * k))
          + list(range(3 * k + 3, 3 * k + 6))
      )
      h = np.block([
          [np.asarray(system.Q[k]), np.asarray(system.S[k])],
          [np.asarray(system.S[k]).T, np.asarray(system.R[k])],
      ])
      lifted_h[np.ix_(ids, ids)] += h
      lifted_g[ids] += np.concatenate(
          [np.asarray(rhs.gx[k]), np.asarray(rhs.gy[k])]
      )
    lifted_h[6:9, 6:9] += system.QT
    lifted_g[6:9] += rhs.gTx
    np.testing.assert_allclose(lifted_h, dense_h, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(lifted_g, dense_g, atol=1e-12, rtol=1e-12)
    vector = jnp.arange(1, size + 1, dtype=jnp.float64)
    vector /= jnp.linalg.norm(vector)
    eps = 2e-4
    f0 = lag(jnp.zeros(size))
    fp, fm = lag(eps * vector), lag(-eps * vector)
    np.testing.assert_allclose(
        (fp - fm) / (2 * eps), dense_g @ vector, atol=1e-8, rtol=1e-7
    )
    np.testing.assert_allclose(
        (fp - 2 * f0 + fm) / (eps * eps),
        vector @ dense_h @ vector,
        atol=1e-7,
        rtol=1e-6,
    )
    # A nonzero defect must preserve physical cross-edge multiplier curvature.
    self.assertGreater(
        float(np.linalg.norm(np.asarray(system.S[0])[:, -3:])), 1e-4
    )


if __name__ == '__main__':
  unittest.main()
