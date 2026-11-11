from absl.testing import absltest
from absl.testing import parameterized
from unittest import mock
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel


def _dynamics(x, u, k, p):
  return jp.array([x[0] + u[0] ** 2 + k * p['a'], x[1] + jp.sin(u[0])])


def _cost(x, u, k, p):
  return jp.sum(x**2) + jp.sum(u**2) + x[0] * u[0] + k * p['a']


def _terminal(x, p):
  return jp.sum(x**4)


def _stage_eq(x, u, k, p):
  return jp.array([x[0] * u[0]])


def _stage_ineq(x, u, k, p):
  return jp.array([x[1] ** 2 + u[0] ** 2])


def _terminal_eq(x, p):
  return jp.array([x[0] ** 2, x[1] ** 2])


def _terminal_ineq(x, p):
  return jp.array([x[0] ** 3, x[1] ** 3])


def _model(**kwargs):
  options = dict(
      N=2,
      nx=2,
      nu=1,
      initial_state=lambda p: p['x0'],
      dynamics=_dynamics,
      stage_cost=_cost,
      terminal_cost=_terminal,
      stage_equality_constraint=_stage_eq,
      stage_inequality_constraint=_stage_ineq,
      terminal_equality_constraint=_terminal_eq,
      terminal_inequality_constraint=_terminal_ineq,
      x_bounds=(np.array([-1.0, -np.inf]), np.array([3.0, np.inf])),
      u_bounds=(np.array([-np.inf]), np.array([2.0])),
      stage_inequality_bounds=(np.array([-2.0]), np.array([4.0])),
      terminal_inequality_bounds=(np.array([-3.0, -4.0]), np.array([5.0, 6.0])),
  )
  options.update(kwargs)
  return StagewiseOcpModel(**options)


class StagewiseOcpModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(jax.enable_x64())
    self.params = {'x0': jp.array([0.2, 0.3]), 'a': 0.4}
    self.xs = jp.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    self.us = jp.array([[0.1], [0.2]])

  def test_evaluators_are_composable_without_aot_compilation(self):
    with mock.patch.object(
        jax.stages.Lowered,
        'compile',
        side_effect=AssertionError('Model AOT compilation'),
    ):
      m = _model(params_trace=self.params)
    self.assertFalse(hasattr(m, '_params'))
    composed = jax.jit(lambda x, u, p: jp.sum(m.objective(x, u, p)))
    executable = composed.lower(self.xs, self.us, self.params).compile()
    changed = self.params | {'a': 0.9}
    np.testing.assert_allclose(
        executable(self.xs, self.us, changed),
        jp.sum(m.objective(self.xs, self.us, changed)),
    )
    self.assertAlmostEqual(
        float(
            executable(self.xs, self.us, changed)
            - executable(self.xs, self.us, self.params)
        ),
        0.5,
    )

  @parameterized.parameters(True, False)
  def test_values_and_derivative_blocks(self, gauss_newton):
    m = _model(gauss_newton=gauss_newton, params_trace=self.params)
    xs, us, p = self.xs, self.us, self.params
    ks = jp.arange(2)
    costs = jp.concatenate([
        jax.vmap(_cost, in_axes=(0, 0, 0, None))(xs[:-1], us, ks, p),
        _terminal(xs[-1], p)[None],
    ])
    np.testing.assert_allclose(m.objective(xs, us, params=p), costs)
    self.assertIsInstance(m.objective(xs, us, params=p), jax.Array)
    gx, gu = m.objective_gradient(xs, us, params=p)
    self.assertEqual(gx.shape, (3, 2))
    self.assertEqual(gu.shape, (2, 1))
    np.testing.assert_allclose(
        gx[:-1], 2 * xs[:-1] + jp.concatenate([us, jp.zeros_like(us)], axis=1)
    )
    np.testing.assert_allclose(gx[-1], 4 * xs[-1] ** 3)
    np.testing.assert_allclose(gu, 2 * us + xs[:-1, :1])

    A, B = m.dynamics_jacobian(xs[:-1], us, xs[1:], params=p)
    np.testing.assert_allclose(A, np.broadcast_to(np.eye(2), (2, 2, 2)))
    np.testing.assert_allclose(
        B, jp.stack([2 * us[:, 0], jp.cos(us[:, 0])], axis=1)[:, :, None]
    )
    with self.assertRaisesRegex(ValueError, 'expected shape'):
      m.dynamics_jacobian(xs[:-1], us, xs, params=p)
    np.testing.assert_allclose(
        m.dynamics_residual(xs[:-1], us, xs[1:], params=p),
        jax.vmap(_dynamics, in_axes=(0, 0, 0, None))(xs[:-1], us, ks, p)
        - xs[1:],
    )

    def stage_ineq(x, u, k, p):
      return jp.concatenate([x[:1], u, _stage_ineq(x, u, k, p)])

    def terminal_ineq(x, p):
      return jp.concatenate([x[:1], _terminal_ineq(x, p)])

    for name, fn in (
        ('stage_equality', _stage_eq),
        ('stage_inequality', stage_ineq),
    ):
      np.testing.assert_allclose(
          getattr(m, name + '_residual')(xs[:-1], us, params=p),
          jax.vmap(fn, in_axes=(0, 0, 0, None))(xs[:-1], us, ks, p),
      )
      actual = getattr(m, name + '_jacobian')(xs[:-1], us, params=p)
      expected = jax.vmap(
          jax.jacfwd(fn, argnums=(0, 1)), in_axes=(0, 0, 0, None)
      )(xs[:-1], us, ks, p)
      for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b)
    for name, fn in (
        ('terminal_equality', _terminal_eq),
        ('terminal_inequality', terminal_ineq),
    ):
      np.testing.assert_allclose(
          getattr(m, name + '_residual')(xs[-1], params=p), fn(xs[-1], p)
      )
      np.testing.assert_allclose(
          getattr(m, name + '_jacobian')(xs[-1], params=p),
          jax.jacfwd(fn)(xs[-1], p),
      )
    np.testing.assert_allclose(
        m.stage_inequality_bounds[0], [-1.0, -np.inf, -2.0]
    )
    np.testing.assert_allclose(m.stage_inequality_bounds[1], [3.0, 2.0, 4.0])
    np.testing.assert_allclose(
        m.terminal_inequality_bounds[0], [-1.0, -3.0, -4.0]
    )
    np.testing.assert_allclose(m.terminal_inequality_bounds[1], [3.0, 5.0, 6.0])
    np.testing.assert_allclose(m.initial_state(params=p), p['x0'])
    np.testing.assert_allclose(m.initial_residual(xs[0], params=p), 0.0)
    np.testing.assert_allclose(m.initial_jacobian(xs[0], params=p), np.eye(2))

    multipliers = (
        jp.array([2.0, 3.0]),
        jp.array([[0.2, 0.3], [0.4, 0.5]]),
        jp.array([[0.6], [0.7]]),
        jp.full((2, 3), 0.8),
        jp.array([0.9, 1.0]),
        jp.full(3, 1.1),
    )
    scale = 1.3

    def reference(z):
      x, u = z[:6].reshape(3, 2), z[6:].reshape(2, 1)
      result = scale * (
          sum(_cost(x[k], u[k], k, p) for k in range(2)) + _terminal(x[-1], p)
      )
      if not gauss_newton:
        init, ld, le, li, lte, lti = multipliers
        result += init @ (x[0] - p['x0'])
        for k in range(2):
          result += ld[k] @ (_dynamics(x[k], u[k], k, p) - x[k + 1])
          result += le[k] @ _stage_eq(x[k], u[k], k, p)
          result += li[k] @ stage_ineq(x[k], u[k], k, p)
        result += lte @ _terminal_eq(x[-1], p) + lti @ terminal_ineq(x[-1], p)
      return result

    dense = jax.hessian(reference)(jp.concatenate([xs.ravel(), us.ravel()]))
    hxx, hxu, huu = m.lagrangian_hessian(
        xs, us, *multipliers, objective_factor=scale, params=p
    )
    self.assertEqual(hxx.shape, (3, 2, 2))
    self.assertEqual(hxu.shape, (2, 2, 1))
    self.assertEqual(huu.shape, (2, 1, 1))
    for k in range(3):
      np.testing.assert_allclose(
          hxx[k], dense[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
      )
      if k < 2:
        np.testing.assert_allclose(
            hxu[k], dense[2 * k : 2 * k + 2, 6 + k : 7 + k]
        )
        np.testing.assert_allclose(huu[k], dense[6 + k : 7 + k, 6 + k : 7 + k])
    changed = m.lagrangian_hessian(
        xs,
        us,
        multipliers[0] * 100,
        *multipliers[1:],
        objective_factor=scale,
        params=p
    )
    for actual, expected in zip(changed, (hxx, hxu, huu)):
      np.testing.assert_array_equal(actual, expected)
    with self.assertRaisesRegex(ValueError, 'expected shape'):
      m.lagrangian_hessian(xs, us, jp.zeros(3), *multipliers[1:], 1.0, params=p)

  @parameterized.parameters(0, 2)
  def test_empty_constraints_and_controls(self, N):
    m = StagewiseOcpModel(
        N,
        2,
        0,
        initial_state=lambda p: jp.ones(2),
        dynamics=lambda x, u, k, p: x,
        stage_cost=lambda x, u, k, p: jp.sum(x * x),
        terminal_cost=lambda x, p: jp.sum(x * x),
    )
    m.initialize(None)
    xs, us = jp.ones((N + 1, 2)), jp.zeros((N, 0))
    self.assertEqual(m.objective(xs, us, params=None).shape, (N + 1,))
    self.assertEqual(m.objective_gradient(xs, us, params=None)[1].shape, (N, 0))
    self.assertEqual(
        m.stage_equality_jacobian(xs[:-1], us, params=None)[1].shape, (N, 0, 0)
    )
    self.assertEqual(
        m.dynamics_jacobian(xs[:-1], us, xs[1:], params=None)[1].shape,
        (N, 2, 0),
    )
    self.assertEqual(
        m.terminal_inequality_residual(xs[-1], params=None).shape, (0,)
    )
    blocks = m.lagrangian_hessian(
        xs,
        us,
        jp.zeros(2),
        jp.zeros((N, 2)),
        jp.zeros((N, 0)),
        jp.zeros((N, 0)),
        jp.zeros(0),
        jp.zeros(0),
        1.0,
        params=None,
    )
    self.assertEqual(blocks[0].shape, (N + 1, 2, 2))
    self.assertEqual(blocks[2].shape, (N, 0, 0))

  @parameterized.parameters(
      dict(N=-1),
      dict(nx=0),
      dict(nu=-1),
      dict(N=1.5),
      dict(x_bounds=(np.zeros(2), np.zeros(2))),
      dict(u_bounds=(np.zeros(2), np.ones(2))),
      dict(x_bounds=(np.array([np.nan, 0.0]), np.ones(2))),
      dict(stage_inequality_bounds=(np.zeros(2), np.ones(2))),
      dict(terminal_inequality_bounds=(np.zeros(2), np.zeros(2))),
      dict(stage_cost=lambda x, u, k, p: x),
      dict(dynamics=lambda x, u, k, p: u),
      dict(initial_state=lambda p: jp.zeros(1)),
      dict(stage_equality_constraint=lambda x, u, k, p: jp.zeros((1, 1))),
  )
  def test_invalid_configuration(self, **kwargs):
    with self.assertRaises(ValueError):
      _model(params_trace=self.params, **kwargs)


if __name__ == '__main__':
  absltest.main()
