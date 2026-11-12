"""Public OCP behavior, with analytical and independent rotation references."""

from dataclasses import dataclass
from functools import partial

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jp
import numpy as np
from scipy.spatial.transform import Rotation

from mujodico.trajopt._src import manifold
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel


_STATE_WEIGHTS = np.array([0.8, 0.5, 1.1, 0.7, 0.4, 0.9])
_CONTROL_WEIGHTS = np.array([1.2, 0.8])
_TERMINAL_WEIGHTS = np.array([1.2, 0.4, 0.9, 0.6, 0.7, 1.1])
_COORDINATES = (('euclidean', False), ('quaternion', True))


def _rotation(q):
  return Rotation.from_quat(np.roll(q, -1))


def _quat(v):
  return np.roll(Rotation.from_rotvec(v).as_quat(), 1)


def _retract(x, dx, positions):
  """NumPy/SciPy reference; each quaternion consumes three tangent entries."""
  result, start = [], 0
  for count, i in enumerate(positions):
    result.extend(x[start:i] + dx[start - count : i - count])
    rotation = _rotation(x[i : i + 4]) * Rotation.from_rotvec(
        dx[i - count : i - count + 3]
    )
    result.extend(np.roll(rotation.as_quat(), 1))
    start = i + 4
  result.extend(x[start:] + dx[start - len(positions) :])
  return np.asarray(result)


def _difference(value, reference, positions):
  result, start = [], 0
  for i in positions:
    result.extend(value[start:i] - reference[start:i])
    result.extend(
        (
            _rotation(reference[i : i + 4]).inv() * _rotation(value[i : i + 4])
        ).as_rotvec()
    )
    start = i + 4
  result.extend(value[start:] - reference[start:])
  return np.asarray(result)


def _stage_cost(x, u, k, p, xp=jp):
  return (
      p['gain'] * xp.sum(xp.asarray(_STATE_WEIGHTS) * x * x)
      + xp.sum(xp.asarray(_CONTROL_WEIGHTS) * u * u)
      + x[0] * u[0]
      + 0.3 * x[2] * u[1]
      + (k + 1) * p['offset']
  )


def _terminal_cost(x, p, xp=jp):
  return (
      xp.sum(xp.asarray(_TERMINAL_WEIGHTS) * x * x)
      + 0.2 * x[0] ** 4
      + p['gain'] * x[5]
  )


def _dynamics(x, u, k, p, *, quaternion, xp=jp):
  if quaternion:
    increment = xp.array([u[0], x[0] * u[1], p['drift'] * x[5]])
    if xp is np:
      q = np.roll(
          (_rotation(x[1:5]) * Rotation.from_rotvec(increment)).as_quat(), 1
      )
    else:
      q = manifold.quat_multiply(x[1:5], manifold.quat_exp(increment))
    return xp.concatenate([
        (x[0] + u[0] ** 2 + (k + 1) * p['drift'])[None],
        q,
        (x[5] + xp.sin(u[1]))[None],
    ])
  return xp.array([
      x[0] + u[0] ** 2 + (k + 1) * p['drift'],
      x[1] + xp.sin(u[1]),
      x[2] + x[0] * u[1],
      x[3] + 0.2 * u[0],
      x[4] + 0.1 * x[5] ** 2,
      x[5] + u[0] * u[1],
  ])


def _stage_equality(x, u, k, p, xp=jp):
  return xp.array([x[0] * u[0] + x[5] ** 2, x[2] * u[1] + k * p['drift']])


def _stage_inequality(x, u, k, p, xp=jp):
  return xp.array([x[1] ** 2 + u[0] ** 2, x[4] * u[1] - x[5]])


def _terminal_equality(x, p, xp=jp):
  return xp.array([x[0] ** 2, x[2] * x[5], x[4] - p['drift']])


def _terminal_inequality(x, p, xp=jp):
  return xp.array([x[3] ** 2 - x[5]])


def _constraint_values(kind, x, u, k, p):
  if kind == 'stage_equality':
    return _stage_equality(x, u, k, p, np)
  if kind == 'stage_inequality':
    return np.r_[x[[0, 3]], u, _stage_inequality(x, u, k, p, np)]
  if kind == 'terminal_equality':
    return _terminal_equality(x, p, np)
  return np.r_[x[[0, 3]], _terminal_inequality(x, p, np)]


def _constraint_jacobian(kind, x, u):
  """Explicit derivatives of the Euclidean fixture, including bound rows."""
  if kind == 'stage_equality':
    jx, ju = np.zeros((2, 6)), np.zeros((2, 2))
    jx[0, [0, 5]], jx[1, 2] = [u[0], 2 * x[5]], u[1]
    ju[0, 0], ju[1, 1] = x[0], x[2]
    return jx, ju
  if kind == 'stage_inequality':
    jx, ju = np.zeros((6, 6)), np.zeros((6, 2))
    jx[0, 0] = jx[1, 3] = 1
    ju[2:4] = np.eye(2)
    jx[4, 1], ju[4, 0] = 2 * x[1], 2 * u[0]
    jx[5, [4, 5]], ju[5, 1] = [u[1], -1], x[4]
    return jx, ju
  jx = np.zeros((3, 6))
  if kind == 'terminal_equality':
    jx[0, 0], jx[1, [2, 5]], jx[2, 4] = 2 * x[0], [x[5], x[2]], 1
  else:
    jx[0, 0] = jx[1, 3] = 1
    jx[2, [3, 5]] = [2 * x[3], -1]
  return jx


def _finite_jacobian(fn, x):
  step = 1e-6
  return np.stack(
      [
          (np.asarray(fn(x + d)) - np.asarray(fn(x - d))) / (2 * step)
          for d in step * np.eye(x.size)
      ],
      axis=-1,
  )


def _finite_hessian(fn, size):
  step = 3e-4
  zero, directions = np.zeros(size), step * np.eye(size)
  result, center = np.zeros((size, size)), fn(zero)
  for i, a in enumerate(directions):
    result[i, i] = (fn(a) - 2 * center + fn(-a)) / step**2
    for j, b in enumerate(directions[:i]):
      result[i, j] = result[j, i] = (
          fn(a + b) - fn(a - b) - fn(-a + b) + fn(-a - b)
      ) / (4 * step**2)
  return result


def _dense_hessian(hxx, hxu, huu):
  K, tx, _ = hxx.shape
  N, _, nu = hxu.shape
  result = np.zeros((K * tx + N * nu, K * tx + N * nu))
  for k in range(K):
    state = slice(k * tx, (k + 1) * tx)
    result[state, state] = hxx[k]
    if k < N:
      control = slice(K * tx + k * nu, K * tx + (k + 1) * nu)
      result[state, control], result[control, state] = hxu[k], hxu[k].T
      result[control, control] = huu[k]
  return result


@dataclass
class _Case:
  model: StagewiseOcpModel
  xs: np.ndarray
  us: np.ndarray
  params: dict
  positions: tuple[int, ...]

  def multipliers(self):
    tx = self.model.nx_tangent
    return (
        np.linspace(0.1, 0.6, tx),
        np.linspace(0.2, 0.7, 2 * tx).reshape(2, tx),
        np.array([[0.3, -0.2], [0.4, 0.1]]),
        np.full((2, 6), 0.2),
        np.array([0.2, -0.3, 0.1]),
        np.array([0.1, 0.2, 0.3]),
    )

  def lagrangian(self, perturbation, multipliers, scale, gauss_newton):
    """Independent scalar Lagrangian in frozen next-state charts."""
    tx = self.model.nx_tangent
    dxs, dus = perturbation[: 3 * tx].reshape(3, tx), perturbation[
        3 * tx :
    ].reshape(2, 2)
    xs = np.stack(
        [_retract(x, dx, self.positions) for x, dx in zip(self.xs, dxs)]
    )
    us = self.us + dus
    value = scale * (
        sum(_stage_cost(xs[k], us[k], k, self.params, np) for k in range(2))
        + _terminal_cost(xs[-1], self.params, np)
    )
    if not gauss_newton:
      initial, dynamics, eq, ineq, terminal_eq, terminal_ineq = multipliers
      # The prescribed initial state is the prediction; xs[0] is the anchor.
      value += initial @ _difference(self.params['x0'], xs[0], self.positions)
      for k in range(2):
        prediction = _dynamics(
            xs[k], us[k], k, self.params, quaternion=bool(self.positions), xp=np
        )
        value += dynamics[k] @ (
            _difference(prediction, self.xs[k + 1], self.positions) - dxs[k + 1]
        )
        value += eq[k] @ _constraint_values(
            'stage_equality', xs[k], us[k], k, self.params
        )
        value += ineq[k] @ _constraint_values(
            'stage_inequality', xs[k], us[k], k, self.params
        )
      value += terminal_eq @ _constraint_values(
          'terminal_equality', xs[-1], None, None, self.params
      )
      value += terminal_ineq @ _constraint_values(
          'terminal_inequality', xs[-1], None, None, self.params
      )
    return value


def _case(quaternion=False, **overrides):
  xs = np.array([
      [0.2, -0.1, 0.3, 0.4, -0.2, 0.1],
      [-0.3, 0.2, 0.1, -0.2, 0.3, 0.4],
      [0.4, 0.1, -0.2, 0.3, 0.2, -0.1],
  ])
  params = dict(
      x0=np.array([-0.1, 0.2, -0.3, 0.1, 0.4, -0.2]),
      gain=0.7,
      drift=0.15,
      offset=0.2,
  )
  positions = (1,) if quaternion else ()
  if quaternion:
    xs[:, 1:5] = [
        _quat(v) for v in ([0.2, -0.1, 0.3], [-0.3, 0.2, 0.1], [0.1, 0.3, -0.2])
    ]
    params['x0'][1:5] = _quat([-0.2, 0.3, -0.1])
  options = dict(
      N=2,
      nx=6,
      nu=2,
      initial_state=lambda p: p['x0'],
      dynamics=partial(_dynamics, quaternion=quaternion),
      stage_cost=_stage_cost,
      terminal_cost=_terminal_cost,
      stage_equality_constraint=_stage_equality,
      stage_inequality_constraint=_stage_inequality,
      terminal_equality_constraint=_terminal_equality,
      terminal_inequality_constraint=_terminal_inequality,
      x_bounds=(
          np.array([-2, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
          np.array([np.inf, np.inf, np.inf, 1.5, np.inf, np.inf]),
      ),
      u_bounds=(np.array([-0.6, -np.inf]), np.array([0.7, 1.2])),
      stage_inequality_bounds=(np.array([-0.4, -np.inf]), np.array([1.8, 3.0])),
      terminal_inequality_bounds=(np.array([-1.0]), np.array([2.0])),
      quaternion_positions=positions,
  )
  options.update(overrides)
  return _Case(
      StagewiseOcpModel(**options),
      xs,
      np.array([[0.1, -0.2], [-0.3, 0.2]]),
      params,
      positions,
  )


def _analytical_hessian(case, multipliers, scale, gauss_newton):
  hxx = np.tile(
      np.diag(2 * scale * case.params['gain'] * _STATE_WEIGHTS), (3, 1, 1)
  )
  hxx[-1] = np.diag(2 * scale * _TERMINAL_WEIGHTS)
  hxx[-1, 0, 0] += scale * 2.4 * case.xs[-1, 0] ** 2
  hxu, huu = np.zeros((2, 6, 2)), np.tile(
      np.diag(2 * scale * _CONTROL_WEIGHTS), (2, 1, 1)
  )
  hxu[:, 0, 0], hxu[:, 2, 1] = scale, 0.3 * scale
  if not gauss_newton:
    _, ld, le, li, lte, lti = multipliers
    for k in range(2):
      hxx[k, 5, 5] += 0.2 * ld[k, 4] + 2 * le[k, 0]
      hxx[k, 1, 1] += 2 * li[k, 4]
      hxu[k, 0, 0] += le[k, 0]
      hxu[k, 0, 1] += ld[k, 2]
      hxu[k, 2, 1] += le[k, 1]
      hxu[k, 4, 1] += li[k, 5]
      huu[k, 0, 0] += 2 * ld[k, 0] + 2 * li[k, 4]
      huu[k, 1, 1] -= ld[k, 1] * np.sin(case.us[k, 1])
      huu[k, 0, 1] += ld[k, 5]
      huu[k, 1, 0] += ld[k, 5]
    hxx[-1, 0, 0] += 2 * lte[0]
    hxx[-1, 2, 5] += lte[1]
    hxx[-1, 5, 2] += lte[1]
    hxx[-1, 3, 3] += 2 * lti[-1]
  return _dense_hessian(hxx, hxu, huu)


class StagewiseOcpModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(jax.enable_x64())

  @parameterized.parameters('constructor', 'initialize', 'evaluation')
  def test_initialization(self, how):
    reference = _case()
    case = (
        _case(params_trace=reference.params)
        if how == 'constructor'
        else reference
    )
    if how != 'constructor':
      with self.assertRaises(RuntimeError):
        _ = case.model.constraint_sizes
      if how == 'initialize':
        case.model.initialize(case.params)
    costs = case.model.objective(case.xs, case.us, case.params)
    self.assertEqual(costs.shape, (3,))
    self.assertEqual(case.model.constraint_sizes, (2, 6, 3, 3))
    self.assertEqual(case.model.evaluation_dtype, np.dtype('float64'))

  @parameterized.named_parameters(*_COORDINATES)
  def test_objective_values(self, quaternion):
    c = _case(quaternion)
    actual = c.model.objective(c.xs, c.us, c.params)
    expected = [
        _stage_cost(c.xs[k], c.us[k], k, c.params, np) for k in range(2)
    ]
    expected.append(_terminal_cost(c.xs[-1], c.params, np))
    self.assertIsInstance(actual, jax.Array)
    np.testing.assert_allclose(actual, expected, atol=1e-14)

  @parameterized.named_parameters(*_COORDINATES)
  def test_objective_gradients(self, quaternion):
    c = _case(quaternion)
    gx, gu = jax.jit(c.model.objective_gradient)(c.xs, c.us, c.params)
    self.assertEqual(gx.shape, (3, c.model.nx_tangent))
    self.assertEqual(gu.shape, (2, 2))
    for array in (gx, gu):
      self.assertIsInstance(array, jax.Array)
    if not quaternion:
      expected_x = 2 * c.params['gain'] * _STATE_WEIGHTS * c.xs
      expected_x[:-1, 0] += c.us[:, 0]
      expected_x[:-1, 2] += 0.3 * c.us[:, 1]
      expected_x[-1] = 2 * _TERMINAL_WEIGHTS * c.xs[-1]
      expected_x[-1, 0] += 0.8 * c.xs[-1, 0] ** 3
      expected_x[-1, 5] += c.params['gain']
      expected_u = 2 * _CONTROL_WEIGHTS * c.us + c.xs[:-1, [0, 2]] * np.array(
          [1.0, 0.3]
      )
      np.testing.assert_allclose(gx, expected_x, atol=1e-14)
      np.testing.assert_allclose(gu, expected_u, atol=1e-14)
    else:
      for k in range(3):
        if k < 2:
          expected = _finite_jacobian(
              lambda d: _stage_cost(
                  _retract(c.xs[k], d[:5], c.positions),
                  c.us[k] + d[5:],
                  k,
                  c.params,
                  np,
              ),
              np.zeros(7),
          )
          np.testing.assert_allclose(np.r_[gx[k], gu[k]], expected, atol=1e-9)
        else:
          expected = _finite_jacobian(
              lambda d: _terminal_cost(
                  _retract(c.xs[-1], d, c.positions), c.params, np
              ),
              np.zeros(5),
          )
          np.testing.assert_allclose(gx[-1], expected, atol=1e-9)

  @parameterized.product(
      quaternion=(False, True), full_dynamics_jacobian=(False, True)
  )
  def test_dynamics_residuals_and_jacobians(
      self, quaternion, full_dynamics_jacobian
  ):
    c = _case(quaternion, full_dynamics_jacobian=full_dynamics_jacobian)
    tx = c.model.nx_tangent
    m = c.model
    residuals = m.dynamics_residual(c.xs[:-1], c.us, c.xs[1:], c.params)
    A, B = jax.jit(m.dynamics_jacobian)(c.xs[:-1], c.us, c.xs[1:], c.params)
    self.assertEqual(residuals.shape, (2, tx))
    self.assertEqual(A.shape, (2, tx, tx))
    self.assertEqual(B.shape, (2, tx, 2))
    for k, (x, u) in enumerate(zip(c.xs[:-1], c.us)):
      prediction = _dynamics(x, u, k, c.params, quaternion=quaternion, xp=np)
      np.testing.assert_allclose(
          residuals[k],
          _difference(prediction, c.xs[k + 1], c.positions),
          atol=1e-14,
      )
      if quaternion:
        expected = _finite_jacobian(
            lambda d: _difference(
                _dynamics(
                    _retract(x, d[:tx], c.positions),
                    u + d[tx : tx + 2],
                    k,
                    c.params,
                    quaternion=True,
                    xp=np,
                ),
                _retract(c.xs[k + 1], d[tx + 2 :], c.positions),
                c.positions,
            ),
            np.zeros(2 * tx + 2),
        )
        jx_u, jnext = expected[:, : tx + 2], expected[:, tx + 2 :]
        expected = (
            -np.linalg.solve(jnext, jx_u) if full_dynamics_jacobian else jx_u
        )
        np.testing.assert_allclose(
            np.concatenate([A[k], B[k]], axis=1), expected, atol=1e-9
        )
        if full_dynamics_jacobian:
          np.testing.assert_allclose(
              jx_u + jnext @ np.concatenate([A[k], B[k]], axis=1),
              0.0,
              atol=1e-9,
          )
        self.assertGreater(np.linalg.norm(jnext + np.eye(tx)), 0.1)
        # For quaternion-log defects, normalizing the constant term preserves
        # the residual: -Jnext**-1 r = r.
        np.testing.assert_allclose(
            -np.linalg.solve(jnext, residuals[k]), residuals[k], atol=1e-9
        )
      else:
        expected_a, expected_b = np.eye(6), np.zeros((6, 2))
        expected_a[2, 0], expected_a[4, 5] = u[1], 0.2 * x[5]
        expected_b[0, 0], expected_b[1, 1], expected_b[2, 1] = (
            2 * u[0],
            np.cos(u[1]),
            x[0],
        )
        expected_b[3, 0], expected_b[5] = 0.2, [u[1], u[0]]
        np.testing.assert_allclose(A[k], expected_a, atol=1e-14)
        np.testing.assert_allclose(B[k], expected_b, atol=1e-14)
    self.assertGreater(np.linalg.norm(residuals), 0.1)
    changed = c.params | {'drift': 0.4}
    self.assertFalse(
        np.allclose(
            residuals, m.dynamics_residual(c.xs[:-1], c.us, c.xs[1:], changed)
        )
    )
    if quaternion:
      opposite = c.xs.copy()
      opposite[:, 1:5] *= -1
      np.testing.assert_allclose(
          m.dynamics_residual(opposite[:-1], c.us, opposite[1:], c.params),
          residuals,
          atol=1e-14,
      )

  @parameterized.named_parameters(*_COORDINATES)
  def test_initial_condition(self, quaternion):
    c = _case(quaternion)
    m, x, tx = c.model, c.xs[0], c.model.nx_tangent
    np.testing.assert_array_equal(m.initial_state(c.params), c.params['x0'])
    np.testing.assert_allclose(
        m.initial_residual(x, c.params),
        _difference(c.params['x0'], x, c.positions),
        atol=1e-14,
    )
    expected = (
        _finite_jacobian(
            lambda d: _difference(
                c.params['x0'], _retract(x, d, c.positions), c.positions
            ),
            np.zeros(tx),
        )
        if quaternion
        else -np.eye(tx)
    )
    np.testing.assert_allclose(
        jax.jit(m.initial_jacobian)(x, c.params),
        expected,
        atol=1e-9 if quaternion else 1e-14,
    )
    np.testing.assert_allclose(
        m.initial_residual(c.params['x0'], c.params), 0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        m.initial_jacobian(c.params['x0'], c.params), -np.eye(tx), atol=1e-14
    )

  @parameterized.product(
      quaternion=(False, True),
      kind=(
          'stage_equality',
          'stage_inequality',
          'terminal_equality',
          'terminal_inequality',
      ),
  )
  def test_constraint_values_and_jacobians(self, quaternion, kind):
    c = _case(quaternion)
    m, tx = c.model, c.model.nx_tangent
    residual = getattr(m, kind + '_residual')
    jacobian = jax.jit(getattr(m, kind + '_jacobian'))
    if kind.startswith('stage'):
      values, (jx, ju) = residual(c.xs[:-1], c.us, c.params), jacobian(
          c.xs[:-1], c.us, c.params
      )
      for k, (x, u) in enumerate(zip(c.xs[:-1], c.us)):
        np.testing.assert_allclose(
            values[k], _constraint_values(kind, x, u, k, c.params), atol=1e-14
        )
        expected = (
            _finite_jacobian(
                lambda d: _constraint_values(
                    kind,
                    _retract(x, d[:tx], c.positions),
                    u + d[tx:],
                    k,
                    c.params,
                ),
                np.zeros(tx + 2),
            )
            if quaternion
            else np.concatenate(_constraint_jacobian(kind, x, u), axis=1)
        )
        np.testing.assert_allclose(
            np.concatenate([jx[k], ju[k]], axis=1),
            expected,
            atol=1e-9 if quaternion else 1e-14,
        )
    else:
      x = c.xs[-1]
      values, jx = residual(x, c.params), jacobian(x, c.params)
      np.testing.assert_allclose(
          values, _constraint_values(kind, x, None, None, c.params), atol=1e-14
      )
      expected = (
          _finite_jacobian(
              lambda d: _constraint_values(
                  kind, _retract(x, d, c.positions), None, None, c.params
              ),
              np.zeros(tx),
          )
          if quaternion
          else _constraint_jacobian(kind, x, None)
      )
      np.testing.assert_allclose(
          jx, expected, atol=1e-9 if quaternion else 1e-14
      )
    self.assertIsInstance(values, jax.Array)
    self.assertIsInstance(jx, jax.Array)

  def test_effective_bounds_and_independent_constraint_counts(self):
    c = _case()
    c.model.initialize(c.params)
    self.assertEqual(c.model.constraint_sizes, (2, 6, 3, 3))
    for actual, expected in zip(
        c.model.stage_inequality_bounds,
        (
            [-2, -np.inf, -0.6, -np.inf, -0.4, -np.inf],
            [np.inf, 1.5, 0.7, 1.2, 1.8, 3],
        ),
    ):
      np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(
        c.model.terminal_inequality_bounds,
        ([-2, -np.inf, -1], [np.inf, 1.5, 2]),
    ):
      np.testing.assert_array_equal(actual, expected)
    default = _case(
        stage_inequality_bounds=None, terminal_inequality_bounds=None
    )
    default.model.initialize(default.params)
    np.testing.assert_array_equal(
        default.model.stage_inequality_bounds[0][-2:], 0.0
    )
    np.testing.assert_array_equal(
        default.model.stage_inequality_bounds[1][-2:], np.inf
    )
    np.testing.assert_array_equal(
        default.model.terminal_inequality_bounds[0][-1:], 0.0
    )
    np.testing.assert_array_equal(
        default.model.terminal_inequality_bounds[1][-1:], np.inf
    )

  @parameterized.product(quaternion=(False, True), gauss_newton=(False, True))
  def test_lagrangian_hessian(self, quaternion, gauss_newton):
    c = _case(quaternion, gauss_newton=gauss_newton)
    multipliers, scale = c.multipliers(), 0.9
    evaluate = jax.jit(c.model.lagrangian_hessian)
    blocks = evaluate(c.xs, c.us, *multipliers, scale, c.params)
    tx = c.model.nx_tangent
    self.assertEqual(
        tuple(block.shape for block in blocks),
        ((3, tx, tx), (2, tx, 2), (2, 2, 2)),
    )
    self.assertTrue(all(isinstance(block, jax.Array) for block in blocks))
    actual = _dense_hessian(*blocks)
    if quaternion:
      expected = _finite_hessian(
          lambda z: c.lagrangian(z, multipliers, scale, gauss_newton),
          3 * tx + 4,
      )
      np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=3e-6)
    else:
      np.testing.assert_allclose(
          actual,
          _analytical_hessian(c, multipliers, scale, gauss_newton),
          atol=1e-14,
      )
    zero_multipliers = tuple(np.zeros_like(v) for v in multipliers)
    objective = _dense_hessian(
        *evaluate(c.xs, c.us, *zero_multipliers, 1.0, c.params)
    )
    constraints = _dense_hessian(
        *evaluate(c.xs, c.us, *multipliers, 0.0, c.params)
    )
    np.testing.assert_allclose(
        actual, scale * objective + constraints, atol=1e-13
    )
    if gauss_newton:
      np.testing.assert_array_equal(constraints, 0.0)

  @parameterized.named_parameters(*_COORDINATES)
  def test_initial_multiplier_curvature(self, quaternion):
    c = _case(quaternion, gauss_newton=False)
    multipliers = list(c.multipliers())
    multipliers[1:] = [np.zeros_like(v) for v in multipliers[1:]]
    hxx, hxu, huu = jax.jit(c.model.lagrangian_hessian)(
        c.xs, c.us, *multipliers, 0.0, c.params
    )
    tx = c.model.nx_tangent
    expected = (
        _finite_hessian(
            lambda d: multipliers[0]
            @ _difference(
                c.params['x0'], _retract(c.xs[0], d, c.positions), c.positions
            ),
            tx,
        )
        if quaternion
        else np.zeros((tx, tx))
    )
    np.testing.assert_allclose(
        hxx[0], expected, atol=1e-8 if quaternion else 1e-14
    )
    np.testing.assert_array_equal(hxx[1:], 0.0)
    np.testing.assert_array_equal(hxu, 0.0)
    np.testing.assert_array_equal(huu, 0.0)
    if quaternion:
      self.assertGreater(np.linalg.norm(hxx[0]), 1e-3)

  @parameterized.named_parameters(*_COORDINATES)
  def test_public_evaluators_compose_with_dynamic_parameters(self, quaternion):
    c = _case(quaternion)

    def composed(xs, us, p):
      return (
          jp.sum(c.model.objective(xs, us, p))
          + jp.sum(c.model.dynamics_residual(xs[:-1], us, xs[1:], p))
          + jp.sum(c.model.initial_residual(xs[0], p))
      )

    executable = jax.jit(composed).lower(c.xs, c.us, c.params).compile()
    changed = c.params | {
        'gain': 1.3,
        'drift': 0.4,
        'offset': -0.1,
        'x0': c.xs[-1],
    }
    for params in (c.params, changed):
      expected = sum(
          _stage_cost(c.xs[k], c.us[k], k, params, np) for k in range(2)
      ) + _terminal_cost(c.xs[-1], params, np)
      expected += sum(
          np.sum(
              _difference(
                  _dynamics(
                      c.xs[k], c.us[k], k, params, quaternion=quaternion, xp=np
                  ),
                  c.xs[k + 1],
                  c.positions,
              )
          )
          for k in range(2)
      )
      expected += np.sum(_difference(params['x0'], c.xs[0], c.positions))
      np.testing.assert_allclose(
          executable(c.xs, c.us, params), expected, atol=1e-13
      )
    self.assertNotAlmostEqual(
        float(executable(c.xs, c.us, c.params)),
        float(executable(c.xs, c.us, changed)),
    )

  @parameterized.named_parameters(
      ('euclidean', ()), ('one_quaternion', (1,)), ('two_quaternions', (6, 1))
  )
  def test_single_and_batched_retractions(self, positions):
    nx, nu = (11, 0) if len(positions) == 2 else (6, 2)
    x = np.linspace(-0.3, 0.5, nx)
    for i in positions:
      x[i : i + 4] = _quat([0.1, -0.2, 0.3])
    m = StagewiseOcpModel(
        0,
        nx,
        nu,
        initial_state=lambda p: p,
        dynamics=lambda x, u, k, p: x,
        stage_cost=lambda x, u, k, p: jp.sum(x * x),
        terminal_cost=lambda x, p: jp.sum(x * x),
        quaternion_positions=positions,
    )
    positions = tuple(sorted(positions))
    self.assertEqual(m.quaternion_positions, positions)
    self.assertEqual(m.nx_tangent, nx - len(positions))
    dx, u, du = (
        np.linspace(-0.2, 0.3, m.nx_tangent),
        np.arange(nu) * 0.1,
        np.full(nu, -0.2),
    )
    state, control = jax.jit(m.retract_state), jax.jit(m.retract_control)
    for alpha in (0.0, 0.35, 1.0):
      updated = state(x, dx, alpha, x)
      np.testing.assert_allclose(
          updated, _retract(x, alpha * dx, positions), atol=1e-14
      )
      np.testing.assert_allclose(
          m.initial_residual(updated, x), -alpha * dx, atol=1e-14
      )
      np.testing.assert_allclose(
          control(u, du, alpha, x), u + alpha * du, atol=1e-14
      )
      for count in (0, 1, 3):
        xs, dxs = np.tile(x, (count, 1)), np.tile(dx, (count, 1))
        actual = state(xs, dxs, alpha, x)
        self.assertEqual(actual.shape, (count, nx))
        np.testing.assert_allclose(
            actual,
            np.tile(_retract(x, alpha * dx, positions), (count, 1)),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            control(np.tile(u, (count, 1)), np.tile(du, (count, 1)), alpha, x),
            np.tile(u + alpha * du, (count, 1)),
        )
        for i in positions:
          np.testing.assert_allclose(
              np.linalg.norm(actual[:, i : i + 4], axis=1), 1.0, atol=1e-14
          )

  @parameterized.product(quaternion=(False, True), N=(0, 2))
  def test_zero_controls_and_empty_constraints(self, quaternion, N):
    x = _quat([0.1, -0.2, 0.3]) if quaternion else np.array([0.2, -0.4])
    positions = (0,) if quaternion else ()
    m = StagewiseOcpModel(
        N,
        x.size,
        0,
        initial_state=lambda p: x,
        dynamics=lambda x, u, k, p: x,
        stage_cost=lambda x, u, k, p: jp.dot(x, x),
        terminal_cost=lambda x, p: jp.dot(x, x),
        quaternion_positions=positions,
        gauss_newton=False,
    )
    xs, us, tx = np.tile(x, (N + 1, 1)), np.zeros((N, 0)), m.nx_tangent
    self.assertEqual(m.objective(xs, us, None).shape, (N + 1,))
    gx, gu = m.objective_gradient(xs, us, None)
    np.testing.assert_allclose(
        gx, np.zeros((N + 1, tx)) if quaternion else 2 * xs, atol=1e-14
    )
    self.assertEqual(gu.shape, (N, 0))
    np.testing.assert_allclose(
        m.dynamics_residual(xs[:-1], us, xs[1:], None),
        np.zeros((N, tx)),
        atol=1e-14,
    )
    A, B = m.dynamics_jacobian(xs[:-1], us, xs[1:], None)
    np.testing.assert_allclose(A, np.tile(np.eye(tx), (N, 1, 1)), atol=1e-14)
    self.assertEqual(B.shape, (N, tx, 0))
    self.assertEqual(m.constraint_sizes, (0, 0, 0, 0))
    for kind in ('stage_equality', 'stage_inequality'):
      self.assertEqual(
          getattr(m, kind + '_residual')(xs[:-1], us, None).shape, (N, 0)
      )
      jx, ju = getattr(m, kind + '_jacobian')(xs[:-1], us, None)
      self.assertEqual((jx.shape, ju.shape), ((N, 0, tx), (N, 0, 0)))
    for kind in ('terminal_equality', 'terminal_inequality'):
      self.assertEqual(getattr(m, kind + '_residual')(x, None).shape, (0,))
      self.assertEqual(getattr(m, kind + '_jacobian')(x, None).shape, (0, tx))
    hxx, hxu, huu = jax.jit(m.lagrangian_hessian)(
        xs,
        us,
        np.zeros(tx),
        np.zeros((N, tx)),
        np.zeros((N, 0)),
        np.zeros((N, 0)),
        np.zeros(0),
        np.zeros(0),
        1.0,
        None,
    )
    np.testing.assert_allclose(
        hxx,
        np.zeros((N + 1, tx, tx))
        if quaternion
        else np.tile(2 * np.eye(tx), (N + 1, 1, 1)),
        atol=1e-14,
    )
    self.assertEqual((hxu.shape, huu.shape), ((N, tx, 0), (N, 0, 0)))

  @parameterized.named_parameters(*_COORDINATES)
  def test_invalid_evaluator_and_retraction_shapes(self, quaternion):
    c = _case(quaternion)
    m, tx = c.model, c.model.nx_tangent
    for call in (
        lambda: m.objective(c.xs[:-1], c.us, c.params),
        lambda: m.objective(c.xs, c.us[:1], c.params),
        lambda: m.dynamics_residual(c.xs[:-1], c.us, c.xs, c.params),
        lambda: m.initial_residual(np.zeros(5), c.params),
        lambda: m.lagrangian_hessian(
            c.xs, c.us, np.zeros(tx + 1), *c.multipliers()[1:], 1.0, c.params
        ),
    ):
      with self.assertRaisesRegex(ValueError, 'expected shape'):
        call()
    for x, dx in (
        (np.zeros(()), np.zeros(())),
        (np.zeros(5), np.zeros(tx)),
        (np.zeros((2, 6)), np.zeros((3, tx))),
        (np.zeros(6), np.zeros((1, tx))),
        (np.zeros((1, 1, 6)), np.zeros((1, 1, tx))),
    ):
      with self.assertRaises(AssertionError):
        m.retract_state(x, dx, 1.0, c.params)
    for u, du in (
        (np.zeros(()), np.zeros(())),
        (np.zeros(3), np.zeros(3)),
        (np.zeros((2, 2)), np.zeros((3, 2))),
        (np.zeros(2), np.zeros((1, 2))),
    ):
      with self.assertRaises(AssertionError):
        m.retract_control(u, du, 1.0, c.params)

  @parameterized.named_parameters(
      ('negative_horizon', dict(N=-1)),
      ('zero_states', dict(nx=0)),
      ('negative_controls', dict(nu=-1)),
      ('fractional_horizon', dict(N=1.5)),
      ('boolean_horizon', dict(N=True)),
      ('fixed_variables', dict(x_bounds=(np.zeros(6), np.zeros(6)))),
      ('reversed_bounds', dict(u_bounds=(np.ones(2), np.zeros(2)))),
      ('nan_bounds', dict(u_bounds=(np.array([np.nan, 0.0]), np.ones(2)))),
      ('bound_shape', dict(x_bounds=(np.zeros(5), np.ones(5)))),
      ('bound_pair', dict(u_bounds=(np.zeros(2),))),
      (
          'stage_bound_count',
          dict(stage_inequality_bounds=(np.zeros(1), np.ones(1))),
      ),
      (
          'terminal_fixed_inequality',
          dict(terminal_inequality_bounds=(np.zeros(1), np.zeros(1))),
      ),
  )
  def test_invalid_dimensions_and_bounds(self, changes):
    with self.assertRaises(ValueError):
      _case(params_trace=_case().params, **changes)

  @parameterized.named_parameters(
      ('initial_state', dict(initial_state=lambda p: jp.zeros(5))),
      ('dynamics', dict(dynamics=lambda x, u, k, p: jp.zeros(5))),
      ('stage_cost', dict(stage_cost=lambda x, u, k, p: x)),
      ('terminal_cost', dict(terminal_cost=lambda x, p: x)),
      (
          'stage_equality',
          dict(stage_equality_constraint=lambda x, u, k, p: jp.zeros((2, 1))),
      ),
      (
          'stage_inequality',
          dict(stage_inequality_constraint=lambda x, u, k, p: 0.0),
      ),
      (
          'terminal_equality',
          dict(terminal_equality_constraint=lambda x, p: jp.zeros((1, 1))),
      ),
      (
          'terminal_inequality',
          dict(terminal_inequality_constraint=lambda x, p: 0.0),
      ),
  )
  def test_invalid_callback_outputs(self, changes):
    with self.assertRaisesRegex(ValueError, 'returned shape'):
      _case(params_trace=_case().params, **changes)

  @parameterized.named_parameters(
      ('negative', (-1,)),
      ('past_end', (3,)),
      ('duplicate', (1, 1)),
      ('overlap', (0, 2)),
      ('fractional', (1.5,)),
      ('boolean', (True,)),
  )
  def test_invalid_quaternion_positions(self, positions):
    with self.assertRaisesRegex(ValueError, 'quaternion_positions'):
      _case(quaternion_positions=positions)


if __name__ == '__main__':
  absltest.main()
