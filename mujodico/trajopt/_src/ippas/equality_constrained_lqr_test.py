from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ipopt_trajopt import IpoptTrajOpt
from mujodico.trajopt._src.ippas import equality_constrained_lqr as lqr

_CASES = (
    ('with_stage_equalities', 2, 67, 5, 3, 2, 'psd'),
    ('no_stage_equalities', 3, 68, 5, 3, 0, 'psd'),
    ('singular_control_cost', 4, 69, 5, 3, 1, 'singular_r'),
    ('fully_constrained_controls', 5, 70, 5, 2, 2, 'singular_r'),
)


def _random_problem_data(seed, T, nx, nu, nc, curvature):
  rng = np.random.default_rng(seed)

  # Eu has orthonormal rows.
  basis, _ = np.linalg.qr(rng.normal(size=(T, nu, nu)))
  Eu = basis[:, :, :nc].swapaxes(-1, -2)

  if curvature == 'pd':
    factor = rng.normal(scale=0.2, size=(T, nx + nu, nx + nu))
    hessian = factor.swapaxes(-1, -2) @ factor + 0.5 * np.eye(nx + nu)
  else:
    # A joint Gram matrix guarantees PSD even with nonzero cross costs.
    # In the singular-R case, R vanishes exactly on the constrained controls
    # and is strictly positive on ker(Eu), so the saddle block is invertible.
    nfree = nu - nc if curvature == 'singular_r' else nu
    control_basis = basis[:, :, nu - nfree :]
    G = rng.uniform(0.7, 1.3, size=(T, nfree, 1)) * control_basis.swapaxes(
        -1, -2
    )
    F = rng.normal(scale=0.2, size=(T, nfree, nx))
    factor = np.concatenate((F, G), axis=-1)
    hessian = factor.swapaxes(-1, -2) @ factor

  terminal_rows = nx if curvature == 'pd' else nx - 1
  terminal_factor = rng.normal(scale=0.3, size=(terminal_rows, nx))
  terminal_Q = terminal_factor.T @ terminal_factor
  if curvature == 'pd':
    terminal_Q += 0.5 * np.eye(nx)

  data = dict(
      Q=np.concatenate((hessian[:, :nx, :nx], terminal_Q[None])),
      N=hessian[:, :nx, nx:],
      R=hessian[:, nx:, nx:],
      q=rng.normal(scale=0.3, size=(T + 1, nx)),
      r=rng.normal(scale=0.3, size=(T, nu)),
      A=0.65 * np.eye(nx) + rng.normal(scale=0.05, size=(T, nx, nx)),
      B=rng.normal(scale=0.2, size=(T, nx, nu)),
      c=rng.normal(scale=0.1, size=(T, nx)),
      Ex=rng.normal(scale=0.1, size=(T, nc, nx)),
      Eu=Eu,
      e=rng.normal(scale=0.1, size=(T, nc)),
      x0=rng.normal(size=nx),
  )
  return jax.tree.map(jp.asarray, data)


class EqualityConstrainedLQRTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())

  def _assert_kkt(self, solution, tol):
    p = jax.tree.map(np.asarray, solution)
    T, nx, nu = p.N.shape
    nc = p.e.shape[-1]

    for value, shape in (
        (p.x, (T + 1, nx)),
        (p.u, (T, nu)),
        (p.lmbda, (T + 1, nx)),
        (p.beta, (T, nc)),
    ):
      self.assertEqual(value.shape, shape)
      self.assertTrue(np.all(np.isfinite(value)))

    mv = lambda a, b: np.einsum('tij,tj->ti', a, b)
    mtv = lambda a, b: np.einsum('tji,tj->ti', a, b)
    residuals = {
        'initial': p.x[0] - p.x0,
        'dynamics': mv(p.A, p.x[:-1]) + mv(p.B, p.u) + p.c - p.x[1:],
        'equalities': (
            mv(p.Ex, p.x[:-1]) + mv(p.Eu, p.u) + p.e - p.delta_c * p.beta
        ),
        'state_stationarity': (
            mv(p.Q[:-1], p.x[:-1])
            + mv(p.N, p.u)
            + mtv(p.A, p.lmbda[1:])
            + mtv(p.Ex, p.beta)
            + p.q[:-1]
            - p.lmbda[:-1]
        ),
        'control_stationarity': (
            mtv(p.N, p.x[:-1])
            + mv(p.R, p.u)
            + mtv(p.B, p.lmbda[1:])
            + mtv(p.Eu, p.beta)
            + p.r
        ),
        'terminal': p.Q[-1] @ p.x[-1] + p.q[-1] - p.lmbda[-1],
    }
    for name, value in residuals.items():
      with self.subTest(condition=name):
        np.testing.assert_allclose(value, 0, rtol=tol, atol=tol)

  def _assert_against_ipopt(self, solution, tol):
    T, nx, nu = solution.N.shape
    nc = solution.e.shape[1]

    params = solution
    solver = IpoptTrajOpt(
        T,
        nx,
        nu,
        local_cost=lambda x, u, xnext, k, p: (
            0.5 * x @ p.Q[k] @ x
            + x @ p.N[k] @ u
            + 0.5 * u @ p.R[k] @ u
            + p.q[k] @ x
            + p.r[k] @ u
        ),
        terminal_cost=lambda x, p: (0.5 * x @ p.Q[-1] @ x + p.q[-1] @ x),
        initial_equality_constraint=lambda x, p: (p.x0 - x),
        local_equality_constraint=lambda x, u, xnext, k, p: jp.concatenate((
            p.A[k] @ x + p.B[k] @ u + p.c[k] - xnext,
            p.Ex[k] @ x + p.Eu[k] @ u + p.e[k],
        )),
        gauss_newton=False,
        params_trace=params,
    )

    x, u, info = solver.solve(
        params,
        xs_init=np.zeros((T + 1, nx)),
        us_init=np.zeros((T, nu)),
        print_level=0,
        sb='yes',
        nlp_scaling_method='none',
    )
    self.assertEqual(info['status'], 0, info['status_msg'])

    local_dual = info['mult_g'][nx:].reshape(T, nx + nc)
    lmbda = np.concatenate((info['mult_g'][:nx][None], local_dual[:, :nx]))
    beta = local_dual[:, nx:]

    for actual, expected in (
        (solution.x, x),
        (solution.u, u),
        (solution.lmbda, lmbda),
        (solution.beta, beta),
    ):
      np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

  @parameterized.named_parameters(*_CASES)
  def test_solve(self, seed, *args):
    tol = 1e-8

    data = _random_problem_data(seed, *args)
    prob = lqr.EqualityConstrainedLQR.create(**data)
    solution = jax.jit(lqr.solve)(prob)
    self._assert_kkt(solution, tol=tol)
    self._assert_against_ipopt(solution, tol=tol)

    data2 = _random_problem_data(seed + 1, *args)
    for name, value in data.items():
      if name not in ('q', 'r', 'c', 'e', 'x0'):
        data2[name] = value
    prob2 = lqr.EqualityConstrainedLQR.create(**data2)
    solution2 = jax.jit(lqr.solve)(prob2)

    prob3 = solution.replace_rhs(
        q=prob2.q, r=prob2.r, c=prob2.c, e=prob2.e, x0=prob2.x0
    )
    self.assertIs(prob3._cache, solution._cache)
    # Replaced residuals must reuse the existing matrix factorization.
    with mock.patch.object(
        lqr, '_build_matrix_cache', side_effect=AssertionError('Cache rebuilt')
    ):
      eager = lqr.solve(prob3)
      solution3 = jax.jit(lqr.solve)(prob3)
    self.assertIs(eager._cache, solution._cache)
    self._assert_kkt(eager, tol=tol)
    self._assert_kkt(solution3, tol=tol)
    self._assert_against_ipopt(solution3, tol=tol)
    for field in ('x', 'u', 'lmbda', 'beta'):
      np.testing.assert_allclose(
          getattr(solution2, field),
          getattr(eager, field),
          rtol=tol,
          atol=tol,
      )
      np.testing.assert_allclose(
          getattr(solution2, field),
          getattr(solution3, field),
          rtol=tol,
          atol=tol,
      )

  @parameterized.parameters(
      'full_rank',
      'state_only',
      'redundant',
      'near_dependent',
      'empty',
      'overdetermined',
  )
  def test_dual_regularization_against_dense_kkt(self, kind):
    T, nx, nu = 3, 2, 2
    nc = 0 if kind == 'empty' else 3 if kind == 'overdetermined' else 2
    data = _random_problem_data(42, T, nx, nu, min(nc, nu), 'pd')
    rng = np.random.default_rng(7)
    Eu = np.broadcast_to(np.eye(nu), (T, nu, nu)).copy()
    if kind == 'state_only':
      Eu[:] = 0
    if kind in ('redundant', 'near_dependent'):
      Eu[:, 1] = Eu[:, 0]
      if kind == 'near_dependent':
        Eu[:, 1, 1] = 1e-14
    if kind == 'overdetermined':
      Eu = np.concatenate((Eu, Eu[:, :1]), axis=1)
    data.update(
        Eu=jp.asarray(Eu[:, :nc]),
        Ex=jp.asarray(rng.normal(size=(T, nc, nx)) * 0.1),
        e=jp.asarray(rng.normal(size=(T, nc)) * 0.1),
        delta_c=jp.array(1e-3),
    )
    prob = lqr.EqualityConstrainedLQR.create(**data)
    solved = jax.jit(lqr.solve)(prob)
    # Assemble the entire saddle matrix independently of the parallel reduction.
    nv = (T + 1) * nx + T * nu
    nd = (T + 1) * nx + T * nc
    H, J = np.zeros((nv, nv)), np.zeros((nd, nv))
    g = np.concatenate((np.ravel(data['q']), np.ravel(data['r'])))
    b = np.concatenate(
        (-np.asarray(data['x0']), np.ravel(data['c']), np.ravel(data['e']))
    )
    x = lambda k: slice(k * nx, (k + 1) * nx)
    u = lambda k: slice((T + 1) * nx + k * nu, (T + 1) * nx + (k + 1) * nu)
    J[:nx, x(0)] = np.eye(nx)
    for k in range(T + 1):
      H[x(k), x(k)] = data['Q'][k]
    for k in range(T):
      H[u(k), u(k)] = data['R'][k]
      H[x(k), u(k)] = data['N'][k]
      H[u(k), x(k)] = data['N'][k].T
      row = slice((k + 1) * nx, (k + 2) * nx)
      J[row, x(k)], J[row, u(k)], J[row, x(k + 1)] = (
          data['A'][k],
          data['B'][k],
          -np.eye(nx),
      )
      row = slice((T + 1) * nx + k * nc, (T + 1) * nx + (k + 1) * nc)
      J[row, x(k)], J[row, u(k)] = data['Ex'][k], data['Eu'][k]
    D = np.diag(
        np.concatenate(
            (np.zeros((T + 1) * nx), np.full(T * nc, -float(prob.delta_c)))
        )
    )
    reference = np.linalg.solve(
        np.block([[H, J.T], [J, D]]), -np.concatenate((g, b))
    )
    np.testing.assert_allclose(
        np.concatenate((solved.x.ravel(), solved.u.ravel())),
        reference[:nv],
        atol=1e-9,
    )
    expected_dual = reference[nv:].copy()
    expected_dual[:nx] *= -1  # LQR uses lambda0*(x0-x).
    np.testing.assert_allclose(
        np.concatenate((solved.lmbda.ravel(), solved.beta.ravel())),
        expected_dual,
        atol=1e-9,
    )
    self._assert_kkt(solved, 1e-9)
    rhs = solved.replace_rhs(q=prob.q + 0.2, e=prob.e + 0.1)
    with mock.patch.object(
        lqr, '_build_matrix_cache', side_effect=AssertionError('Cache rebuilt')
    ):
      cached = jax.jit(lqr.solve)(rhs)
    self._assert_kkt(cached, 1e-9)


if __name__ == '__main__':
  absltest.main()
