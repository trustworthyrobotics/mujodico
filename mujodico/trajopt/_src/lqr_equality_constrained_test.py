from dataclasses import replace
import importlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src import lqr_equality_constrained as lqr
from mujodico.trajopt._src.ipopt_trajopt import IPOptTrajOpt


_CASES = (
    ('with_stage_equalities', 2,      50, 5, 3, 2, 'psd'),
    ('no_stage_equalities', 3,        50, 5, 3, 0, 'psd'),
    ('singular_control_cost', 4,      50, 5, 3, 1, 'singular_r'),
    ('fully_constrained_controls', 5, 50, 5, 2, 2, 'singular_r'),
)


def _random_problem(seed, T, nx, nu, nc, curvature):
  rng = np.random.default_rng(seed)

  # Hu has orthonormal rows.
  basis, _ = np.linalg.qr(rng.normal(size=(T, nu, nu)))
  Hu = basis[:, :, :nc].swapaxes(-1, -2)

  if curvature == 'pd':
    factor = rng.normal(scale=0.2, size=(T, nx + nu, nx + nu))
    hessian = factor.swapaxes(-1, -2) @ factor + 0.5 * np.eye(nx + nu)
  else:
    # A joint Gram matrix guarantees PSD even with nonzero cross costs.
    # In the singular-R case, R vanishes exactly on the constrained controls
    # and is strictly positive on ker(Hu), so the saddle block is invertible.
    nfree = nu - nc if curvature == 'singular_r' else nu
    control_basis = basis[:, :, nu - nfree:]
    G = rng.uniform(0.7, 1.3, size=(T, nfree, 1)) * control_basis.swapaxes(-1, -2)
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
      Hx=rng.normal(scale=0.1, size=(T, nc, nx)),
      Hu=Hu,
      h=rng.normal(scale=0.1, size=(T, nc)),
      x0=rng.normal(size=nx),
  )
  return lqr.EqualityConstrainedLQR.create(**jax.tree.map(jp.asarray, data))


class EqualityConstrainedLQRTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())

  def _assert_kkt(self, solution, tol):
    p = jax.tree.map(np.asarray, solution)
    T, nx, nu = p.N.shape
    nc = p.h.shape[-1]
    for value, shape in (
        (p.x, (T + 1, nx)),
        (p.u, (T, nu)),
        (p.lmbda, (T + 1, nx)),
        (p.nu, (T, nc)),
    ):
      self.assertEqual(value.shape, shape)
      self.assertTrue(np.all(np.isfinite(value)))

    mv = lambda a, b: np.einsum('tij,tj->ti', a, b)
    mtv = lambda a, b: np.einsum('tji,tj->ti', a, b)
    residuals = {
        'initial': p.x[0] - p.x0,
        'dynamics': mv(p.A, p.x[:-1]) + mv(p.B, p.u) + p.c - p.x[1:],
        'equalities': mv(p.Hx, p.x[:-1]) + mv(p.Hu, p.u) + p.h,
        'state_stationarity': (
            mv(p.Q[:-1], p.x[:-1])
            + mv(p.N, p.u)
            + mtv(p.A, p.lmbda[1:])
            + mtv(p.Hx, p.nu)
            + p.q[:-1]
            - p.lmbda[:-1]
        ),
        'control_stationarity': (
            mtv(p.N, p.x[:-1])
            + mv(p.R, p.u)
            + mtv(p.B, p.lmbda[1:])
            + mtv(p.Hu, p.nu)
            + p.r
        ),
        'terminal': p.Q[-1] @ p.x[-1] + p.q[-1] - p.lmbda[-1],
    }
    for name, value in residuals.items():
      with self.subTest(condition=name):
        np.testing.assert_allclose(value, 0, rtol=tol, atol=tol)

  def _assert_against_ipopt(self, solution, tol):
    T, nx, nu = solution.N.shape
    nc = solution.h.shape[1]

    params = solution
    solver = IPOptTrajOpt(
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
        terminal_cost=lambda x, p: (
            0.5 * x @ p.Q[-1] @ x + p.q[-1] @ x
        ),
        initial_equality_constraint=lambda x, p: (
            p.x0 - x
        ),
        local_equality_constraint=lambda x, u, xnext, k, p: jp.concatenate((
            p.A[k] @ x + p.B[k] @ u + p.c[k] - xnext,
            p.Hx[k] @ x + p.Hu[k] @ u + p.h[k],
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
    nu = local_dual[:, nx:]

    for actual, expected in (
        (solution.x, x),
        (solution.u, u),
        (solution.lmbda, lmbda),
        (solution.nu, nu),
    ):
      np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

  @parameterized.named_parameters(*_CASES)
  def test_solve(self, *args):
    problem = _random_problem(*args)
    solution = jax.jit(lqr.solve)(problem)
    self._assert_kkt(solution, tol=1e-8)
    self._assert_against_ipopt(solution, tol=1e-8)


if __name__ == '__main__':
  absltest.main()
