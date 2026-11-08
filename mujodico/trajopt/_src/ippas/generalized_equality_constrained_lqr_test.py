from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import generalized_equality_constrained_lqr as lqr
from mujodico.trajopt._src.ipopt_trajopt import IpoptTrajOpt


_CASES = (
    ('with_stage_equalities', 2,      67, 5, 3, 2, 'psd'),
    ('no_stage_equalities', 3,        68, 5, 3, 0, 'psd'),
    ('singular_control_cost', 4,      69, 5, 3, 1, 'singular_r'),
    ('fully_constrained_controls', 5, 70, 5, 2, 2, 'singular_r'),
)


def _random_problem_data(seed, T, nx, nu, nc, curvature):
  rng = np.random.default_rng(seed)
  d = 2 * nx + nu

  # Split control directions into constrained and free orthonormal bases.
  basis, _ = np.linalg.qr(rng.normal(size=(T, nu, nu)))
  control_curvature = np.broadcast_to(np.eye(nu), (T, nu, nu))
  if curvature == 'singular_r':
    free_basis = basis[:, :, nc:]
    control_curvature = free_basis @ free_basis.swapaxes(-1, -2)

  A = 0.65 * np.eye(nx) + rng.normal(scale=0.05, size=(T, nx, nx))
  B = rng.normal(scale=0.2, size=(T, nx, nu))
  if curvature == 'singular_r':
    # Keep zero-cost constrained controls out of the next-state cost too,
    # so the control Hessian remains singular after dynamics substitution.
    B = B @ control_curvature

  # Construct the Hessian directly on z = [x, u, xnext]. The rectangular
  # Gram factor gives a PSD cost with cross terms involving all three blocks.
  rows = d if curvature == 'pd' else nx
  factor = rng.normal(scale=0.2, size=(T, rows, d))
  if curvature == 'singular_r':
    factor[:, :, nx : nx + nu] = factor[:, :, nx : nx + nu] @ control_curvature
  H = factor.swapaxes(-1, -2) @ factor
  if curvature == 'pd':
    H += 0.5 * np.eye(d)
  else:
    # Positive curvature on free controls makes the local KKT block
    # nonsingular, while the full stage Hessian stays semidefinite.
    H[:, nx : nx + nu, nx : nx + nu] += 0.5 * control_curvature

  # Generate the full equality matrix directly, with independent control
  # rows and random dependence on both current and next states.
  E = rng.normal(scale=0.1, size=(T, nc, d))
  E[:, :, nx : nx + nu] = basis[:, :, :nc].swapaxes(-1, -2)

  terminal_rows = nx if curvature == 'pd' else nx - 1
  terminal_factor = rng.normal(scale=0.3, size=(terminal_rows, nx))
  terminal_Q = terminal_factor.T @ terminal_factor
  if curvature == 'pd':
    terminal_Q += 0.5 * np.eye(nx)

  data = dict(
      H=H,
      g=rng.normal(scale=0.3, size=(T, d)),
      A=A,
      B=B,
      c=rng.normal(scale=0.1, size=(T, nx)),
      E=E,
      e=rng.normal(scale=0.1, size=(T, nc)),
      x0=rng.normal(size=nx),
      Qf=terminal_Q,
      qf=rng.normal(scale=0.3, size=nx),
  )
  return jax.tree.map(jp.asarray, data)


class GeneralizedEqualityConstrainedLQRTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())

  def _assert_kkt(self, solution, tol):
    p = jax.tree.map(np.asarray, solution)
    T, nx, nu = p.B.shape
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
    z = np.concatenate((p.x[:-1], p.u, p.x[1:]), axis=-1)
    gradient = mv(p.H, z) + p.g + mtv(p.E, p.beta)
    next_gradient = gradient[:, nx + nu :]
    residuals = {
        'initial': p.x[0] - p.x0,
        'dynamics': mv(p.A, p.x[:-1]) + mv(p.B, p.u) + p.c - p.x[1:],
        'equalities': mv(p.E, z) + p.e,
        'state_stationarity': (
            gradient[:, :nx]
            + mtv(p.A, p.lmbda[1:])
            + np.concatenate((np.zeros((1, nx)), next_gradient[:-1]))
            - p.lmbda[:-1]
        ),
        'control_stationarity': (
            gradient[:, nx : nx + nu] + mtv(p.B, p.lmbda[1:])
        ),
        'terminal': p.Qf @ p.x[-1] + p.qf + next_gradient[-1] - p.lmbda[-1],
    }
    for name, value in residuals.items():
      with self.subTest(condition=name):
        np.testing.assert_allclose(value, 0, rtol=tol, atol=tol)

  def _assert_against_ipopt(self, solution, tol):
    T, nx, nu = solution.B.shape
    nc = solution.e.shape[1]

    params = solution
    solver = IpoptTrajOpt(
        T,
        nx,
        nu,
        local_cost=lambda x, u, xnext, k, p: (
            0.5
            * jp.concatenate((x, u, xnext))
            @ p.H[k]
            @ jp.concatenate((x, u, xnext))
            + p.g[k] @ jp.concatenate((x, u, xnext))
        ),
        terminal_cost=lambda x, p: (0.5 * x @ p.Qf @ x + p.qf @ x),
        initial_equality_constraint=lambda x, p: (p.x0 - x),
        local_equality_constraint=lambda x, u, xnext, k, p: jp.concatenate((
            p.A[k] @ x + p.B[k] @ u + p.c[k] - xnext,
            p.E[k] @ jp.concatenate((x, u, xnext)) + p.e[k],
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
    prob = lqr.GeneralizedEqualityConstrainedLQR.create(**data)
    solution = jax.jit(lqr.solve)(prob)
    self._assert_kkt(solution, tol=tol)
    self._assert_against_ipopt(solution, tol=tol)

    data2 = _random_problem_data(seed + 1, *args)
    for name, value in data.items():
      if name not in ('g', 'c', 'e', 'x0', 'qf'):
        data2[name] = value
    prob2 = lqr.GeneralizedEqualityConstrainedLQR.create(**data2)
    solution2 = jax.jit(lqr.solve)(prob2)

    prob3 = solution.replace_rhs(
        g=prob2.g, c=prob2.c, e=prob2.e, x0=prob2.x0, qf=prob2.qf
    )
    solution3 = jax.jit(lqr.solve)(prob3)
    self._assert_kkt(solution3, tol=tol)
    self._assert_against_ipopt(solution3, tol=tol)
    for field in ('x', 'u', 'lmbda', 'beta'):
      np.testing.assert_allclose(
          getattr(solution2, field),
          getattr(solution3, field),
          rtol=tol,
          atol=tol,
      )


if __name__ == '__main__':
  absltest.main()
