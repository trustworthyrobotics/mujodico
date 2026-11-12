from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import equality_constrained_lqr as lqr
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd

_RHS_FIELDS = (
    'r_x',
    'r_u',
    'r_f',
    'r_e',
    'r_g',
    'r_c_l',
    'r_c_u',
    'r_gf',
    'r_cf_l',
    'r_cf_u',
    'r0',
)
_STEP_FIELDS = (
    'dx',
    'du',
    'dlmbda',
    'dbeta',
    'ds',
    'dsf',
    'dz_l',
    'dz_u',
    'dzf_l',
    'dzf_u',
)
_CASES = (
    ('two_sided', 2, 67, 5, 3, 2, 4, 2, 'two_sided', 'pd'),
    ('no_stage_equalities', 3, 68, 5, 3, 0, 2, 3, 'mixed', 'pd'),
    ('lower_only', 4, 5, 4, 3, 1, 3, 2, 'lower', 'pd'),
    ('upper_only', 5, 6, 4, 3, 1, 3, 2, 'upper', 'pd'),
    ('mixed_bounds', 6, 7, 4, 3, 1, 4, 5, 'mixed', 'pd'),
    ('unbounded', 7, 8, 4, 3, 1, 3, 2, 'unbounded', 'pd'),
    ('no_stage_inequalities', 8, 5, 4, 3, 1, 0, 2, 'mixed', 'pd'),
    ('no_terminal_inequalities', 9, 6, 4, 3, 1, 3, 0, 'mixed', 'pd'),
    ('single_stage_unconstrained', 10, 1, 4, 3, 0, 0, 0, 'mixed', 'pd'),
    ('singular_control_hessian', 11, 7, 4, 3, 1, 3, 2, 'mixed', 'singular_r'),
    ('fully_constrained_controls', 12, 5, 4, 2, 2, 3, 2, 'mixed', 'singular_r'),
)
_solve_jit = jax.jit(pd.solve)


def _random_problem_data(seed, T, nx, nu, ne, ng, ngf, bounds, curvature):
  rng = np.random.default_rng(seed)
  basis, _ = np.linalg.qr(rng.normal(size=(T, nu, nu)))
  Eu = basis[:, :, :ne].swapaxes(-1, -2)

  if curvature == 'singular_r':
    # Curvature vanishes on constrained controls and is positive on ker(Eu).
    free = basis[:, :, ne:]
    G = rng.uniform(0.7, 1.3, size=(T, nu - ne, 1)) * free.swapaxes(-1, -2)
    F = rng.normal(scale=0.2, size=(T, nu - ne, nx))
    factor = np.concatenate((F, G), axis=-1)
    hessian = factor.swapaxes(-1, -2) @ factor
  else:
    factor = rng.normal(scale=0.2, size=(T, nx + nu, nx + nu))
    hessian = factor.swapaxes(-1, -2) @ factor + 0.5 * np.eye(nx + nu)

  terminal_factor = rng.normal(scale=0.3, size=(nx - 1, nx))
  terminal_Hxx = terminal_factor.T @ terminal_factor
  Gu = rng.normal(scale=0.2, size=(T, ng, nu))
  if curvature == 'singular_r':
    # The bound contribution must also vanish on constrained controls.
    Gu = Gu @ (free @ free.swapaxes(-1, -2))

  def bound_data(shape):
    slack = rng.normal(scale=0.3, size=shape)
    lower = slack - rng.uniform(0.5, 1.5, size=shape)
    upper = slack + rng.uniform(0.5, 1.5, size=shape)
    if bounds in ('upper', 'unbounded'):
      lower[:] = -np.inf
    if bounds in ('lower', 'unbounded'):
      upper[:] = np.inf
    if bounds == 'mixed':
      # Include all four bound patterns, varying across stages and components.
      kind = np.arange(slack.size).reshape(shape) % 4
      lower[(kind == 2) | (kind == 3)] = -np.inf
      upper[(kind == 1) | (kind == 3)] = np.inf
    # Omitted-side multipliers intentionally remain nonzero: they are ignored.
    z_l = rng.uniform(0.2, 1.0, size=shape)
    z_u = rng.uniform(0.2, 1.0, size=shape)
    return lower, upper, slack, z_l, z_u

  g_l, g_u, s, z_l, z_u = bound_data((T, ng))
  gf_l, gf_u, sf, zf_l, zf_u = bound_data((ngf,))
  data = dict(
      Hxx=np.concatenate((hessian[:, :nx, :nx], terminal_Hxx[None])),
      Hxu=hessian[:, :nx, nx:],
      Huu=hessian[:, nx:, nx:],
      A=0.65 * np.eye(nx) + rng.normal(scale=0.05, size=(T, nx, nx)),
      B=rng.normal(scale=0.2, size=(T, nx, nu)),
      Ex=rng.normal(scale=0.1, size=(T, ne, nx)),
      Eu=Eu,
      Gx=rng.normal(scale=0.2, size=(T, ng, nx)),
      Gu=Gu,
      g_l=g_l,
      g_u=g_u,
      Gf=rng.normal(scale=0.2, size=(ngf, nx)),
      gf_l=gf_l,
      gf_u=gf_u,
      s=s,
      sf=sf,
      z_l=z_l,
      z_u=z_u,
      zf_l=zf_l,
      zf_u=zf_u,
  )
  for name, shape in (
      ('r_x', (T + 1, nx)),
      ('r_u', (T, nu)),
      ('r_f', (T, nx)),
      ('r_e', (T, ne)),
      ('r_g', (T, ng)),
      ('r_c_l', (T, ng)),
      ('r_c_u', (T, ng)),
      ('r_gf', (ngf,)),
      ('r_cf_l', (ngf,)),
      ('r_cf_u', (ngf,)),
      ('r0', (nx,)),
  ):
    data[name] = rng.normal(scale=0.3, size=shape)
  # A dense nonsymmetric initial Jacobian also exercises the transpose solve
  # used to recover the original initial-equality multiplier.
  data['E0'] = np.eye(nx) + rng.normal(scale=0.1, size=(nx, nx))
  return jax.tree.map(jp.asarray, data)


class IpPrimalDualSystemTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())

  def _assert_kkt(self, solution, tol=1e-8):
    p = jax.tree.map(np.asarray, solution)
    T, nx, nu = p.Hxu.shape
    ne = p.r_e.shape[1]
    ng = p.r_g.shape[1]
    ngf = p.r_gf.shape[0]
    shapes = (
        (T + 1, nx),
        (T, nu),
        (T + 1, nx),
        (T, ne),
        (T, ng),
        (ngf,),
        (T, ng),
        (T, ng),
        (ngf,),
        (ngf,),
    )
    for name, shape in zip(_STEP_FIELDS, shapes):
      with self.subTest(field=name):
        value = getattr(p, name)
        self.assertEqual(value.shape, shape)
        self.assertTrue(np.all(np.isfinite(value)))

    mv = lambda a, b: np.einsum('tij,tj->ti', a, b)
    mtv = lambda a, b: np.einsum('tji,tj->ti', a, b)
    dz = p.dz_l - p.dz_u
    state_dual = np.concatenate((
        (p.E0.T @ p.dlmbda[0])[None],
        -p.dlmbda[1:-1],
    ))
    residuals = {
        'initial': p.E0 @ p.dx[0] + p.r0,
        'dynamics': mv(p.A, p.dx[:-1]) + mv(p.B, p.du) - p.dx[1:] + p.r_f,
        'equalities': mv(p.Ex, p.dx[:-1]) + mv(p.Eu, p.du) + p.r_e,
        'state_stationarity': (
            mv(p.Hxx[:-1], p.dx[:-1])
            + mv(p.Hxu, p.du)
            + mtv(p.A, p.dlmbda[1:])
            + mtv(p.Ex, p.dbeta)
            - mtv(p.Gx, dz)
            + state_dual
            + p.r_x[:-1]
        ),
        'control_stationarity': (
            mtv(p.Hxu, p.dx[:-1])
            + mv(p.Huu, p.du)
            + mtv(p.B, p.dlmbda[1:])
            + mtv(p.Eu, p.dbeta)
            - mtv(p.Gu, dz)
            + p.r_u
        ),
        'terminal_stationarity': (
            p.Hxx[-1] @ p.dx[-1]
            - p.Gf.T @ (p.dzf_l - p.dzf_u)
            - p.dlmbda[-1]
            + p.r_x[-1]
        ),
        'stage_slack': mv(p.Gx, p.dx[:-1]) + mv(p.Gu, p.du) - p.ds + p.r_g,
        'terminal_slack': p.Gf @ p.dx[-1] - p.dsf + p.r_gf,
    }
    # Select finite-bound equations before arithmetic, avoiding infinity * 0.
    for prefix, lower, upper, slack, z_l, z_u, ds, dz_l, dz_u, r_l, r_u in (
        (
            'stage',
            p.g_l,
            p.g_u,
            p.s,
            p.z_l,
            p.z_u,
            p.ds,
            p.dz_l,
            p.dz_u,
            p.r_c_l,
            p.r_c_u,
        ),
        (
            'terminal',
            p.gf_l,
            p.gf_u,
            p.sf,
            p.zf_l,
            p.zf_u,
            p.dsf,
            p.dzf_l,
            p.dzf_u,
            p.r_cf_l,
            p.r_cf_u,
        ),
    ):
      finite_l, finite_u = np.isfinite(lower), np.isfinite(upper)
      residuals[prefix + '_lower_complementarity'] = (
          z_l[finite_l] * ds[finite_l]
          + (slack[finite_l] - lower[finite_l]) * dz_l[finite_l]
          + r_l[finite_l]
      )
      residuals[prefix + '_upper_complementarity'] = (
          -z_u[finite_u] * ds[finite_u]
          + (upper[finite_u] - slack[finite_u]) * dz_u[finite_u]
          + r_u[finite_u]
      )
      np.testing.assert_array_equal(dz_l[~finite_l], 0)
      np.testing.assert_array_equal(dz_u[~finite_u], 0)

    for name, value in residuals.items():
      with self.subTest(condition=name):
        np.testing.assert_allclose(value, 0, rtol=tol, atol=tol)

  @parameterized.named_parameters(*_CASES)
  def test_solve(self, seed, *args):
    prob = pd.IpPrimalDualSystem.create(**_random_problem_data(seed, *args))
    for field in _STEP_FIELDS:
      self.assertIsNone(getattr(prob, field))
    eager = pd.solve(prob)
    compiled = _solve_jit(prob)
    self._assert_kkt(eager)
    self._assert_kkt(compiled)
    for field in _STEP_FIELDS:
      np.testing.assert_allclose(
          getattr(eager, field),
          getattr(compiled, field),
          rtol=1e-8,
          atol=1e-8,
      )

  def test_replace_rhs_reuses_factors(self):
    args = (7, 4, 3, 1, 4, 5, 'mixed', 'pd')
    data = _random_problem_data(6, *args)
    solution = _solve_jit(pd.IpPrimalDualSystem.create(**data))
    data2 = _random_problem_data(16, *args)
    rhs = {name: data2[name] for name in _RHS_FIELDS}
    replaced = solution.replace_rhs(**rhs)
    self.assertIs(replaced._cache, solution._cache)
    for field in _STEP_FIELDS:
      self.assertIsNone(getattr(replaced, field))

    # Replaced residuals must reuse both the E0 and LQR factorizations.
    with (
        mock.patch.object(
            lqr, '_build_matrix_cache', side_effect=AssertionError
        ),
        mock.patch.object(
            jax.scipy.linalg, 'lu_factor', side_effect=AssertionError
        ),
    ):
      eager = pd.solve(replaced)
      compiled = _solve_jit(replaced)
    self.assertIs(eager._cache.reduced._cache, solution._cache.reduced._cache)
    self.assertIs(eager._cache.initial_lu, solution._cache.initial_lu)
    fresh = _solve_jit(pd.IpPrimalDualSystem.create(**(data | rhs)))
    for result in (eager, compiled, fresh):
      self._assert_kkt(result)
    for field in _STEP_FIELDS:
      for cached in (eager, compiled):
        np.testing.assert_allclose(
            getattr(cached, field),
            getattr(fresh, field),
            rtol=1e-8,
            atol=1e-8,
        )

  def test_omitted_sides_ignore_nonfinite_values(self):
    data = _random_problem_data(6, 7, 4, 3, 1, 4, 5, 'mixed', 'pd')
    reference = _solve_jit(pd.IpPrimalDualSystem.create(**data))
    for bound, multiplier, residual in (
        ('g_l', 'z_l', 'r_c_l'),
        ('g_u', 'z_u', 'r_c_u'),
        ('gf_l', 'zf_l', 'r_cf_l'),
        ('gf_u', 'zf_u', 'r_cf_u'),
    ):
      finite = jp.isfinite(data[bound])
      data[multiplier] = jp.where(finite, data[multiplier], jp.inf)
      data[residual] = jp.where(finite, data[residual], jp.nan)
    solution = _solve_jit(pd.IpPrimalDualSystem.create(**data))
    self._assert_kkt(solution)
    for field in _STEP_FIELDS:
      np.testing.assert_allclose(
          getattr(solution, field),
          getattr(reference, field),
          rtol=1e-8,
          atol=1e-8,
      )


if __name__ == '__main__':
  absltest.main()
