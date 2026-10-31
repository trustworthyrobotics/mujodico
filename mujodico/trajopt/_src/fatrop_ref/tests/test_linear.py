"""Dense independent KKT oracles for both manuscript scan orders."""

import unittest
from unittest import mock

import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from mujodico.trajopt._src.fatrop_ref.linear_solver import (
    Direction,
    NONFINITE,
    RANK_DEFICIENT,
    RHS,
    SUCCESS,
    System,
    factorize,
    matvec,
    solve_rhs,
)
from mujodico.trajopt._src.fatrop_ref.linear_solver import Segment, compose


def random_problem(n=5, nx=3, ny=2, ne=1, nt=1, net=1, seed=9):
  rng = np.random.default_rng(seed)
  z = rng.normal(size=(n, nx + ny, nx + ny))
  h = np.swapaxes(z, -1, -2) @ z + np.eye(nx + ny)
  zt = rng.normal(size=(nx + nt, nx + nt))
  ht = zt.T @ zt + np.eye(nx + nt)
  system = System(
      h[:, :nx, :nx],
      h[:, :nx, nx:],
      h[:, nx:, nx:],
      rng.normal(size=(n, nx, nx)) * 0.3,
      rng.normal(size=(n, nx, ny)),
      rng.normal(size=(n, ne, nx)),
      rng.normal(size=(n, ne, ny)),
      np.full((n, ne), 0.15),
      ht[:nx, :nx],
      ht[:nx, nx:],
      ht[nx:, nx:],
      rng.normal(size=(net, nx)),
      rng.normal(size=(net, nt)),
      np.full(net, 0.12),
  )
  rhs = RHS(
      rng.normal(size=(n, nx)),
      rng.normal(size=(n, ny)),
      rng.normal(size=(n, ne)),
      rng.normal(size=(n, nx)),
      rng.normal(size=nx),
      rng.normal(size=nt),
      rng.normal(size=net),
      rng.normal(size=nx),
  )
  return jax.tree.map(jnp.asarray, system), jax.tree.map(jnp.asarray, rhs)


def dense_oracle(s, r):
  """Assemble primal Hessian and constraint Jacobian directly, no scan formulas."""
  s, r = jax.tree.map(np.asarray, (s, r))
  n, nx, ny, ne = s.Q.shape[0], s.Q.shape[1], s.R.shape[-1], s.E.shape[1]
  nt, net = s.RT.shape[-1], s.ET.shape[0]
  px, pu = (n + 1) * nx, n * ny
  npv = px + pu + nt
  nlambda, nbeta = (n + 1) * nx, n * ne
  nc = nlambda + nbeta + net
  h = np.zeros((npv, npv))
  jac = np.zeros((nc, npv))
  damp = np.zeros(nc)
  xsl = lambda k: slice(k * nx, (k + 1) * nx)
  ysl = lambda k: slice(px + k * ny, px + (k + 1) * ny)
  terminal_y = slice(px + pu, npv)
  jac[:nx, :nx] = -np.eye(nx)
  for k in range(n):
    x, y = xsl(k), ysl(k)
    h[x, x] += s.Q[k]
    h[x, y] += s.S[k]
    h[y, x] += s.S[k].T
    h[y, y] += s.R[k]
    dyn = slice((k + 1) * nx, (k + 2) * nx)
    jac[dyn, x], jac[dyn, y], jac[dyn, xsl(k + 1)] = s.A[k], s.B[k], -np.eye(nx)
    eq = slice(nlambda + k * ne, nlambda + (k + 1) * ne)
    jac[eq, x], jac[eq, y], damp[eq] = s.E[k], s.F[k], s.D[k]
  h[xsl(n), xsl(n)] += s.QT
  h[xsl(n), terminal_y] += s.ST
  h[terminal_y, xsl(n)] += s.ST.T
  h[terminal_y, terminal_y] += s.RT
  te = slice(nlambda + nbeta, nc)
  jac[te, xsl(n)], jac[te, terminal_y], damp[te] = s.ET, s.FT, s.DT
  full = np.block([[h, jac.T], [jac, -np.diag(damp)]])
  gradient = np.concatenate((r.gx.reshape(-1), r.gTx, r.gy.reshape(-1), r.gTy))
  constraint = np.concatenate((r.dx0, r.b.reshape(-1), r.c.reshape(-1), r.cT))
  # Initial equation is -dx0=-prescribed, like every other equality RHS.
  sol = np.linalg.solve(full, -np.concatenate((gradient, constraint)))
  primal, dual = sol[:npv], sol[npv:]
  return Direction(
      primal[:px].reshape(n + 1, nx),
      primal[px : px + pu].reshape(n, ny),
      dual[nlambda : nlambda + nbeta].reshape(n, ne),
      dual[nlambda + nbeta :],
      primal[px + pu :],
      dual[:nlambda].reshape(n + 1, nx),
  )


class LinearTest(unittest.TestCase):

  def assert_tree_close(self, a, b, atol=2e-10):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
      np.testing.assert_allclose(x, y, rtol=atol, atol=atol)

  def test_dense_oracle_and_both_orders(self):
    for n in (1, 3, 8):
      for ne, nt, net in ((0, 0, 0), (2, 1, 2), (1, 0, 1)):
        with self.subTest(n=n, ne=ne, nt=nt, net=net):
          s, r = random_problem(n=n, ne=ne, nt=nt, net=net)
          expected = dense_oracle(s, r)
          for backend in ('serial', 'parallel'):
            f = factorize(s, backend)
            self.assertEqual(int(f.status), SUCCESS)
            for rhs_backend in ('serial', 'parallel'):
              result = solve_rhs(f, r, rhs_backend)
              self.assert_tree_close(result, expected)
              residual = matvec(s, result)
              target = RHS(*[-a for a in r[:-1]], r.dx0)
              self.assert_tree_close(residual, target)

  def test_zero_horizon(self):
    s, r = random_problem(n=0, ne=0, nt=0, net=1)
    for backend in ('serial', 'parallel'):
      f = factorize(s, backend)
      self.assertEqual(int(f.status), SUCCESS)
      self.assert_tree_close(solve_rhs(f, r, backend), dense_oracle(s, r))

  def test_repeated_rhs_uses_existing_factors(self):
    s, r = random_problem()
    f = factorize(s)
    r2 = jax.tree.map(lambda a: a * 0.7 + 0.11, r)
    expected = dense_oracle(s, r2)
    with mock.patch(
        'jax.scipy.linalg.lu_factor', side_effect=AssertionError('refactorized')
    ):
      self.assert_tree_close(solve_rhs(f, r2), expected)

  def test_deficient_constraint_rows_with_positive_damping(self):
    s, r = random_problem(ne=3)
    s = s._replace(F=jnp.zeros_like(s.F))
    self.assertEqual(int(factorize(s).status), SUCCESS)
    self.assert_tree_close(solve_rhs(factorize(s), r), dense_oracle(s, r))
    s = s._replace(D=jnp.zeros_like(s.D))
    self.assertEqual(int(factorize(s).status), RANK_DEFICIENT)

  def test_nonfinite_distinct_status(self):
    s, _ = random_problem()
    s = s._replace(Q=s.Q.at[0, 0, 0].set(jnp.nan))
    self.assertEqual(int(factorize(s).status), NONFINITE)

  def test_jit(self):
    s, r = random_problem(n=3)
    f = jax.jit(factorize)(s)
    result = jax.jit(solve_rhs)(f, r)
    self.assert_tree_close(result, dense_oracle(s, r))

  def test_five_tuple_associativity(self):
    rng = np.random.default_rng(31)

    def segment():
      p, c = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
      return Segment(
          jnp.asarray(p.T @ p),
          jnp.asarray(rng.normal(size=3)),
          jnp.asarray(rng.normal(size=(3, 3))),
          jnp.asarray(rng.normal(size=3)),
          jnp.asarray(c.T @ c),
      )

    a, b, c = segment(), segment(), segment()
    self.assert_tree_close(compose(compose(a, b), c), compose(a, compose(b, c)))


if __name__ == '__main__':
  unittest.main()
