"""Independent full Newton equations for bounds and manifold edge curvature."""

import unittest
import json
import os
import subprocess

import jax

from mujodico.trajopt._src.fatrop_ref import iterate as bounds

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from mujodico.trajopt._src.fatrop_ref import search_direction
from mujodico.trajopt._src.fatrop_ref import search_direction as regularization
from mujodico.trajopt._src.fatrop_ref.linear_solver import SUCCESS, factorize, solve_rhs
from mujodico.trajopt._src.fatrop_ref.manifold import Quaternion, quat_exp, quat_mul
from mujodico.trajopt._src.fatrop_ref.options import Options
from mujodico.trajopt._src.fatrop_ref.problem import Primal, Problem


def tangent_functions(problem, primal, params):
  n, nx, nu = problem.N, problem.nxt, problem.nut
  shapes = (
      (n + 1, nx),
      (n, nu),
      primal.s.shape,
      primal.st.shape,
      primal.ut.shape,
  )
  sizes = [int(np.prod(shape)) for shape in shapes]

  def retract(z):
    values, start = [], 0
    for shape, size in zip(shapes, sizes):
      values.append(z[start : start + size].reshape(shape))
      start += size
    dx, du, ds, dst, dut = values
    xs = jax.vmap(lambda x, d: problem.state_manifold.retract(x, d, params))(
        primal.xs, dx
    )
    us = jax.vmap(lambda u, d: problem.control_manifold.retract(u, d, params))(
        primal.us, du
    )
    return Primal(xs, us, primal.s + ds, primal.st + dst, primal.ut + dut)

  def functions(z):
    value = problem.evaluate(retract(z), params)
    return value.objective, jnp.concatenate((value.eq.reshape(-1), value.eqT))

  return functions, jnp.zeros(sum(sizes)), sizes


def independent_step(
    problem,
    primal,
    eq,
    eqT,
    params,
    delta_w,
    delta_c,
    bd=None,
    zl=None,
    zu=None,
    mu=0.0,
):
  """Full tangent KKT, optionally including every finite bound-dual variable."""
  functions, zero, sizes = tangent_functions(problem, primal, params)
  multipliers = jnp.concatenate((eq.reshape(-1), eqT))

  def lag(z):
    objective, c = functions(z)
    return objective + jnp.dot(multipliers, c)

  h = np.asarray(jax.hessian(lag)(zero)) + delta_w * np.eye(zero.size)
  gradient = np.asarray(jax.grad(lag)(zero)).copy()
  j = np.asarray(jax.jacfwd(lambda z: functions(z)[1])(zero))
  c = np.asarray(functions(zero)[1])
  nx = problem.nxt
  initial = np.zeros((nx, zero.size))
  initial[:, :nx] = -np.eye(nx)
  j = np.concatenate((initial, j))
  c = np.concatenate((np.zeros(nx), c))
  de = np.concatenate((
      np.zeros(nx),
      np.tile(
          np.concatenate((np.zeros(nx), np.full(eq.shape[1] - nx, delta_c))),
          problem.N,
      ),
      np.full(eqT.size, delta_c),
  ))
  if bd is None:
    kkt = np.block([[h, j.T], [j, -np.diag(de)]])
    solution = np.linalg.solve(kkt, -np.concatenate((gradient, c)))
    return solution[: zero.size], solution[zero.size + nx :], h, None
  # Native-bound tests below use Euclidean coordinates: the direct Jacobians
  # of finite bounds are coordinate selectors, assembled independently.
  flatten = lambda tree: np.concatenate(
      [np.asarray(a).reshape(-1) for a in tree]
  )
  ml, mupper = flatten(bd.ml).astype(bool), flatten(bd.mu).astype(bool)
  dl, du, zlower, zupper = (
      flatten(bd.dl),
      flatten(bd.du),
      flatten(zl),
      flatten(zu),
  )
  il, iu = np.flatnonzero(ml), np.flatnonzero(mupper)
  vl, vu = np.eye(zero.size)[il], np.eye(zero.size)[iu]
  gradient += -zlower + zupper
  cl, cu = dl[il] * zlower[il] - mu, du[iu] * zupper[iu] - mu
  nl, nu, nc = il.size, iu.size, c.size
  kkt = np.block([
      [h, j.T, -vl.T, vu.T],
      [j, -np.diag(de), np.zeros((nc, nl)), np.zeros((nc, nu))],
      [
          zlower[il, None] * vl,
          np.zeros((nl, nc)),
          np.diag(dl[il]),
          np.zeros((nl, nu)),
      ],
      [
          -zupper[iu, None] * vu,
          np.zeros((nu, nc)),
          np.zeros((nu, nl)),
          np.diag(du[iu]),
      ],
  ])
  solution = np.linalg.solve(kkt, -np.concatenate((gradient, c, cl, cu)))
  return (
      solution[: zero.size],
      solution[zero.size + nx : zero.size + nc],
      h,
      (
          il,
          iu,
          solution[zero.size + nc : zero.size + nc + nl],
          solution[zero.size + nc + nl :],
      ),
  )


class NewtonTest(unittest.TestCase):

  def compare_step(
      self, problem, primal, params, bounded=False, cross_edge=False
  ):
    ev = problem.evaluate(primal, params)
    eq = jnp.linspace(-0.37, 0.49, ev.eq.size).reshape(ev.eq.shape)
    eqT = jnp.linspace(0.21, 0.43, ev.eqT.size)
    system, rhs = problem.linearize(primal, eq, eqT, params)
    bd = zl = zu = None
    if bounded:
      lower, upper = problem.bounds(primal, params)
      bd = bounds.data(primal, lower, upper)
      zl, zu = bounds.initialize_duals(bd)
      zl = jax.tree.map(lambda x: x * 0.7, zl)
      zu = jax.tree.map(lambda x: x * 0.9, zu)
      theta, force = bounds.condense(bd, zl, zu, 0.13)
      system, rhs = search_direction.add_bounds(
          problem, primal, system, rhs, theta, force
      )
    dw, dc = 3.0, 0.08
    system = regularization.shifted(
        system, dw, dc, problem.nxt, problem.nut, primal.s.shape[-1]
    )
    factors = factorize(system)
    self.assertEqual(int(factors.status), SUCCESS)
    result = solve_rhs(factors, rhs)
    primal_step = np.concatenate((
        np.asarray(result.dx).reshape(-1),
        np.asarray(result.dy[:, : problem.nut]).reshape(-1),
        np.asarray(
            result.dy[:, problem.nut : problem.nut + primal.s.shape[-1]]
        ).reshape(-1),
        np.asarray(result.dyT[problem.nt :]),
        np.asarray(result.dyT[: problem.nt]),
    ))
    dual_step = np.concatenate(
        (np.asarray(result.beta).reshape(-1), np.asarray(result.betaT))
    )
    expected, expected_dual, hessian, dual_bounds = independent_step(
        problem, primal, eq, eqT, params, dw, dc, bd, zl, zu, 0.13
    )
    np.testing.assert_allclose(primal_step, expected, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(dual_step, expected_dual, rtol=2e-9, atol=2e-9)
    if cross_edge:
      nx = problem.nxt
      self.assertGreater(
          np.linalg.norm(hessian[nx : 2 * nx, 2 * nx : 3 * nx]), 1e-4
      )
    if bounded:
      dq = search_direction.bound_step(problem, primal, result)
      dzl, dzu = bounds.dual_step(bd, zl, zu, dq, 0.13)
      flatten = lambda tree: np.concatenate(
          [np.asarray(a).reshape(-1) for a in tree]
      )
      il, iu, expected_l, expected_u = dual_bounds
      np.testing.assert_allclose(
          flatten(dzl)[il], expected_l, rtol=2e-9, atol=2e-9
      )
      np.testing.assert_allclose(
          flatten(dzu)[iu], expected_u, rtol=2e-9, atol=2e-9
      )

  def test_euclidean_full_complementarity_system(self):
    problem = Problem(
        horizon=2,
        num_states=1,
        num_controls=1,
        initial_state=lambda p: jnp.array([0.2]),
        dynamics=lambda x, u, p: x + 0.3 * u + 0.08 * x * x,
        stage_cost=lambda x, u, p: 0.5 * jnp.dot(x, x) + 0.7 * jnp.dot(u, u),
        terminal_cost=lambda x, p: 1.2 * jnp.sum((x - 0.7) ** 2),
        stage_inequality=lambda x, u, p: x * x + 0.2 * u,
        stage_inequality_bounds=(-1.0, 2.0),
        x_bounds=(-2.0, 2.0),
        u_bounds=(-0.7, 0.8),
        terminal_equality=lambda x, p: x - 0.4,
    )
    primal = problem.initialize(
        {}, xs=jnp.array([[0.2], [0.4], [0.1]]), us=jnp.array([[0.1], [-0.2]])
    )
    self.compare_step(problem, primal, {}, bounded=True)

  def test_quaternion_exact_cross_edge_system(self):
    manifold = Quaternion()
    goal = quat_exp(jnp.array([0.5, -0.3, 0.2]))
    problem = Problem(
        horizon=2,
        num_states=4,
        num_controls=3,
        state_manifold=manifold,
        initial_state=lambda p: quat_exp(jnp.array([0.1, 0.15, -0.2])),
        dynamics=lambda q, u, p: quat_mul(q, quat_exp(0.4 * u)),
        stage_cost=lambda q, u, p: 0.5 * jnp.dot(u, u)
        + 0.2 * jnp.sum(manifold.local(goal, q) ** 2),
        terminal_cost=lambda q, p: 0.7 * jnp.sum(manifold.local(goal, q) ** 2),
        terminal_equality=lambda q, p: manifold.local(goal, q)[:1],
    )
    xs = jax.vmap(quat_exp)(
        jnp.array([[0.1, 0.15, -0.2], [0.3, -0.2, 0.4], [-0.2, 0.4, 0.1]])
    )
    us = jnp.array([[0.2, -0.1, 0.3], [-0.15, 0.25, 0.1]])
    primal = problem.initialize({}, xs=xs, us=us)
    self.compare_step(problem, primal, {}, cross_edge=True)

  @unittest.skipUnless(
      os.environ.get('IPPAS_NATIVE_REFERENCE'),
      'set IPPAS_NATIVE_REFERENCE to the native reference executable',
  )
  def test_native_two_stage_quadratic_reference(self):
    output = subprocess.check_output(
        [os.environ['IPPAS_NATIVE_REFERENCE']], text=True
    )
    reference = json.loads(output.strip().splitlines()[-1])
    self.assertEqual(reference['status'], 0)
    problem = Problem(
        horizon=1,
        num_states=1,
        num_controls=1,
        initial_state=lambda p: jnp.array([1.0]),
        dynamics=lambda x, u, p: x + u,
        stage_cost=lambda x, u, p: jnp.sum(u * u),
        terminal_cost=lambda x, p: jnp.sum((x - 3.0) ** 2),
    )
    primal = problem.initialize({})
    ev = problem.evaluate(primal, {})
    system, rhs = problem.linearize(
        primal, jnp.zeros_like(ev.eq), jnp.zeros_like(ev.eqT), {}
    )
    for backend in ('serial', 'parallel'):
      factors = factorize(system, backend)
      self.assertEqual(int(factors.status), SUCCESS)
      direction = solve_rhs(factors, rhs, backend)
      solution = problem.retract(primal, direction, 1.0, {})
      np.testing.assert_allclose(solution.us[0, 0], reference['u'], atol=1e-9)
      np.testing.assert_allclose(
          solution.xs[:, 0], [reference['x0'], reference['x1']], atol=1e-9
      )
      np.testing.assert_allclose(
          problem.evaluate(solution, {}).objective,
          reference['objective'],
          atol=1e-9,
      )

  def test_regularization_handles_negative_curvature(self):
    problem = Problem(
        horizon=1,
        num_states=1,
        num_controls=1,
        initial_state=lambda p: jnp.array([0.2]),
        dynamics=lambda x, u, p: x + u,
        stage_cost=lambda x, u, p: -2.0 * jnp.sum(u * u),
        terminal_cost=lambda x, p: jnp.sum(x * x),
    )
    primal = problem.initialize({})
    ev = problem.evaluate(primal, {})
    system, _ = problem.linearize(
        primal, jnp.zeros_like(ev.eq), jnp.zeros_like(ev.eqT), {}
    )
    reg = regularization.factorize(system, 0.1, 0.0, Options(), 1, 1, 0)
    self.assertEqual(int(reg.status), SUCCESS)
    self.assertGreater(float(reg.delta_w), 0.0)
    self.assertTrue(bool(regularization.curvature_ok(reg.factors.system)))

  def test_regularization_handles_terminal_state_only_equality(self):
    problem = Problem(
        horizon=1,
        num_states=1,
        num_controls=1,
        initial_state=lambda p: jnp.array([0.2]),
        dynamics=lambda x, u, p: x + u,
        stage_cost=lambda x, u, p: jnp.sum(u * u),
        terminal_cost=lambda x, p: jnp.sum(x * x),
        terminal_equality=lambda x, p: jnp.array([x[0] - 0.7, x[0] - 0.7]),
    )
    primal = problem.initialize({})
    ev = problem.evaluate(primal, {})
    system, _ = problem.linearize(
        primal, jnp.zeros_like(ev.eq), jnp.zeros_like(ev.eqT), {}
    )
    reg = regularization.factorize(system, 0.1, 0.0, Options(), 1, 1, 0)
    self.assertEqual(int(reg.status), SUCCESS)
    self.assertGreater(float(reg.delta_c), 0.0)

  def test_large_native_bound_relaxation_is_capped(self):
    lower = jnp.array([-1e8, -jnp.inf, -2.0])
    upper = jnp.array([1e8, jnp.inf, 4.0])
    lo, hi = jax.jit(bounds.relax)(lower, upper, 1e-8, 1e-6)
    for displacement in (float(lower[0] - lo[0]), float(hi[0] - upper[0])):
      self.assertGreater(displacement, 0.9e-6)
      self.assertLessEqual(displacement, 1.01e-6)
    self.assertTrue(bool(jnp.isneginf(lo[1])))
    self.assertTrue(bool(jnp.isposinf(hi[1])))
    np.testing.assert_allclose(lo[2], -2.0 - 2e-8, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(hi[2], 4.0 + 4e-8, rtol=0.0, atol=1e-15)
    # Exercise the production initializer, not only the bound helper.
    from mujodico.trajopt._src.fatrop_ref.initialization import initialize

    problem = Problem(
        horizon=1,
        num_states=1,
        num_controls=1,
        initial_state=lambda p: jnp.array([0.0]),
        dynamics=lambda x, u, p: x + u,
        stage_cost=lambda x, u, p: jnp.sum(u * u),
        terminal_cost=lambda x, p: jnp.sum(x * x),
        u_bounds=(1e8, 1e8 + 10.0),
    )
    opts = Options(initialize_dual=False, constr_viol_tol=1e-6)
    _, lo, hi, original_lo, original_hi = initialize(problem, {}, None, opts)
    self.assertLessEqual(float(original_lo.us[0, 0] - lo.us[0, 0]), 1.01e-6)
    self.assertLessEqual(float(hi.us[0, 0] - original_hi.us[0, 0]), 1.01e-6)

  def test_regularization_cannot_accept_a_shift_above_maximum(self):
    problem = Problem(
        horizon=1,
        num_states=1,
        num_controls=1,
        initial_state=lambda p: jnp.array([0.2]),
        dynamics=lambda x, u, p: x + u,
        stage_cost=lambda x, u, p: -5e-7 * jnp.sum(u * u),
        terminal_cost=lambda x, p: jnp.sum(x * x),
    )
    primal = problem.initialize({})
    ev = problem.evaluate(primal, {})
    system, _ = problem.linearize(
        primal, jnp.zeros_like(ev.eq), jnp.zeros_like(ev.eqT), {}
    )
    options = Options(delta_w0=1e-4, delta_w_max=1e-6)
    result = regularization.factorize(system, 0.1, 0.0, options, 1, 1, 0)
    self.assertNotEqual(int(result.status), SUCCESS)
    self.assertGreater(float(result.delta_w), options.delta_w_max)
    # The forbidden shift would fix curvature; it still must not be accepted.
    self.assertTrue(bool(regularization.curvature_ok(result.factors.system)))
    self.assertEqual(int(result.attempts), 2)

  def test_jitted_explicit_float32_preserves_dtype_with_global_x64(self):
    from mujodico.trajopt._src.fatrop_ref.solver import IppasSolver
    from mujodico.trajopt._src.fatrop_ref.convergence import Status

    self.assertTrue(jax.config.jax_enable_x64)
    params = {
        'x0': jnp.array([1.0], dtype=jnp.float32),
        'goal': jnp.array([3.0], dtype=jnp.float32),
    }
    solver = IppasSolver(
        horizon=1,
        num_states=1,
        num_controls=1,
        initial_state=lambda p: p['x0'],
        dynamics=lambda x, u, p: x + u,
        stage_cost=lambda x, u, p: jnp.sum(u * u),
        terminal_cost=lambda x, p: jnp.sum((x - p['goal']) ** 2),
        options=Options(
            max_iter=10,
            tol=1e-5,
            linear_tol=1e-6,
            enable_restoration=False,
            enable_watchdog=False,
        ),
    )
    result = jax.jit(solver.solve)(params)
    self.assertEqual(int(result.status), Status.SUCCESS)
    self.assertEqual(result.xs.dtype, jnp.float32)
    self.assertEqual(result.us.dtype, jnp.float32)
    self.assertEqual(result.zl.xs.dtype, jnp.float32)
    self.assertEqual(result.trace.objective.dtype, jnp.float32)
    np.testing.assert_allclose(result.xs, [[1.0], [2.0]], atol=1e-6)
    np.testing.assert_allclose(result.us, [[1.0]], atol=1e-6)


if __name__ == '__main__':
  unittest.main()
