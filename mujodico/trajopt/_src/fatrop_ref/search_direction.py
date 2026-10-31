"""Newton directions, inertia correction, bound condensation and refinement."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from . import iterate as bounds
from . import linear_solver as ls

from .problem import Primal


class Regularized(NamedTuple):
  factors: object
  delta_w: object
  delta_c: object
  attempts: object
  status: object


def shifted(system, delta_w, delta_c, nx, nu, ns):
  """Copies share a primal shift, so elimination gives exactly delta_w I."""
  s = system
  n = s.Q.shape[0]
  wx = jnp.full((n,), 0.5, dtype=s.Q.dtype).at[0].set(1.0)
  wy = jnp.concatenate((
      jnp.ones(nu + ns, dtype=s.Q.dtype),
      jnp.full((nx,), 0.5, dtype=s.Q.dtype),
  ))
  dyn_mask = jnp.concatenate((
      jnp.zeros(nx, dtype=s.Q.dtype),
      jnp.ones(s.D.shape[-1] - nx, dtype=s.Q.dtype),
  ))
  return s._replace(
      Q=s.Q + delta_w * wx[:, None, None] * jnp.eye(nx, dtype=s.Q.dtype),
      R=s.R + delta_w * jnp.diag(wy),
      QT=s.QT + 0.5 * delta_w * jnp.eye(nx, dtype=s.Q.dtype),
      RT=s.RT + delta_w * jnp.eye(s.RT.shape[-1], dtype=s.Q.dtype),
      D=s.D + delta_c * dyn_mask,
      DT=s.DT + delta_c,
  )


def curvature_ok(system):
  w = jnp.concatenate(
      (
          jnp.concatenate((system.Q, system.S), axis=-1),
          jnp.concatenate((jnp.swapaxes(system.S, -1, -2), system.R), axis=-1),
      ),
      axis=-2,
  )
  wt = jnp.block([[system.QT, system.ST], [system.ST.T, system.RT]])
  scale = jnp.maximum(1.0, jnp.max(jnp.abs(w)))
  eps = 100 * jnp.finfo(w.dtype).eps * scale
  return (jnp.min(jnp.linalg.eigvalsh(w)) >= -eps) & (
      jnp.min(jnp.linalg.eigvalsh(wt)) >= -eps
  )


def dual_floor(system):
  """A positive pivot must exceed the factorizer's precision-scaled test."""
  size = max(
      system.R.shape[-1] + system.E.shape[-2],
      system.RT.shape[-1] + system.ET.shape[-2],
      1,
  )
  scale = jnp.maximum(
      1.0,
      jnp.max(
          jnp.stack([
              jnp.max(jnp.abs(a), initial=0.0)
              for a in (system.R, system.F, system.RT, system.FT)
          ])
      ),
  )
  return 10 * jnp.finfo(system.Q.dtype).eps * size * scale


def factorize(system, mu, last_delta, options, nx, nu, ns):
  backend = (
      'parallel' if options.linear_backend == 'pas' else options.linear_backend
  )

  def attempt(dw, dc):
    ss = shifted(system, dw, dc, nx, nu, ns)
    factors = ls.factorize(ss, backend)
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in system]))
    # 3 is curvature failure, kept distinct from rank and nonfinite failures.
    status = jnp.where(
        ~finite, ls.NONFINITE, jnp.where(~curvature_ok(ss), 3, factors.status)
    )
    return factors, status

  zero = jnp.asarray(0.0, system.Q.dtype)
  factors, status = attempt(zero, zero)
  initial = Regularized(factors, zero, zero, jnp.asarray(1, jnp.int32), status)
  maximum = min(options.delta_w_max, float(jnp.finfo(system.Q.dtype).max))

  def cond(r):
    return (
        (r.status != 0)
        & (r.status != ls.NONFINITE)
        & (r.attempts < options.max_regularization)
        & (r.delta_w <= maximum)
    )

  def body(r):
    first = jnp.where(
        last_delta > 0,
        jnp.maximum(last_delta * options.kappa_wmin, options.delta_wmin),
        options.delta_w0,
    )
    dw = jnp.where(
        r.status == 3,
        jnp.where(
            r.delta_w == 0,
            first,
            r.delta_w
            * jnp.where(
                last_delta > 0, options.kappa_wplus, options.kappa_wplusem
            ),
        ),
        r.delta_w,
    )
    dc_new = jnp.maximum(
        options.delta_c * mu**options.kappa_c, dual_floor(r.factors.system)
    )
    # A repeated rank failure after dual damping needs primal damping too.
    dw = jnp.where(
        (r.status == ls.RANK_DEFICIENT) & (r.delta_c > 0),
        jnp.where(dw == 0, first, dw * options.kappa_wplus),
        dw,
    )
    dc = jnp.where(
        r.status == ls.RANK_DEFICIENT, jnp.maximum(dc_new, r.delta_c), r.delta_c
    )
    factors, status = attempt(dw, dc)
    status = jnp.where(dw > maximum, 3, status)
    return Regularized(factors, dw, dc, r.attempts + 1, status)

  return jax.lax.while_loop(cond, body, initial)


def identity_system(system, nx, nu, ns):
  """Identity global primal metric used for least-squares multiplier seeds."""
  s = system._replace(
      Q=jnp.zeros_like(system.Q),
      S=jnp.zeros_like(system.S),
      R=jnp.zeros_like(system.R),
      QT=jnp.zeros_like(system.QT),
      ST=jnp.zeros_like(system.ST),
      RT=jnp.zeros_like(system.RT),
  )
  return shifted(s, 1.0, 0.0, nx, nu, ns)


class SearchResult(NamedTuple):
  direction: object
  dzl: object
  dzu: object
  factors: object
  rhs: object
  delta_w: object
  delta_c: object
  attempts: object
  residual: object
  refinements: object
  status: object


def bound_step(problem, primal, direction):
  nx = problem.nxt
  nu = problem.nut
  ns = primal.s.shape[-1]
  nt = primal.ut.shape[-1]
  sm = problem.state_map.astype(primal.xs.dtype)
  um = problem.control_map.astype(primal.us.dtype)
  return Primal(
      direction.dx @ sm.T,
      direction.dy[:, :nu] @ um.T,
      direction.dy[:, nu : nu + ns],
      direction.dyT[nt:],
      direction.dyT[:nt],
  )


def add_bounds(problem, primal, system, rhs, theta, force):
  sm = problem.state_map.astype(primal.xs.dtype)
  um = problem.control_map.astype(primal.us.dtype)
  nu = problem.nut
  ns = primal.s.shape[-1]
  nt = primal.ut.shape[-1]
  q = jax.vmap(lambda d: sm.T @ (d[:, None] * sm))(theta.xs)
  r = jax.vmap(lambda d: um.T @ (d[:, None] * um))(theta.us)
  rr = system.R.at[:, :nu, :nu].add(r)
  rr = rr.at[:, nu : nu + ns, nu : nu + ns].add(jax.vmap(jnp.diag)(theta.s))
  rt = system.RT.at[:nt, :nt].add(jnp.diag(theta.ut))
  rt = rt.at[nt:, nt:].add(jnp.diag(theta.st))
  gy = rhs.gy.at[:, :nu].add(force.us @ um)
  gy = gy.at[:, nu : nu + ns].add(force.s)
  gty = rhs.gTy.at[:nt].add(force.ut).at[nt:].add(force.st)
  return system._replace(
      Q=system.Q + q[:-1], QT=system.QT + q[-1], R=rr, RT=rt
  ), rhs._replace(
      gx=rhs.gx + force.xs[:-1] @ sm,
      gTx=rhs.gTx + force.xs[-1] @ sm,
      gy=gy,
      gTy=gty,
  )


def residual(system, direction, rhs):
  out = ls.matvec(system, direction)
  return out._replace(**{
      name: (
          getattr(out, name) - getattr(rhs, name)
          if name == 'dx0'
          else getattr(out, name) + getattr(rhs, name)
      )
      for name in out._fields
  })


def refine(factors, rhs, direction, options):
  backend = (
      'parallel' if options.linear_backend == 'pas' else options.linear_backend
  )
  r = residual(factors.system, direction, rhs)
  scale = 1.0 + bounds.tree_max(rhs)
  initial = (
      direction,
      bounds.tree_max(r) / scale,
      jnp.asarray(0, jnp.int32),
      jnp.asarray(True),
  )

  def cond(c):
    return (
        c[3]
        & (c[1] > options.linear_tol)
        & jnp.isfinite(c[1])
        & (c[2] < options.max_refinement)
    )

  def body(c):
    d, err, count, _ = c
    r = residual(factors.system, d, rhs)
    correction_rhs = r._replace(dx0=-r.dx0)
    correction = ls.solve_rhs(factors, correction_rhs, backend)
    trial = jax.tree_util.tree_map(lambda a, b: a + b, d, correction)
    error = bounds.tree_max(residual(factors.system, trial, rhs)) / scale
    good = jnp.isfinite(error) & (error < err)
    d = jax.tree_util.tree_map(lambda a, b: jnp.where(good, a, b), trial, d)
    return d, jnp.where(good, error, err), count + 1, good

  direction, error, count, _ = jax.lax.while_loop(cond, body, initial)
  return direction, error, count


def compute(
    problem,
    primal,
    eq,
    eqT,
    zl,
    zu,
    params,
    mu,
    last_delta,
    lower,
    upper,
    options,
):
  mode = (
      'cost' if options.hessian_mode == 'gauss_newton' else options.hessian_mode
  )
  system, rhs = problem.linearize(
      primal, eq, eqT, params, hessian_mode=mode, stagewise=options.stagewise
  )
  bd = bounds.data(primal, lower, upper)
  theta, force = bounds.condense(bd, zl, zu, mu, options.kappa_d)
  system, rhs = add_bounds(problem, primal, system, rhs, theta, force)
  reg = factorize(
      system,
      mu,
      last_delta,
      options,
      problem.nxt,
      problem.nut,
      primal.s.shape[-1],
  )
  backend = (
      'parallel' if options.linear_backend == 'pas' else options.linear_backend
  )
  direction = ls.solve_rhs(reg.factors, rhs, backend)
  direction, error, count = refine(reg.factors, rhs, direction, options)
  dq = bound_step(problem, primal, direction)
  dzl, dzu = bounds.dual_step(bd, zl, zu, dq, mu)
  status = jnp.where(
      (reg.status == 0)
      & (
          ~jnp.isfinite(error)
          | (error > jnp.maximum(1e-5, options.linear_tol * 1e4))
      ),
      ls.NONFINITE,
      reg.status,
  )
  return SearchResult(
      direction,
      dzl,
      dzu,
      reg.factors,
      rhs,
      reg.delta_w,
      reg.delta_c,
      reg.attempts,
      error,
      count,
      status,
  )
