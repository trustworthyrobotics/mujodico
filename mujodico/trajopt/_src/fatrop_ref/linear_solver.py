"""Reusable serial/associative Newton solves derived from IP-PAS.tex.

The equations and signs are documented by ``matvec``. Matrices are stagewise;
no horizon-sized dense matrices are formed. The only decompositions during an
RHS solve use factors already computed by ``factorize``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp


class Segment(NamedTuple):
  P: jax.Array
  p: jax.Array
  A: jax.Array
  b: jax.Array
  C: jax.Array


def compose(left: Segment, right: Segment) -> Segment:
  """Eliminate the shared boundary of consecutive segments (batched)."""
  eye = jnp.eye(left.P.shape[-1], dtype=left.P.dtype)
  U = eye + right.P @ left.C
  V = eye + left.C @ right.P
  PA = jnp.linalg.solve(U, right.P @ left.A)
  z = jnp.linalg.solve(
      U, (right.p + (right.P @ left.b[..., None])[..., 0])[..., None]
  )[..., 0]
  VA = jnp.linalg.solve(V, left.A)
  vb = jnp.linalg.solve(
      V, (left.b - (left.C @ right.p[..., None])[..., 0])[..., None]
  )[..., 0]
  VC = jnp.linalg.solve(V, left.C)
  AT = jnp.swapaxes(left.A, -1, -2)
  return Segment(
      left.P + AT @ PA,
      left.p + (AT @ z[..., None])[..., 0],
      right.A @ VA,
      right.b + (right.A @ vb[..., None])[..., 0],
      right.C + right.A @ VC @ jnp.swapaxes(right.A, -1, -2),
  )


def serial_backward(P, A, C, PT):
  """Return all suffix value Hessians, including the terminal one."""

  def step(next_P, inputs):
    stage_P, stage_A, stage_C = inputs
    U = jnp.eye(PT.shape[-1], dtype=PT.dtype) + next_P @ stage_C
    current = stage_P + stage_A.T @ jnp.linalg.solve(U, next_P @ stage_A)
    current = (current + current.T) * 0.5
    return current, current

  _, values = jax.lax.scan(step, PT, (P, A, C), reverse=True)
  return jnp.concatenate((values, PT[None]), axis=0)


def serial_affine_suffix(T, d, terminal):
  def step(value, inputs):
    matrix, offset = inputs
    value = matrix @ value + offset
    return value, value

  _, values = jax.lax.scan(step, terminal, (T, d), reverse=True)
  return jnp.concatenate((values, terminal[None]), axis=0)


def serial_affine_forward(F, a, x0):
  def step(value, inputs):
    matrix, offset = inputs
    value = matrix @ value + offset
    return value, value

  _, values = jax.lax.scan(step, x0, (F, a))
  return jnp.concatenate((x0[None], values), axis=0)


class _SerialScan:
  backward = staticmethod(serial_backward)
  affine_suffix = staticmethod(serial_affine_suffix)
  affine_forward = staticmethod(serial_affine_forward)


def parallel_backward(P, A, C, PT):
  n, nx = P.shape[:2]
  z = jnp.zeros((n + 1, nx), dtype=PT.dtype)
  zm = jnp.zeros((1, nx, nx), dtype=PT.dtype)
  elements = Segment(
      jnp.concatenate((P, PT[None])),
      z,
      jnp.concatenate((A, zm)),
      z,
      jnp.concatenate((C, zm)),
  )
  # Reverse the array and the operation, preserving chronological composition.
  reversed_elements = jax.tree.map(lambda x: x[::-1], elements)
  suffix = jax.lax.associative_scan(
      lambda later, earlier: compose(earlier, later), reversed_elements
  )
  values = suffix.P[::-1]
  return (values + jnp.swapaxes(values, -1, -2)) * 0.5


def parallel_affine_suffix(T, d, terminal):
  nx = terminal.shape[0]
  matrices = jnp.concatenate((T, jnp.zeros((1, nx, nx), dtype=T.dtype)))
  offsets = jnp.concatenate((d, terminal[None]))

  def reverse_combine(later, earlier):
    lm, ld = later
    em, ed = earlier
    return em @ lm, ed + (em @ ld[..., None])[..., 0]

  _, result = jax.lax.associative_scan(
      reverse_combine, (matrices[::-1], offsets[::-1])
  )
  return result[::-1]


def parallel_affine_forward(F, a, x0):
  if F.shape[0] == 0:
    return x0[None]

  def combine(earlier, later):
    em, ed = earlier
    lm, ld = later
    return lm @ em, ld + (lm @ ed[..., None])[..., 0]

  matrices, offsets = jax.lax.associative_scan(combine, (F, a))
  result = (matrices @ x0[..., None])[..., 0] + offsets
  return jnp.concatenate((x0[None], result))


class _ParallelScan:
  backward = staticmethod(parallel_backward)
  affine_suffix = staticmethod(parallel_affine_suffix)
  affine_forward = staticmethod(parallel_affine_forward)


SUCCESS = 0
NONFINITE = 1
RANK_DEFICIENT = 2


class System(NamedTuple):
  Q: jax.Array
  S: jax.Array
  R: jax.Array
  A: jax.Array
  B: jax.Array
  E: jax.Array
  F: jax.Array
  D: jax.Array
  QT: jax.Array
  ST: jax.Array
  RT: jax.Array
  ET: jax.Array
  FT: jax.Array
  DT: jax.Array


class RHS(NamedTuple):
  gx: jax.Array
  gy: jax.Array
  c: jax.Array
  b: jax.Array
  gTx: jax.Array
  gTy: jax.Array
  cT: jax.Array
  dx0: jax.Array


class Direction(NamedTuple):
  dx: jax.Array
  dy: jax.Array
  beta: jax.Array
  betaT: jax.Array
  dyT: jax.Array
  lam: jax.Array


class Factors(NamedTuple):
  system: System
  lu: jax.Array
  piv: jax.Array
  luT: jax.Array
  pivT: jax.Array
  K: jax.Array
  L: jax.Array
  KT: jax.Array
  MinvK: jax.Array
  MinvL: jax.Array
  MTinvKT: jax.Array
  P: jax.Array
  Abar: jax.Array
  C: jax.Array
  PV: jax.Array
  interface_lu: jax.Array
  interface_piv: jax.Array
  Fforward: jax.Array
  Tbackward: jax.Array
  status: jax.Array


def _scan(backend):
  if backend == 'serial':
    return _SerialScan
  if backend == 'parallel':
    return _ParallelScan
  raise ValueError("backend must be 'serial' or 'parallel'")


def _transpose(x):
  return jnp.swapaxes(x, -1, -2)


def _mv(a, x):
  return (a @ x[..., None])[..., 0]


def _batch_factor(a):
  if a.shape[-1] == 0:
    return a, jnp.zeros(a.shape[:-1], dtype=jnp.int32)
  if a.ndim == 3:
    return jax.vmap(jsp.lu_factor)(a)
  return jsp.lu_factor(a)


def _solve(lu, piv, rhs):
  if lu.shape[-1] == 0:
    return rhs
  if lu.ndim == 3:
    return jax.vmap(jsp.lu_solve)((lu, piv), rhs)
  return jsp.lu_solve((lu, piv), rhs)


def _rank_bad(matrix, lu):
  if matrix.shape[-1] == 0 or (matrix.ndim == 3 and matrix.shape[0] == 0):
    return jnp.array(False)
  size = matrix.shape[-1]
  scale = jnp.max(jnp.abs(matrix), axis=(-2, -1))
  tol = (
      jnp.finfo(matrix.dtype).eps
      * size
      * jnp.maximum(scale, jnp.finfo(matrix.dtype).tiny)
  )
  return jnp.any(
      jnp.abs(jnp.diagonal(lu, axis1=-2, axis2=-1)) <= tol[..., None]
  )


def factorize(system: System, backend='parallel') -> Factors:
  """Cache local and boundary factors; ``status`` reports unusable matrices.

  Local equality damping has the sign ``-D``. D may be zero on independent
  equalities; positive damping makes deficient equality rows eliminable.
  """
  scan = _scan(backend)
  s = system
  n, nx = s.Q.shape[:2]
  ny, ne = s.R.shape[-1], s.E.shape[-2]
  nt, net = s.RT.shape[-1], s.ET.shape[-2]
  M = jnp.concatenate(
      (
          jnp.concatenate((s.R, _transpose(s.F)), axis=-1),
          jnp.concatenate((s.F, -jax.vmap(jnp.diag)(s.D)), axis=-1),
      ),
      axis=-2,
  )
  K = jnp.concatenate((_transpose(s.S), s.E), axis=-2)
  L = jnp.concatenate(
      (_transpose(s.B), jnp.zeros((n, ne, nx), s.Q.dtype)), axis=-2
  )
  MT = jnp.block([[s.RT, s.FT.T], [s.FT, -jnp.diag(s.DT)]])
  KT = jnp.concatenate((s.ST.T, s.ET), axis=0)
  lu, piv = _batch_factor(M)
  luT, pivT = _batch_factor(MT)
  MinvK, MinvL = _solve(lu, piv, K), _solve(lu, piv, L)
  MTinvKT = _solve(luT, pivT, KT)
  P = s.Q - _transpose(K) @ MinvK
  Abar = s.A - _transpose(L) @ MinvK
  C = _transpose(L) @ MinvL
  P = (P + _transpose(P)) * 0.5
  C = (C + _transpose(C)) * 0.5
  PT = s.QT - KT.T @ MTinvKT
  PT = (PT + PT.T) * 0.5
  PV = scan.backward(P, Abar, C, PT)
  interface = jnp.eye(nx, dtype=s.Q.dtype) + C @ PV[1:]
  interface_lu, interface_piv = _batch_factor(interface)
  Fforward = _solve(interface_lu, interface_piv, Abar)
  # (I + P C)^T = I + C P for symmetric P, C.
  Tbackward = _transpose(Fforward)
  # Tbackward must be Abar^T (I + P C)^-1, which is Fforward^T.
  all_finite = jnp.array(True)
  for array in s:
    all_finite = all_finite & jnp.all(jnp.isfinite(array))
  rank_bad = (
      _rank_bad(M, lu) | _rank_bad(MT, luT) | _rank_bad(interface, interface_lu)
  )
  factors_finite = jnp.all(jnp.isfinite(PV)) & jnp.all(jnp.isfinite(Fforward))
  status = jnp.where(
      ~all_finite,
      NONFINITE,
      jnp.where(rank_bad | ~factors_finite, RANK_DEFICIENT, SUCCESS),
  )
  return Factors(
      s,
      lu,
      piv,
      luT,
      pivT,
      K,
      L,
      KT,
      MinvK,
      MinvL,
      MTinvKT,
      P,
      Abar,
      C,
      PV,
      interface_lu,
      interface_piv,
      Fforward,
      Tbackward,
      status.astype(jnp.int32),
  )


def solve_rhs(factors: Factors, rhs: RHS, backend='parallel') -> Direction:
  """Solve a new RHS without decomposing any matrix again."""
  scan = _scan(backend)
  f, r = factors, rhs
  s = f.system
  n, ny = s.Q.shape[0], s.R.shape[-1]
  nt = s.RT.shape[-1]
  e = jnp.concatenate((r.gy, r.c), axis=-1)
  eT = jnp.concatenate((r.gTy, r.cT))
  Minve = _solve(f.lu, f.piv, e)
  MTinve = _solve(f.luT, f.pivT, eT)
  p = r.gx - _mv(_transpose(f.K), Minve)
  b = r.b - _mv(_transpose(f.L), Minve)
  pT = r.gTx - f.KT.T @ MTinve
  d = p + _mv(f.Tbackward @ f.PV[1:], b)
  pV = scan.affine_suffix(f.Tbackward, d, pT)
  a = _solve(f.interface_lu, f.interface_piv, b - _mv(f.C, pV[1:]))
  dx = scan.affine_forward(f.Fforward, a, r.dx0)
  lam = _mv(f.PV, dx) + pV
  z = -_mv(f.MinvK, dx[:-1]) - _mv(f.MinvL, lam[1:]) - Minve
  zT = -f.MTinvKT @ dx[-1] - MTinve
  return Direction(dx, z[:, :ny], z[:, ny:], zT[nt:], zT[:nt], lam)


def matvec(system: System, direction: Direction) -> RHS:
  """Apply the regularized lifted KKT operator.

  A solved RHS obeys gx_out=-rhs.gx, gy_out=-rhs.gy, c_out=-rhs.c,
  b_out=-rhs.b, terminal analogues, and dx0_out=rhs.dx0.
  The initial costate enforces stationarity of the prescribed initial state.
  """
  s, v = system, direction
  gx = (
      _mv(s.Q, v.dx[:-1])
      + _mv(s.S, v.dy)
      + _mv(_transpose(s.E), v.beta)
      + _mv(_transpose(s.A), v.lam[1:])
      - v.lam[:-1]
  )
  gy = (
      _mv(s.R, v.dy)
      + _mv(_transpose(s.S), v.dx[:-1])
      + _mv(_transpose(s.F), v.beta)
      + _mv(_transpose(s.B), v.lam[1:])
  )
  c = _mv(s.E, v.dx[:-1]) + _mv(s.F, v.dy) - s.D * v.beta
  b = _mv(s.A, v.dx[:-1]) + _mv(s.B, v.dy) - v.dx[1:]
  gTx = s.QT @ v.dx[-1] + s.ST @ v.dyT + s.ET.T @ v.betaT - v.lam[-1]
  gTy = s.RT @ v.dyT + s.ST.T @ v.dx[-1] + s.FT.T @ v.betaT
  cT = s.ET @ v.dx[-1] + s.FT @ v.dyT - s.DT * v.betaT
  return RHS(gx, gy, c, b, gTx, gTy, cT, v.dx[0])


def residual(system: System, direction: Direction, rhs: RHS) -> RHS:
  """Residual of every equation, including the prescribed initial increment."""
  value = matvec(system, direction)
  return RHS(
      *(a + b for a, b in zip(value[:-1], rhs[:-1])), value.dx0 - rhs.dx0
  )
