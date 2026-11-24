"""Dense linear algebra with fusible operations for small matrices.

Unrolled Cholesky, triangular solves, and matrix products let XLA fuse the
operations for matrices of size 1 through SMALL_N. This reduces library kernel
launches in batched GPU dynamics and contact solves. Larger systems use the
JAX library routines. The operations support differentiation and vmap.
"""

import jax
from jax import numpy as jp

SMALL_N = 16


def is_small(n: int) -> bool:
  return 0 < n <= SMALL_N


def mv(A, x):
  """A @ x for a matrix A and a vector x."""
  if is_small(A.shape[-1]):
    return jp.sum(A * x[None, :], axis=-1)
  return A @ x


def mtv(A, x):
  """A.T @ x."""
  if is_small(A.shape[-1]):
    return jp.sum(A * x[:, None], axis=0)
  return A.T @ x


def mm(A, B):
  """A @ B for 2-D A, B."""
  if is_small(max(A.shape[-1], B.shape[-1])):
    return jp.sum(A[:, :, None] * B[None, :, :], axis=1)
  return A @ B


def cho_factor(A):
  """Return the lower Cholesky factor of SPD A as ``(L, True)``.

  The result can be passed to :func:`cho_solve`, following the contract of
  ``jax.scipy.linalg.cho_factor``.
  """
  n = A.shape[-1]
  if not is_small(n):
    return jax.scipy.linalg.cho_factor(A, lower=True)
  # Entries are built as a list of rows (no scatter updates).
  L = [[None] * n for _ in range(n)]
  for j in range(n):
    s = A[j, j]
    for k in range(j):
      s = s - L[j][k] * L[j][k]
    d = jp.sqrt(s)
    L[j][j] = d
    for i in range(j + 1, n):
      s = A[i, j]
      for k in range(j):
        s = s - L[i][k] * L[j][k]
      L[i][j] = s / d
  z = jp.zeros_like(A[0, 0])
  return (jp.stack([jp.stack([L[i][j] if j <= i else z for j in range(n)])
                    for i in range(n)]), True)


def cho_solve(c, b):
  """Solve ``A x = b`` given ``c = cho_factor(A)``.

  The right-hand side b is a vector or a matrix of column vectors.
  """
  L, lower = c
  n = L.shape[-1]
  if not is_small(n) or not lower:
    return jax.scipy.linalg.cho_solve(c, b)
  y = []
  for i in range(n):   # L y = b
    s = b[i]
    for k in range(i):
      s = s - L[i, k] * y[k]
    y.append(s / L[i, i])
  x = [None] * n
  for i in reversed(range(n)):   # L^T x = y
    s = y[i]
    for k in range(i + 1, n):
      s = s - L[k, i] * x[k]
    x[i] = s / L[i, i]
  return jp.stack(x)


def spd_solve(A, b):
  """Compute ``A^{-1} b`` for SPD A and vector or matrix b by Cholesky.

  Small systems differentiate through the unrolled Cholesky solve. The
  library path uses ``dx = A^{-1} (db - dA x)`` and reuses the inverse across
  tangent directions to reduce batched triangular solves on the GPU.
  """
  if is_small(A.shape[-1]):
    return cho_solve(cho_factor(A), b)
  return _spd_solve(A, b)


@jax.custom_jvp
def _spd_solve(A, b):
  return cho_solve(cho_factor(A), b)


@_spd_solve.defjvp
def _spd_solve_jvp(primals, tangents):
  A, b = primals
  dA, db = tangents
  c = cho_factor(A)
  x = cho_solve(c, b)
  n = A.shape[-1]
  Ainv = cho_solve(c, jp.eye(n, dtype=A.dtype))
  return x, Ainv @ (db - dA @ x)
