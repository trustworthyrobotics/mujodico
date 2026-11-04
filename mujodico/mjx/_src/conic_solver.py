"""
Author: Wei-Chen Li

Solves the conic program
  minimize    ½ xᵀ P x + qᵀ x
  subject to  A x + s = b
              s ∈ 𝒦
cones are either zero cone, non-negative cone or second-order cone.

References
[cvxopt] The CVXOPT linear and quadratic cone program solver. Available at
         https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
"""

import dataclasses
import functools
import warnings

import jax
import jax.numpy as jp


@dataclasses.dataclass(frozen=True)
class ZeroCone:
  dim: int

  def __post_init__(self):
    if self.dim <= 0:
      raise ValueError('ZeroCone requires dim >= 1')


@dataclasses.dataclass(frozen=True)
class NonNegativeCone:
  dim: int

  def __post_init__(self):
    if self.dim <= 0:
      raise ValueError('NonNegativeCone requires dim >= 1')

  @property
  def deg(self):
    return self.dim

  @property
  def jordan_identity(self):
    return jp.ones(self.dim)

  def _jordan_product(self, s: jax.Array, z: jax.Array) -> jax.Array:
    return s * z

  def _inverse_jordan_product(self, u: jax.Array, v: jax.Array) -> jax.Array:
    return v / u

  def _project(self, s: jax.Array, margin: float) -> jax.Array:
    """[cvxopt, sec 5.3]"""
    return s + jp.maximum(0.0, margin - jp.min(s))

  def _line_search(self, s: jax.Array, ds: jax.Array) -> jax.Array:
    """[cvxopt, sec 8.1]"""
    return 1.0 / jp.maximum(jp.max(-ds / s), 0.0)

  def _compute_scaling(
      self, s: jax.Array, z: jax.Array
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """[cvxopt, sec 4.1]"""
    w = jp.sqrt(s / z)
    return jp.diag(w), jp.diag(1.0 / w), jp.sqrt(s * z)

  def _update_scaling(
      self,
      W: jax.Array,
      Winv: jax.Array,
      s_scaled: jax.Array,
      z_scaled: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """[cvxopt, sec 9.1]"""
    s_sqrt = jp.sqrt(s_scaled)
    z_sqrt = jp.sqrt(z_scaled)
    w = s_sqrt / z_sqrt * jp.diag(W)
    return jp.diag(w), jp.diag(1.0 / w), s_sqrt * z_sqrt


@dataclasses.dataclass(frozen=True)
class SecondOrderCone:
  dim: int

  def __post_init__(self):
    if self.dim < 2:
      raise ValueError('SecondOrderCone requires dim >= 2')

  @property
  def deg(self):
    return 1

  @property
  def jordan_identity(self):
    return jp.zeros(self.dim).at[0].set(1.0)

  @staticmethod
  def _norm(s: jax.Array) -> jax.Array:
    """Lorentz norm of a vector in the interior of a second-order cone."""
    tail_norm = jp.linalg.norm(s[1:])
    determinant = (s[0] - tail_norm) * (s[0] + tail_norm)
    return jp.sqrt(jp.maximum(determinant, jp.finfo(s.dtype).tiny))

  def _jordan_product(self, s: jax.Array, z: jax.Array) -> jax.Array:
    return jp.concatenate((
        jp.sum(s * z, keepdims=True),
        s[0] * z[1:] + z[0] * s[1:]
    ))

  def _inverse_jordan_product(self, u: jax.Array, v: jax.Array) -> jax.Array:
    """[cvxopt, sec 5.4]"""
    u0, u1 = u[0], u[1:]
    v0, v1 = v[0], v[1:]

    det_u = self._norm(u) ** 2

    r0 = u0 * v0 - jp.dot(u1, v1)
    r1 = -u1 * v0 + ((det_u * v1) + (u1 * jp.dot(u1, v1))) / u0

    return jp.concatenate((jp.array([r0]), r1)) / det_u

  def _project(self, s: jax.Array, margin: float) -> jax.Array:
    """[cvxopt, sec 5.3]"""
    shift = jp.maximum(0.0, margin + jp.linalg.norm(s[1:]) - s[0])
    return s.at[0].add(shift)

  def _line_search(self, s: jax.Array, ds: jax.Array) -> jax.Array:
    """[cvxopt, sec 8.2]"""

    # Map s to the Jordan identity. The eigenvalues of the transformed
    # direction are rho[0] ± ||rho[1:]||, so its smaller eigenvalue bounds α.
    norm = self._norm(s)
    sn = s / norm
    inner = sn[0] * ds[0] - jp.dot(sn[1:], ds[1:])
    rho0 = inner / norm
    rho1 = (ds[1:] - (inner + ds[0]) / (sn[0] + 1.0) * sn[1:]) / norm
    bound = 1.0 / jp.maximum(jp.linalg.norm(rho1) - rho0, 0.0)
    return jp.where(ds[0] >= jp.linalg.norm(ds[1:]), jp.inf, bound)

  def _compute_scaling(
      self, s: jax.Array, z: jax.Array
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """[cvxopt, sec 4.2]"""
    assert s.size == self.dim
    assert z.size == self.dim
    s_norm, z_norm = self._norm(s), self._norm(z)
    sn, zn = s / s_norm, z / z_norm
    reflection = jp.ones_like(s).at[1:].set(-1.0)
    identity = jp.zeros_like(s).at[0].set(1.0)

    w = (sn + reflection * zn) / jp.sqrt(2.0 * (1.0 + jp.dot(sn, zn)))
    v = (w + identity) / jp.sqrt(2.0 * (w[0] + 1.0))
    beta = jp.sqrt(s_norm / z_norm)
    J = jp.diag(reflection)
    W = beta * (2.0 * jp.outer(v, v) - J)
    Jv = reflection * v
    Winv = (2.0 * jp.outer(Jv, Jv) - J) / beta
    return W, Winv, W @ z

  def _update_scaling(
      self,
      W: jax.Array,
      Winv: jax.Array,
      s_scaled: jax.Array,
      z_scaled: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """[cvxopt, sec 9.2]"""
    T, Tinv, lmbda = self._compute_scaling(s_scaled, z_scaled)
    return T @ W, Winv @ Tinv, lmbda


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        'P',
        'q',
        'A',
        'b',
        'G',
        'h',
        'comps',
        'x',
        'y',
        'z',
        's',
        'W',
        'Winv',
        'lmbda',
        'lu_kkt',
    ],
    meta_fields=['cones', 'dim', 'iterations', 'tolerance'],
)
@dataclasses.dataclass(frozen=True)
class Context:
  P: jax.Array
  q: jax.Array
  A: jax.Array
  b: jax.Array
  G: jax.Array
  h: jax.Array
  cones: tuple[NonNegativeCone | SecondOrderCone]
  comps: tuple[jax.Array]
  x: jax.Array
  y: jax.Array
  z: jax.Array
  s: jax.Array
  W: jax.Array
  Winv: jax.Array
  lmbda: jax.Array
  dim: int
  iterations: int
  tolerance: float
  lu_kkt: tuple[jax.Array, jax.Array] | None = None

  _ALLOWED_FIELDS = frozenset({
      'x',
      'y',
      'z',
      's',
      'W',
      'Winv',
      'lmbda',
      'lu_kkt',
  })

  def replace(self, **kwargs):
    invalid = kwargs.keys() - self._ALLOWED_FIELDS
    if invalid:
      raise ValueError(f'Attempt replacing immutable fields: {invalid}')
    return dataclasses.replace(self, **kwargs)


def _initialize(**kwargs) -> Context:
  """Initializes the context from problem parameters. [cvxopt, sec 5.3]"""

  dim = sum([cone.dim for cone in kwargs['cones']])
  W = jp.eye(dim)
  Winv = jp.eye(dim)

  ctx = Context(
      P=kwargs['P'],
      q=kwargs['q'],
      A=kwargs['A'],
      b=kwargs['b'],
      G=kwargs['G'],
      h=kwargs['h'],
      cones=kwargs['cones'],
      comps=kwargs['comps'],
      x=kwargs['x'],
      y=kwargs['y'],
      z=kwargs['z'],
      s=kwargs['s'],
      W=W,
      Winv=Winv,
      lmbda=jp.zeros(dim),
      dim=dim,
      iterations=kwargs['iterations'],
      tolerance=kwargs['tolerance'],
  )

  if any([var is None for var in (ctx.x, ctx.y, ctx.z)]):
    ctx, x, y, z = _solve_kkt(ctx=ctx, bx=-ctx.q, by=ctx.b, bz=ctx.h)
    ctx = ctx.replace(x=x, y=y, z=z)

  if ctx.s is None:
    s = ctx.h - ctx.G @ ctx.x
    ctx = ctx.replace(s=s)

  z, s = ctx.z, ctx.s
  i = 0
  for cone in ctx.cones:
    end = i + cone.dim
    z = z.at[i:end].set(cone._project(z[i:end], margin=1.0))
    s = s.at[i:end].set(cone._project(s[i:end], margin=1.0))
    i = end
  ctx = ctx.replace(z=z, s=s)

  ctx = _compute_scaling(ctx)

  return ctx


def _compute_scaling(ctx: Context) -> Context:
  """Compute W so that W z = W⁻ᵀ s = λ. [cvxopt, sec 4]"""

  W, Winv, lmbda = ctx.W, ctx.Winv, ctx.lmbda

  i = 0
  for cone in ctx.cones:
    end = i + cone.dim

    Wk, Wkinv, lmbdak = cone._compute_scaling(ctx.s[i:end], ctx.z[i:end])
    W = W.at[i:end, i:end].set(Wk)
    Winv = Winv.at[i:end, i:end].set(Wkinv)
    lmbda = lmbda.at[i:end].set(lmbdak)

    i = end

  ctx = ctx.replace(W=W, Winv=Winv, lmbda=lmbda, lu_kkt=None)

  return ctx


def _update_scaling(
    ctx: Context, dz: jax.Array, ds: jax.Array, alpha: jax.Array
) -> Context:
  """Update W and λ so that W (z + α dz) = W⁻ᵀ (s + α ds) = λ. [cvxopt, sec 9]"""

  W, Winv, lmbda = ctx.W, ctx.Winv, ctx.lmbda

  i = 0
  for cone in ctx.cones:
    end = i + cone.dim

    Wk = W[i:end, i:end]
    Wkinv = Winv[i:end, i:end]

    sk_scaled = ctx.lmbda[i:end] + alpha * (Wkinv.T @ ds[i:end])
    zk_scaled = ctx.lmbda[i:end] + alpha * (Wk @ dz[i:end])

    Wk, Wkinv, lmbdak = cone._update_scaling(Wk, Wkinv, sk_scaled, zk_scaled)
    W = W.at[i:end, i:end].set(Wk)
    Winv = Winv.at[i:end, i:end].set(Wkinv)
    lmbda = lmbda.at[i:end].set(lmbdak)

    i = end

  ctx = ctx.replace(W=W, Winv=Winv, lmbda=lmbda, lu_kkt=None)

  return ctx


def _mu_e(cones, comps: tuple[jax.Array]) -> jax.Array:
  """Returns μ e"""
  result = []

  i = 0
  for cone, comp in zip(cones, comps):
    end = i + cone.dim
    result.append(comp * cone.jordan_identity)
    i = end

  return jp.concatenate(result) if result else jp.zeros(0)


def _jordan_product(cones, s: jax.Array, z: jax.Array) -> jax.Array:
  """Computes s ∘ z"""
  result = []

  i = 0
  for cone in cones:
    end = i + cone.dim
    result.append(cone._jordan_product(s[i:end], z[i:end]))
    i = end

  return jp.concatenate(result) if result else jp.zeros(0)


def _inverse_jordan_product(cones, u: jax.Array, v: jax.Array) -> jax.Array:
  """Computes u ⋄ v. The operator ⋄ satisfies s ⋄ (s ∘ z) = z"""
  result = []

  i = 0
  for cone in cones:
    end = i + cone.dim
    result.append(cone._inverse_jordan_product(u[i:end], v[i:end]))
    i = end

  return jp.concatenate(result) if result else jp.zeros(0)


def _line_search(ctx: Context, dz: jax.Array, ds: jax.Array) -> jax.Array:
  """Computes maximum α so that z + α Δz ∈ 𝒦 and s + α Δs ∈ 𝒦. [cvxopt, sec 8]"""
  alpha = jp.ones(1)

  i = 0
  for cone in ctx.cones:
    end = i + cone.dim
    alpha = jp.minimum(alpha, 0.99 * cone._line_search(ctx.z[i:end], dz[i:end]))
    alpha = jp.minimum(alpha, 0.99 * cone._line_search(ctx.s[i:end], ds[i:end]))
    i = end

  return alpha


def _optimality_conditions(
    P, q, A, b, G, h, cones, comps, x, y, z, s
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """
  Computes the optimality values:
    P x + Aᵀ y + Gᵀ z + q
    A x               - b
    G x           + s - h
    s ∘ z - μ e
  """
  return (
      P @ x + A.T @ y + G.T @ z + q,
      A @ x - b,
      G @ x + s - h,
      _jordan_product(cones, s, z) - _mu_e(cones, comps),
  )


def _residual(
    ctx: Context,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Returns the KKT residual."""
  rx, ry, rz, _ = _optimality_conditions(
      ctx.P,
      ctx.q,
      ctx.A,
      ctx.b,
      ctx.G,
      ctx.h,
      ctx.cones,
      ctx.comps,
      ctx.x,
      ctx.y,
      ctx.z,
      ctx.s,
  )
  # For SOC blocks, λ ∘ λ generally differs from the raw product s ∘ z
  rs = _jordan_product(ctx.cones, ctx.lmbda, ctx.lmbda) - _mu_e(ctx.cones, ctx.comps)
  return rx, ry, rz, rs


def _solve_kkt(
    ctx: Context, bx: jax.Array, by: jax.Array, bz: jax.Array,
    iterative_refinements: int = 2,
) -> tuple[Context, jax.Array, jax.Array, jax.Array]:
  """Solves the KKT system using its augmented scaled form.

  The original equations are:
    P x + Aᵀ y + Gᵀ z = bx
    A x               = by
    G x        -WᵀW z = bz

  With u = W z and Gs = W⁻ᵀ G, the linear system is:
    P  x + Aᵀ x + Gsᵀ u = bx
    A  x                = by
    Gs x        -  I  u = W⁻ᵀ bz

  Returns the updated context (caches the LU decomposition) and x, y, z.
  """
  n, neq, dim = ctx.q.size, ctx.b.size, ctx.dim
  dtype = bx.dtype

  Gs = ctx.Winv.T @ ctx.G

  kkt = jp.block([
      [ctx.P, ctx.A.T, Gs.T],
      [
          ctx.A,
          jp.zeros((neq, neq), dtype=dtype),
          jp.zeros((neq, dim), dtype=dtype),
      ],
      [
          Gs,
          jp.zeros((dim, neq), dtype=dtype),
          -jp.eye(dim, dtype=dtype),
      ],
  ])

  lu_kkt = ctx.lu_kkt
  if lu_kkt is None:
    lu_kkt = jax.scipy.linalg.lu_factor(kkt)
    ctx = ctx.replace(lu_kkt=lu_kkt)

  rhs = jp.concatenate((bx, by, ctx.Winv.T @ bz))
  solution = jax.scipy.linalg.lu_solve(lu_kkt, rhs)
  for _ in range(iterative_refinements):
    residual = rhs - kkt @ solution
    solution += jax.scipy.linalg.lu_solve(lu_kkt, residual)

  x, y, u = jp.split(solution, (n, n + neq))
  return ctx, x, y, ctx.Winv @ u


def _solve_newton(
    ctx: Context, bx: jax.Array, by: jax.Array, bz: jax.Array, bs: jax.Array
) -> tuple[Context, jax.Array, jax.Array, jax.Array, jax.Array]:
  """
  Solves the newton step:
    P x + Aᵀ y + Gᵀ z = bx
    A x               = by
    G x           + s = bz
    λ ∘ (W z + W⁻ᵀ s) = bs
  """

  WT_lmbdainv_bs = ctx.W.T @ _inverse_jordan_product(ctx.cones, ctx.lmbda, bs)
  ctx, x, y, z = _solve_kkt(ctx, bx, by, bz - WT_lmbdainv_bs)
  s = bz - ctx.G @ x
  return ctx, x, y, z, s


def _corrector_mu_e(
    ctx: Context, ds: jax.Array, dz: jax.Array, alpha: jax.Array
) -> jax.Array:
  """Computes the σ μ̂ e quantity in [cvxopt, eq. 19].
  Adapted because the target mu is user specified and not necessarily zero."""
  result = []

  i = 0
  for cone, comp in zip(ctx.cones, ctx.comps):
    end = i + cone.dim

    sk = ctx.s[i:end]
    zk = ctx.z[i:end]
    sk_pred = sk + alpha * ds[i:end]
    zk_pred = zk + alpha * dz[i:end]

    if isinstance(cone, NonNegativeCone):
      mu0 = sk * zk
      mu1 = sk_pred * zk_pred
    else:
      mu0 = jp.dot(sk, zk) / cone.deg
      mu1 = jp.dot(sk_pred, zk_pred) / cone.deg
    mu_target = comp

    rho = jp.abs(mu1 - mu_target) / jp.maximum(jp.abs(mu0 - mu_target), jp.finfo(ds.dtype).tiny)

    # if mu1 is closer to mu_target than mu0, sigma is close to 0.
    # if mu0 is closer to mu_target than mu1, sigma is close to 1.
    sigma = jp.clip(rho**3, 0.1, 1.0)

    result.append((sigma * mu0 + (1 - sigma) * mu_target) * cone.jordan_identity)

    i = end

  return jp.concatenate(result) if result else jp.zeros(0)


def _solve_impl(
    P: jax.Array,
    q: jax.Array,
    A: jax.Array,
    b: jax.Array,
    G: jax.Array,
    h: jax.Array,
    cones: tuple[NonNegativeCone | SecondOrderCone],
    comps: tuple[jax.Array],
    *,
    x_init: jax.Array | None = None,
    y_init: jax.Array | None = None,
    z_init: jax.Array | None = None,
    s_init: jax.Array | None = None,
    iterations: int = 100,
    tolerance: float = 1e-8,
) -> Context:
  """Implements the algorithm in [cvxopt, sec 5.1]"""

  ctx = _initialize(
      P=P,
      q=q,
      A=A,
      b=b,
      G=G,
      h=h,
      cones=cones,
      comps=comps,
      x=x_init,
      y=y_init,
      z=z_init,
      s=s_init,
      iterations=iterations,
      tolerance=tolerance,
  )

  rx, ry, rz, rs = _residual(ctx)
  init_val = (ctx, rx, ry, rz, rs, 0)

  def cond_fun(val):
    ctx, rx, ry, rz, rs, iter = val
    return (iter < ctx.iterations) & (
        (jp.linalg.norm(rx, jp.inf) > ctx.tolerance)
        | (jp.linalg.norm(ry, jp.inf) > ctx.tolerance)
        | (jp.linalg.norm(rz, jp.inf) > ctx.tolerance)
        | (jp.linalg.norm(rs, jp.inf) > ctx.tolerance)
    )

  def body_fun(val):
    ctx, rx, ry, rz, rs, iter = val

    ctx, dx, dy, dz, ds = _solve_newton(ctx, -rx, -ry, -rz, -rs)
    alpha = _line_search(ctx, dz, ds)

    bs = (
        - _jordan_product(ctx.cones, ctx.lmbda, ctx.lmbda)
        - _jordan_product(ctx.cones, ctx.Winv.T @ ds, ctx.W @ dz)
        + _corrector_mu_e(ctx, ds, dz, alpha)
    )

    ctx, dx, dy, dz, ds = _solve_newton(ctx, -rx, -ry, -rz, bs)
    alpha = _line_search(ctx, dz, ds)

    x = ctx.x + alpha * dx
    y = ctx.y + alpha * dy
    z = ctx.z + alpha * dz
    s = ctx.s + alpha * ds
    ctx = ctx.replace(x=x, y=y, z=z, s=s)

    ctx = _update_scaling(ctx, dz, ds, alpha)

    rx, ry, rz, rs = _residual(ctx)

    return (ctx, rx, ry, rz, rs, iter + 1)

  val = jax.lax.while_loop(cond_fun, body_fun, init_val)
  ctx = val[0]

  return ctx


@functools.partial(jax.custom_jvp, nondiff_argnames=('cones',))
def _solve(
    P: jax.Array,
    q: jax.Array,
    A: jax.Array,
    b: jax.Array,
    G: jax.Array,
    h: jax.Array,
    cones: tuple[NonNegativeCone | SecondOrderCone],
    comps: tuple[jax.Array],
    x_init: jax.Array | None = None,
    y_init: jax.Array | None = None,
    z_init: jax.Array | None = None,
    s_init: jax.Array | None = None,
    iterations: int = 100,
    tolerance: float = 1e-8,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """
  Solves the conic program
    minimize    ½ xᵀ P x + qᵀ x
    subject to  A x = b
                G x + s = h
                s ∈ 𝒦
  cones are either non-negative cone or second-order cone.
  Returns x, y, z, s
  where y is the dual of (A x = b), and z is the dual of (G x + s = h).
  """

  ctx = _solve_impl(
      P,
      q,
      A,
      b,
      G,
      h,
      cones,
      comps,
      x_init=x_init,
      y_init=y_init,
      z_init=z_init,
      s_init=s_init,
      iterations=iterations,
      tolerance=tolerance,
  )

  return ctx.x, ctx.y, ctx.z, ctx.s


@_solve.defjvp
def _solve_jvp(cones, primals, tangents):
  """
  Solve the primal as :func:`_solve` and computes tangents via:
    P dx + Aᵀ dy + Gᵀ dz = -(dP x + dAᵀ y + dGᵀ z + dq)
    A dx                 = db - dA x
    G dx            + ds = dh - dG x
    λ ∘ (W dz + W⁻ᵀ ds) = dμ e
  """

  (
      P,
      q,
      A,
      b,
      G,
      h,
      comps,
      x_init,
      y_init,
      z_init,
      s_init,
      iterations,
      tolerance,
  ) = primals

  ctx = _solve_impl(
      P,
      q,
      A,
      b,
      G,
      h,
      cones,
      comps,
      x_init=x_init,
      y_init=y_init,
      z_init=z_init,
      s_init=s_init,
      iterations=iterations,
      tolerance=tolerance,
  )
  x, y, z, s = ctx.x, ctx.y, ctx.z, ctx.s

  def f(P, q, A, b, G, h, comps):
    return _optimality_conditions(P, q, A, b, G, h, cones, comps, x, y, z, s)

  _, (lx, ly, lz, ls) = jax.jvp(f, primals[:7], tangents[:7])

  ctx, dx, dy, dz, ds = _solve_newton(ctx, -lx, -ly, -lz, -ls)

  return (x, y, z, s), (dx, dy, dz, ds)


@functools.cache
def _warn_float32():
  warnings.warn(
      'conic_solver.solve is running with float64 disabled. For better '
      'numerical stability, set jax.config.update("jax_enable_x64", True) '
      'before creating the problem arrays.',
      stacklevel=8,
  )


def solve(
    P: jax.Array,
    q: jax.Array,
    A: jax.Array,
    b: jax.Array,
    cones: tuple[ZeroCone | NonNegativeCone | SecondOrderCone],
    complementarities: jax.Array | None = None,
    *,
    x_init: jax.Array | None = None,
    z_init: jax.Array | None = None,
    s_init: jax.Array | None = None,
    iterations: int = 100,
    tolerance: float = 1e-8,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """
  Solves the conic program
    minimize    ½ xᵀ P x + qᵀ x
    subject to  A x + s = b
                s ∈ 𝒦
  cones are either zero cone, non-negative cone or second-order cone.
  Returns x, z, s
  where z is the dual of (A x + s = b).
  """

  if not jax.config.x64_enabled:
    _warn_float32()

  complementarities = (
      jp.array(complementarities)
      if (complementarities is not None)
      else jp.zeros(len(cones))
  )

  eq_idx = []
  nonneg_blocks = []
  other_blocks = []

  i = 0
  for cone, comp in zip(cones, complementarities):
    end = i + cone.dim
    idx = list(range(i, end))

    if isinstance(cone, ZeroCone):
      eq_idx.extend(idx)
    elif isinstance(cone, NonNegativeCone):
      nonneg_blocks.append((idx, cone, comp))
    else:
      other_blocks.append((idx, cone, comp))

    i = end

  cone_idx = []
  reordered_cones = []
  reordered_comps = []

  # add grouped nonnegative cones
  for idx, _, _ in nonneg_blocks:
    cone_idx.extend(idx)
  if nonneg_blocks:
    dim = sum([cone.dim for _, cone, _ in nonneg_blocks])
    comps = jp.concatenate(
        [jp.full(cone.dim, comp) for _, cone, comp in nonneg_blocks]
    )
    reordered_cones.append(NonNegativeCone(dim=dim))
    reordered_comps.append(comps)

  # add other cones
  for idx, cone, comp in other_blocks:
    cone_idx.extend(idx)
    reordered_cones.append(cone)
    reordered_comps.append(comp)

  eq_idx = jp.array(eq_idx, dtype=jp.int32)
  cone_idx = jp.array(cone_idx, dtype=jp.int32)

  Aeq = A[eq_idx]
  beq = b[eq_idx]
  zeq_init = z_init[eq_idx] if z_init is not None else None

  Acone = A[cone_idx]
  bcone = b[cone_idx]
  zcone_init = z_init[cone_idx] if z_init is not None else None
  scone_init = s_init[cone_idx] if s_init is not None else None

  x, zeq, zcone, scone = _solve(
      P=P,
      q=q,
      A=Aeq,
      b=beq,
      G=Acone,
      h=bcone,
      cones=tuple(reordered_cones),
      comps=tuple(reordered_comps),
      x_init=x_init,
      y_init=zeq_init,
      z_init=zcone_init,
      s_init=scone_init,
      iterations=iterations,
      tolerance=tolerance,
  )

  z = jp.zeros(A.shape[0], dtype=A.dtype)
  z = z.at[eq_idx].set(zeq)
  z = z.at[cone_idx].set(zcone)

  s = jp.zeros(A.shape[0], dtype=A.dtype)
  s = s.at[eq_idx].set(beq - Aeq @ x)
  s = s.at[cone_idx].set(scone)

  return x, z, s
