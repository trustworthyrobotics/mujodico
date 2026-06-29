import dataclasses
import functools

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
    assert s.size == self.dim
    assert z.size == self.dim
    return s * z

  def _inverse_jordan_product(self, u: jax.Array, v: jax.Array) -> jax.Array:
    assert u.size == self.dim
    assert v.size == self.dim
    return v / u

  def _line_search(self, s: jax.Array, ds: jax.Array) -> jax.Array:
    assert s.size == self.dim
    assert ds.size == self.dim
    return 1.0 / jp.maximum(jp.max(-ds / s), 0.0)


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

  def _jordan_product(self, s: jax.Array, z: jax.Array) -> jax.Array:
    assert s.size == self.dim
    assert z.size == self.dim
    return jp.concatenate((
      jp.sum(s * z, keepdims=True),
      s[0] * z[1:] + z[0] * s[1:]
    ))

  def _inverse_jordan_product(self, u: jax.Array, v: jax.Array) -> jax.Array:
    assert u.size == self.dim
    assert v.size == self.dim

    u0, u1 = u[0], u[1:]
    v0, v1 = v[0], v[1:]

    det = u0 * u0 - jp.dot(u1, u1)

    r0 = u0 * v0 - jp.dot(u1, v1)
    r1 = -u1 * v0 + ((det * v1) + (u1 * jp.dot(u1, v1))) / u0

    return jp.concatenate((jp.array([r0]), r1)) / det

  def _line_search(self, s: jax.Array, ds: jax.Array) -> jax.Array:
    assert s.size == self.dim
    assert ds.size == self.dim
    raise NotImplementedError()


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
        'chol_Q',
        'chol_S',
    ],
    meta_fields=['cones', 'dim', 'iterations', 'tolerance', '_EPS'],
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
  chol_Q: tuple[jax.Array, jax.Array] | None = None
  chol_S: tuple[jax.Array, jax.Array] | None = None
  _EPS: float = 1e-8  # numerical noise floor

  _ALLOWED_FIELDS = frozenset({
      'x',
      'y',
      'z',
      's',
      'W',
      'Winv',
      'lmbda',
      'chol_Q',
      'chol_S',
  })

  def replace(self, **kwargs):
    invalid = kwargs.keys() - self._ALLOWED_FIELDS
    if invalid:
      raise ValueError(f'Attempt replacing immutable fields: {invalid}')
    return dataclasses.replace(self, **kwargs)


def _initialize(**kwargs) -> Context:
  """Initializes the context from problem parameters."""

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

  if dim:
    # TODO(wei-chen): Implement for second-order cone
    z = ctx.z + jp.maximum(0, -jp.min(ctx.z) + 1)
    s = ctx.s + jp.maximum(0, -jp.min(ctx.s) + 1)
    ctx = ctx.replace(z=z, s=s)

  ctx = _compute_scaling(ctx)

  return ctx


def _compute_scaling(ctx: Context) -> Context:
  """Compute W so that W z = W⁻ᵀ s = ."""

  W, Winv, lmbda = ctx.W, ctx.Winv, ctx.lmbda

  i = 0
  for cone in ctx.cones:
    end = i + cone.dim

    sk = ctx.s[i:end]
    zk = ctx.z[i:end]

    if isinstance(cone, NonNegativeCone):
      wk = jp.sqrt(sk / zk)
      lmbdak = jp.sqrt(sk * zk)

      W = W.at[i:end, i:end].set(jp.diag(wk))
      Winv = Winv.at[i:end, i:end].set(jp.diag(1.0 / wk))
      lmbda = lmbda.at[i:end].set(lmbdak)

    elif isinstance(cone, SecondOrderCone):
      raise NotImplementedError()

    i = end

  ctx = ctx.replace(W=W, Winv=Winv, lmbda=lmbda, chol_Q=None, chol_S=None)

  return ctx


def _update_scaling(
    ctx: Context, dz: jax.Array, ds: jax.Array, alpha: jax.Array
) -> Context:
  """Update W and λ so that W z = W⁻ᵀ s = λ."""

  W, Winv, lmbda = ctx.W, ctx.Winv, ctx.lmbda

  i = 0
  for cone in ctx.cones:
    end = i + cone.dim

    Wk = W[i:end, i:end]
    Wkinv = Winv[i:end, i:end]

    if isinstance(cone, NonNegativeCone):
      sk_scaled = ctx.lmbda[i:end] + alpha * (Wkinv.T @ ds[i:end])
      zk_scaled = ctx.lmbda[i:end] + alpha * (Wk @ dz[i:end])
      sk_scaled_sqrt = jp.sqrt(jp.maximum(sk_scaled, ctx._EPS))
      zk_scaled_sqrt = jp.sqrt(jp.maximum(zk_scaled, ctx._EPS))
      wk = sk_scaled_sqrt / zk_scaled_sqrt * jp.diag(Wk)
      lmbdak = sk_scaled_sqrt * zk_scaled_sqrt

      W = W.at[i:end, i:end].set(jp.diag(wk))
      Winv = Winv.at[i:end, i:end].set(jp.diag(1.0 / wk))
      lmbda = lmbda.at[i:end].set(lmbdak)

    elif isinstance(cone, SecondOrderCone):
      raise NotImplementedError()

    i = end

  ctx = ctx.replace(W=W, Winv=Winv, lmbda=lmbda, chol_Q=None, chol_S=None)

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
  """Computes α so that z + α Δz ∈ 𝒦 and s + α Δs ∈ 𝒦."""
  alpha = jp.ones(1)

  i = 0
  for cone in ctx.cones:
    end = i + cone.dim
    alpha = jp.minimum(alpha, cone._line_search(ctx.z[i:end], dz[i:end]))
    alpha = jp.minimum(alpha, cone._line_search(ctx.s[i:end], ds[i:end]))
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
  return _optimality_conditions(
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


def _solve_kkt(
    ctx: Context, bx: jax.Array, by: jax.Array, bz: jax.Array
) -> tuple[Context, jax.Array, jax.Array, jax.Array]:
  """
  Solves the KKT system:
    P x + Aᵀ y + Gᵀ z = bx
    A x               = by
    G x        -WᵀW x = bz
  """

  def _robust_cho_factor(Q, reg=ctx._EPS * 100):
    d = 1.0 / jp.sqrt(jp.maximum(jp.diag(Q), ctx._EPS))
    Q = d[:, None] * Q * d[None, :]
    delta = reg * jp.max(jp.diag(Q)) if Q.size else 0.0
    Q = Q + delta * jp.eye(Q.shape[0], dtype=Q.dtype)
    chol, _ = jax.scipy.linalg.cho_factor(Q)
    return chol, d

  def _robust_cho_solve(chol_d, r):
    chol, d = chol_d
    d = d.reshape((-1,) + (1,) * (r.ndim - d.ndim))
    r = d * r
    y = jax.scipy.linalg.cho_solve((chol, False), r)
    x = d * y
    return x

  P = ctx.P
  A = ctx.A
  G = ctx.G
  Winv = ctx.Winv

  WTW_inv = Winv @ Winv.T

  chol_Q = ctx.chol_Q
  if chol_Q is None:
    Q = P + G.T @ WTW_inv @ G
    chol_Q = _robust_cho_factor(Q)

  def Q_solve(rhs):
    return _robust_cho_solve(chol_Q, rhs)

  rhs_x = bx + G.T @ WTW_inv @ bz

  chol_S = ctx.chol_S
  if chol_S is None:
    S = A @ Q_solve(A.T)
    chol_S = _robust_cho_factor(S)

  rhs_y = A @ Q_solve(rhs_x) - by
  y = _robust_cho_solve(chol_S, rhs_y)

  x = Q_solve(rhs_x - A.T @ y)

  z = WTW_inv @ (G @ x - bz)

  if ctx.chol_Q is None or ctx.chol_S is None:
    ctx = ctx.replace(chol_Q=chol_Q, chol_S=chol_S)

  return ctx, x, y, z


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
  s = WT_lmbdainv_bs - ctx.W.T @ ctx.W @ z
  return ctx, x, y, z, s


def _corrector_mu_e(
    ctx: Context, ds: jax.Array, dz: jax.Array, alpha: jax.Array
) -> jax.Array:
  result = jp.zeros(ctx.dim)

  i = 0
  for cone, comp in zip(ctx.cones, ctx.comps):
    end = i + cone.dim

    if isinstance(cone, NonNegativeCone):
      mu0 = ctx.s[i:end] * ctx.z[i:end]
      mu1 = (ctx.s[i:end] + alpha * ds[i:end]) * (
          ctx.z[i:end] + alpha * dz[i:end]
      )
    else:
      mu0 = jp.dot(ctx.s[i:end], ctx.z[i:end]) / cone.deg
      mu1 = (
          jp.dot(
              ctx.s[i:end] + alpha * ds[i:end], ctx.z[i:end] + alpha * dz[i:end]
          )
          / cone.deg
      )
    mu_target = comp

    rho = jp.abs(mu1 - mu_target) / jp.maximum(
        jp.abs(mu0 - mu_target), ctx._EPS
    )
    sigma = jp.clip(rho, 0.0, 1.0) ** 3

    result = result.at[i:end].set(
        sigma * (mu1 - mu_target) * cone.jordan_identity
    )

    i = end

  return result


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
  """See :func:`_solve`"""

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
        -rs
        - _jordan_product(ctx.cones, ctx.Winv.T @ ds, ctx.W @ dz)
        + _corrector_mu_e(ctx, ds, dz, alpha)
    )

    ctx, dx, dy, dz, ds = _solve_newton(ctx, -rx, -ry, -rz, bs)
    alpha = _line_search(ctx, dz, ds) * 0.99

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
    A dx                 = -db
    G dx            + ds = -dh
     λ ∘ (W dz + W⁻ᵀ ds) = 0
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
