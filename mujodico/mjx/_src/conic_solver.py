from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jp


@dataclass(frozen=True)
class ZeroCone:
  dim: int

  def __post_init__(self):
    if self.dim <= 0:
      raise ValueError("ZeroCone requires dim >= 1")


@dataclass(frozen=True)
class NonNegativeCone:
  dim: int
  complementarity: float = 0.0

  def __post_init__(self):
    if self.dim <= 0:
      raise ValueError("NonNegativeCone requires dim >= 1")
    if self.complementarity < 0.0:
      raise ValueError("Cone complementarity must be non-negative")

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


@dataclass(frozen=True)
class SecondOrderCone:
  dim: int
  complementarity: float = 0.0

  def __post_init__(self):
    if self.dim < 2:
      raise ValueError("SecondOrderCone requires dim >= 2")
    if self.complementarity < 0.0:
      raise ValueError("Cone complementarity must be non-negative")

  @property
  def deg(self):
    return 1

  @property
  def jordan_identity(self):
    return jp.zeros(self.dim).at[0].set(1.0)

  def _jordan_product(self, s: jax.Array, z: jax.Array) -> jax.Array:
    assert s.size == self.dim
    assert z.size == self.dim
    raise NotImplementedError()

  def _inverse_jordan_product(self, u: jax.Array, v: jax.Array) -> jax.Array:
    assert u.size == self.dim
    assert v.size == self.dim
    raise NotImplementedError()

  def _line_search(self, s: jax.Array, ds: jax.Array) -> jax.Array:
    assert s.size == self.dim
    assert ds.size == self.dim
    raise NotImplementedError()


@jax.tree_util.register_pytree_node_class
@dataclass
class Context:
  P: jax.Array
  q: jax.Array
  A: jax.Array
  b: jax.Array
  G: jax.Array
  h: jax.Array
  cones: tuple[NonNegativeCone | SecondOrderCone]
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

  def tree_flatten(self):
    children = (
      self.P, self.q, self.A, self.b, self.G, self.h,
      self.x, self.y, self.z, self.s,
      self.W, self.Winv, self.lmbda,
      self.chol_Q, self.chol_S,
    )
    aux = (
      self.cones,
      self.dim,
      self.iterations,
      self.tolerance,
      self._EPS,
    )
    return children, aux

  @classmethod
  def tree_unflatten(cls, aux, children):
      cones, dim, iterations, tolerance, eps = aux
      P, q, A, b, G, h, x, y, z, s, W, Winv, lmbda, chol_Q, chol_S = children

      return cls(
          P, q, A, b, G, h, cones,
          x, y, z, s,
          W, Winv, lmbda,
          dim, iterations, tolerance,
          chol_Q, chol_S, eps,
      )


def _initialize(**kwargs) -> Context:
  dim = sum([cone.dim for cone in kwargs['cones']])
  W = jp.eye(dim)
  Winv= jp.eye(dim)

  ctx = Context(
    P=kwargs['P'],
    q=kwargs['q'],
    A=kwargs['A'],
    b=kwargs['b'],
    G=kwargs['G'],
    h=kwargs['h'],
    cones=kwargs['cones'],
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
    x, y, z = _solve_kkt(ctx=ctx, bx=-ctx.q, by=ctx.b, bz=ctx.h)
    ctx.x = ctx.x or x
    ctx.y = ctx.y or y
    ctx.z = ctx.z or z

  if ctx.s is None:
    ctx.s = ctx.h - ctx.G @ ctx.x

  if dim:
    # TODO(wei-chen): Implement for second-order cone
    ctx.z += jp.maximum(0, -jp.min(ctx.z) + 1)
    ctx.s += jp.maximum(0, -jp.min(ctx.s) + 1)

  _compute_scaling(ctx)

  return ctx


def _compute_scaling(ctx: Context):
  i = 0
  for cone in ctx.cones:
    end = i + cone.dim

    sk = ctx.s[i:end]
    zk = ctx.z[i:end]

    if isinstance(cone, NonNegativeCone):
      w = jp.sqrt(sk / zk)
      ctx.W = ctx.W.at[i:end, i:end].set(jp.diag(w))
      ctx.Winv = ctx.Winv.at[i:end, i:end].set(jp.diag(1.0 / w))
      lmbda = jp.sqrt(sk * zk)
      ctx.lmbda = ctx.lmbda.at[i:end].set(lmbda)

    elif isinstance(cone, SecondOrderCone):
      raise NotImplementedError()

    i = end

  ctx.chol_Q = None
  ctx.chol_S = None


def _update_scaling(ctx: Context, dz: jax.Array, ds: jax.Array, alpha: jax.Array):
  i = 0
  for cone in ctx.cones:
    end = i + cone.dim

    W = ctx.W[i:end, i:end]
    Winv = ctx.Winv[i:end, i:end]

    if isinstance(cone, NonNegativeCone):
      s_scaled = ctx.lmbda[i:end] + alpha * (Winv.T @ ds[i:end])
      z_scaled = ctx.lmbda[i:end] + alpha * (W @ dz[i:end])
      s_scaled_sqrt = jp.sqrt(jp.maximum(s_scaled, ctx._EPS))
      z_scaled_sqrt = jp.sqrt(jp.maximum(z_scaled, ctx._EPS))
      w = s_scaled_sqrt / z_scaled_sqrt * jp.diag(W)
      ctx.W = ctx.W.at[i:end, i:end].set(jp.diag(w))
      ctx.Winv = ctx.Winv.at[i:end, i:end].set(jp.diag(1.0 / w))
      lmbda = s_scaled_sqrt * z_scaled_sqrt
      ctx.lmbda = ctx.lmbda.at[i:end].set(lmbda)

    elif isinstance(cone, SecondOrderCone):
      raise NotImplementedError()

    i = end

  ctx.chol_Q = None
  ctx.chol_S = None


def _mu_e(ctx: Context) -> jax.Array:
  result = jp.zeros(ctx.dim)

  i = 0
  for cone in ctx.cones:
    end = i + cone.dim
    result = result.at[i:end].set(cone.complementarity * cone.jordan_identity)
    i = end

  return result


def _jordan_product(ctx: Context, s: jax.Array, z: jax.Array) -> jax.Array:
  result = jp.zeros(ctx.dim)

  i = 0
  for cone in ctx.cones:
    end = i + cone.dim
    result = result.at[i:end].set(cone._jordan_product(s[i:end], z[i:end]))
    i = end

  return result


def _inverse_jordan_product(ctx: Context, u: jax.Array, v: jax.Array) -> jax.Array:
  result = jp.zeros(ctx.dim)

  i = 0
  for cone in ctx.cones:
    end = i + cone.dim
    result = result.at[i:end].set(cone._inverse_jordan_product(u[i:end], v[i:end]))
    i = end

  return result


def _line_search(ctx: Context, dz: jax.Array, ds: jax.Array) -> jax.Array:
  alpha = jp.ones(1)

  i = 0
  for cone in ctx.cones:
    end = i + cone.dim
    alpha = jp.minimum(alpha, cone._line_search(ctx.z[i:end], dz[i:end]))
    alpha = jp.minimum(alpha, cone._line_search(ctx.s[i:end], ds[i:end]))
    i = end

  return alpha


def _residual(ctx: Context) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  rx = ctx.P @ ctx.x + ctx.A.T @ ctx.y + ctx.G.T @ ctx.z + ctx.q
  ry = ctx.A @ ctx.x - ctx.b
  rz = ctx.s + ctx.G @ ctx.x - ctx.h
  rs = _jordan_product(ctx, ctx.lmbda, ctx.lmbda) - _mu_e(ctx)

  return rx, ry, rz, rs


def _solve_kkt(
    ctx: Context, bx: jax.Array, by: jax.Array, bz: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
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

  if ctx.chol_Q is None:
    Q = P + G.T @ WTW_inv @ G
    ctx.chol_Q = _robust_cho_factor(Q)

  def Q_solve(rhs):
      return _robust_cho_solve(ctx.chol_Q, rhs)

  rhs_x = bx + G.T @ WTW_inv @ bz

  if ctx.chol_S is None:
    S = A @ Q_solve(A.T)
    ctx.chol_S = _robust_cho_factor(S)

  rhs_y = A @ Q_solve(rhs_x) - by
  y = _robust_cho_solve(ctx.chol_S, rhs_y)

  x = Q_solve(rhs_x - A.T @ y)

  z = WTW_inv @ (G @ x - bz)

  return x, y, z


def _solve_newton(
    ctx: Context, bx: jax.Array, by: jax.Array, bz: jax.Array, bs: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """
  Solves the newton step:
    P x + Aᵀ y + Gᵀ z = bx
    A x               = by
    G x           + s = bz
    λ ∘ (W z + W⁻ᵀ s) = bs
  """

  WT_lmbdainv_bs = ctx.W.T @ _inverse_jordan_product(ctx, ctx.lmbda, bs)
  x, y, z = _solve_kkt(ctx, bx, by, bz - WT_lmbdainv_bs)
  s = WT_lmbdainv_bs - ctx.W.T @ ctx.W @ z
  return x, y, z, s


def _corrector_mu_e(
    ctx: Context, ds: jax.Array, dz: jax.Array, alpha: jax.Array
) -> jax.Array:
  result = jp.zeros(ctx.dim)

  i = 0
  for cone in ctx.cones:
    end = i + cone.dim

    mu0 = jp.dot(ctx.s[i:end], ctx.z[i:end]) / cone.deg
    mu1 = jp.dot(ctx.s[i:end] + alpha * ds[i:end], ctx.z[i:end] + alpha * dz[i:end]) / cone.deg
    mu_target = cone.complementarity

    rho = jp.abs(mu1 - mu_target) / jp.maximum(jp.abs(mu0 - mu_target), ctx._EPS)
    sigma = jp.clip(rho, 0.0, 1.0)**3

    result = result.at[i:end].set(sigma * (mu1 - mu_target) * cone.jordan_identity)

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
  *,
  x_init: jax.Array | None = None,
  y_init: jax.Array | None = None,
  z_init: jax.Array | None = None,
  s_init: jax.Array | None = None,
  iterations: int = 100,
  tolerance: float = 1e-8,
) -> Context:

  ctx = _initialize(
    P=P, q=q, A=A, b=b, G=G, h=h, cones=cones,
    x=x_init, y=y_init, z=z_init, s=s_init,
    iterations=iterations, tolerance=tolerance,
  )

  rx, ry, rz, rs = _residual(ctx)
  init_val = (ctx, rx, ry, rz, rs, 0)

  def cond_fun(val):
    ctx, rx, ry, rz, rs, iter = val
    return (
      (iter < ctx.iterations) &
      ((jp.linalg.norm(rx, jp.inf) > ctx.tolerance) |
       (jp.linalg.norm(ry, jp.inf) > ctx.tolerance) |
       (jp.linalg.norm(rz, jp.inf) > ctx.tolerance) |
       (jp.linalg.norm(rs, jp.inf) > ctx.tolerance)))

  def body_fun(val):
    ctx, rx, ry, rz, rs, iter = val

    dx, dy, dz, ds = _solve_newton(ctx, -rx, -ry, -rz, -rs)
    alpha = _line_search(ctx, dz, ds)

    bs = (-rs
      - _jordan_product(ctx, ctx.Winv.T @ ds, ctx.W @ dz)
      + _corrector_mu_e(ctx, ds, dz, alpha)
    )

    dx, dy, dz, ds = _solve_newton(ctx, -rx, -ry, -rz, bs)
    alpha = _line_search(ctx, dz, ds) * 0.99

    ctx.x = ctx.x + alpha * dx
    ctx.y = ctx.y + alpha * dy
    ctx.z = ctx.z + alpha * dz
    ctx.s = ctx.s + alpha * ds
    _update_scaling(ctx, dz, ds, alpha)

    rx, ry, rz, rs = _residual(ctx)

    return (ctx, rx, ry, rz, rs, iter + 1)

  val = jax.lax.while_loop(cond_fun, body_fun, init_val)
  ctx = val[0]

  return ctx


@partial(jax.custom_vjp, nondiff_argnames=('cones',))
def _solve(
  P: jax.Array,
  q: jax.Array,
  A: jax.Array,
  b: jax.Array,
  G: jax.Array,
  h: jax.Array,
  cones: tuple[NonNegativeCone | SecondOrderCone],
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

  ctx = _solve_impl(P, q, A, b, G, h, cones,
                    x_init=x_init, y_init=y_init, z_init=z_init, s_init=s_init,
                    iterations=iterations, tolerance=tolerance)

  return ctx.x, ctx.y, ctx.z, ctx.s


def _solve_fwd(
  P: jax.Array,
  q: jax.Array,
  A: jax.Array,
  b: jax.Array,
  G: jax.Array,
  h: jax.Array,
  cones: tuple[NonNegativeCone | SecondOrderCone],
  x_init: jax.Array | None = None,
  y_init: jax.Array | None = None,
  z_init: jax.Array | None = None,
  s_init: jax.Array | None = None,
  iterations: int = 100,
  tolerance: float = 1e-8,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], Context]:

  ctx = _solve_impl(P, q, A, b, G, h, cones,
                    x_init=x_init, y_init=y_init, z_init=z_init, s_init=s_init,
                    iterations=iterations, tolerance=tolerance)

  return (ctx.x, ctx.y, ctx.z, ctx.s), ctx


def _solve_bwd(
  cones,
  ctx: Context,
  g: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
           None, None, None, None, None, None]:

  gx, gy, gz, gs = g

  W = ctx.W
  lx, ly, lz = _solve_kkt(ctx, gx, gy, gz - W.T @ W @ gs)
  ls = _inverse_jordan_product(ctx, ctx.lmbda, W @ (gs - lz))  # never used

  x, y, z, s = ctx.x, ctx.y, ctx.z, ctx.s

  def f(P, q, A, b, G, h):
    return (
      P @ x + A.T @ y + G.T @ z + q,
      A @ x                     - b,
      G @ x                 + s - h,
      # (s ∘ z − μ e) is not a function of parameters and thus ignored
    )

  _, f_vjp = jax.vjp(f, ctx.P, ctx.q, ctx.A, ctx.b, ctx.G, ctx.h)

  return f_vjp((-lx, -ly, -lz)) + (None, None, None, None, None, None)


_solve.defvjp(_solve_fwd, _solve_bwd)


def solve(
  P: jax.Array,
  q: jax.Array,
  A: jax.Array,
  b: jax.Array,
  cones: tuple[ZeroCone | NonNegativeCone | SecondOrderCone],
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

  eq_idx = []
  nonneg_blocks = {}
  other_blocks = []

  i = 0
  for cone in cones:
    end = i + cone.dim
    idx = list(range(i, end))

    if isinstance(cone, ZeroCone):
      eq_idx.extend(idx)

    elif isinstance(cone, NonNegativeCone):
      nonneg_blocks.setdefault(cone.complementarity, []).extend(idx)

    else:
      other_blocks.append((idx, cone))

    i = end

  cone_idx = []
  nonzero_cones = []

  # add grouped nonnegative cones
  for comp, idx in nonneg_blocks.items():
    cone_idx.extend(idx)
    nonzero_cones.append(NonNegativeCone(dim=len(idx), complementarity=comp))

  # add other cones
  for idx, cone in other_blocks:
    cone_idx.extend(idx)
    nonzero_cones.append(cone)

  nonzero_cones = tuple(nonzero_cones)

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
    cones=nonzero_cones,
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
