"""Elastic ℓ₁ restoration Newton equations and structured elimination.

For c + p − n = 0, elastic stationarity is ρ + y − zᵖ = 0 and
ρ − y − zⁿ = 0. Eliminating elastic bound duals and p/n produces

  Dₑ = (δw + zᵖ/p)⁻¹ + (δw + zⁿ/n)⁻¹,
  c̄ = c − (δw + zᵖ/p)⁻¹ f̄p + (δw + zⁿ/n)⁻¹ f̄n.

For original slacks, W = Hss + zˡ/distˡ + zᵘ/distᵘ. Eliminate
G Δv − Δs − Dₑ Δy = −c̄ and W Δs − Δy = −b_s using
W_eff = W/(1 + W Dₑ). Stage equalities retain the diagonal dual shift;
dynamics and the initial condition remain hard. The resulting problem is
an equality-constrained LQR with the original state/control dimensions.
RHS-only changes reuse factors; matrix changes require clearing _cache.
"""

from dataclasses import dataclass, replace
from collections.abc import Callable
from typing import TypeVar
from typing import NamedTuple

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import equality_constrained_lqr as lqr
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas.ip_iterate import Iterate, IterateData, all_finite, max_abs, distances
from mujodico.trajopt._src.ippas.options import Options
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel
from mujodico.trajopt._src.types import Params

Bounds = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


class ElasticBlocks(NamedTuple):
  e: jax.Array
  g: jax.Array
  gf: jax.Array


def blocks(base: pd.IpPrimalDualSystem) -> ElasticBlocks:
  return ElasticBlocks(base.r_e, base.r_g, base.r_gf)


def multipliers(it: Iterate) -> ElasticBlocks:
  return ElasticBlocks(it.beta, it.y, it.yf)


Tree = TypeVar('Tree')


def tree(fn: Callable[..., jax.Array], *args: Tree) -> Tree:
  """Map an operation over corresponding elastic blocks, preserving structure."""
  return jax.tree.map(fn, *args)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RestorationIterate:
  base: Iterate
  p: ElasticBlocks
  n: ElasticBlocks
  zp: ElasticBlocks
  zn: ElasticBlocks


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Cache:
  reduced: lqr.EqualityConstrainedLQR
  initial_lu: tuple[jax.Array, jax.Array]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RestorationSystem:
  base: pd.IpPrimalDualSystem
  p: ElasticBlocks
  n: ElasticBlocks
  zp: ElasticBlocks
  zn: ElasticBlocks
  r_p: ElasticBlocks
  r_n: ElasticBlocks
  r_cp: ElasticBlocks
  r_cn: ElasticBlocks
  Hs: jax.Array
  Hsf: jax.Array
  delta_w: jax.Array
  dp: ElasticBlocks | None = None
  dn: ElasticBlocks | None = None
  dzp: ElasticBlocks | None = None
  dzn: ElasticBlocks | None = None
  _cache: _Cache | None = None

  def replace_rhs(self, **rhs) -> 'RestorationSystem':
    """Retain factors for SOC or full-system iterative refinement."""
    extra = {
        name: rhs.pop(name, getattr(self, name))
        for name in ('r_p', 'r_n', 'r_cp', 'r_cn')
    }
    return replace(
        self,
        base=self.base.replace_rhs(**rhs),
        **extra,
        dp=None,
        dn=None,
        dzp=None,
        dzn=None,
    )


def proximity(model: StagewiseOcpModel, it: Iterate, ref: Iterate) -> jax.Array:
  """Scaled squared proximity in Euclidean/reference quaternion coordinates."""
  dx = jax.vmap(model._state_difference)(it.x, ref.x)
  weights = []
  previous = 0
  for addr in model.quaternion_addrs:
    weights.extend((
        1 / jp.maximum(1, jp.abs(ref.x[:, previous:addr])),
        jp.ones((ref.x.shape[0], 3), ref.x.dtype),
    ))
    previous = addr + 4
  weights.append(1 / jp.maximum(1, jp.abs(ref.x[:, previous:])))
  result = jp.sum((jp.concatenate(weights, axis=-1) * dx) ** 2)
  for value, reference in ((it.u, ref.u), (it.s, ref.s), (it.sf, ref.sf)):
    result += jp.sum(
        ((value - reference) / jp.maximum(1, jp.abs(reference))) ** 2
    )
  return result / 2


def evaluate_trial(
    model: StagewiseOcpModel,
    it: RestorationIterate,
    params: Params,
    ref: Iterate,
    zeta: jax.Array,
    options: Options,
) -> dict[str, jax.Array]:
  """Restoration values only; the original objective is never evaluated."""
  b = it.base
  return {
      'objective': (
          zeta * proximity(model, b, ref)
          + options.resto_penalty_parameter
          * sum(jp.sum(v) for v in (*it.p, *it.n))
      ),
      'r0': model.initial_residual(b.x[0], params),
      'r_f': model.dynamics_residual(b.x[:-1], b.u, b.x[1:], params),
      'r_e': (
          model.stage_equality_residual(b.x[:-1], b.u, params, xs_next=b.x[1:]) + it.p.e - it.n.e
      ),
      'r_g': (
          model.stage_inequality_residual(b.x[:-1], b.u, params)
          - b.s
          + it.p.g
          - it.n.g
      ),
      'r_gf': (
          model.terminal_inequality_residual(b.x[-1], params)
          - b.sf
          + it.p.gf
          - it.n.gf
      ),
  }


def evaluate(
    model: StagewiseOcpModel,
    it: RestorationIterate,
    params: Params,
    mu: jax.Array,
    bounds: Bounds,
    options: Options,
    ref: Iterate,
    zeta: jax.Array,
) -> tuple[RestorationSystem, IterateData]:
  """Evaluate restoration curvature, stationarity and elastic feasibility."""
  b = it.base

  # Proximity is separable in stages: differentiate each state/control block,
  # avoiding a horizon-sized Hessian.
  def state_cost(x, xref):
    def value(dx):
      point = model.retract_state(x, dx, 1.0, params)
      delta = model._state_difference(point, xref)
      weights, previous = [], 0
      for addr in model.quaternion_addrs:
        weights.extend((
            1 / jp.maximum(1, jp.abs(xref[previous:addr])),
            jp.ones(3, x.dtype),
        ))
        previous = addr + 4
      weights.append(1 / jp.maximum(1, jp.abs(xref[previous:])))
      return 0.5 * zeta * jp.sum((jp.concatenate(weights) * delta) ** 2)

    zero = jp.zeros(model.nx_tangent, x.dtype)
    return jax.grad(value)(zero), jax.hessian(value)(zero)

  gx, Q = jax.vmap(state_cost)(b.x, ref.x)
  wu = zeta / jp.maximum(1, jp.abs(ref.u)) ** 2
  gu = wu * (b.u - ref.u)
  hxx, hxu, huu = model.lagrangian_hessian(
      b.x,
      b.u,
      b.lmbda[0],
      b.lmbda[1:],
      b.beta,
      b.y,
      jp.zeros(0, b.x.dtype),
      b.yf,
      0.0,
      params,
  )
  value = evaluate_trial(model, it, params, ref, zeta, options)
  base, data = pd.assemble_system(
      model,
      b,
      params,
      mu,
      bounds,
      options,
      value['objective'],
      (gx, gu),
      (hxx + Q, hxu, huu + jax.vmap(jp.diag)(wu)),
  )
  Hs = zeta / jp.maximum(1, jp.abs(ref.s)) ** 2
  Hsf = zeta / jp.maximum(1, jp.abs(ref.sf)) ** 2
  residuals = {name: value[name] for name in ('r_e', 'r_g', 'r_gf')}
  base = replace(
      base,
      **residuals,
      r_s=base.r_s + Hs * (b.s - ref.s),
      r_sf=base.r_sf + Hsf * (b.sf - ref.sf),
  )
  data = replace(data, **residuals)
  lam = multipliers(b)
  rho = options.resto_penalty_parameter
  return (
      RestorationSystem(
          base,
          it.p,
          it.n,
          it.zp,
          it.zn,
          tree(lambda y, z: rho + options.kappa_d * mu + y - z, lam, it.zp),
          tree(lambda y, z: rho + options.kappa_d * mu - y - z, lam, it.zn),
          tree(lambda p, z: p * z - mu, it.p, it.zp),
          tree(lambda n, z: n * z - mu, it.n, it.zn),
          Hs,
          Hsf,
          jp.asarray(0.0, b.x.dtype),
      ),
      data,
  )


def solve(system: RestorationSystem) -> RestorationSystem:
  """Solve the full elastic Newton system using diagonal Schur complements."""
  a = system
  b = a.base
  ap = tree(lambda p, z: p / (z + a.delta_w * p), a.p, a.zp)
  an = tree(lambda n, z: n / (z + a.delta_w * n), a.n, a.zn)
  fp = tree(lambda r, c, p: r + c / p, a.r_p, a.r_cp, a.p)
  fn = tree(lambda r, c, n: r + c / n, a.r_n, a.r_cn, a.n)
  de = tree(lambda p, n: p + n, ap, an)
  c = tree(lambda c, p, n, f, h: c - p * f + n * h, blocks(b), ap, an, fp, fn)

  def slack(s, zl, zu, lo, hi, h, rs, cl, cu, shift, residual):
    dl, du = distances(s, lo, hi)
    il = jp.where(jp.isfinite(lo), 1 / dl, 0.0)
    iu = jp.where(jp.isfinite(hi), 1 / du, 0.0)
    w = h + il * zl + iu * zu
    rhs = rs + il * cl - iu * cu
    fac = 1 / (1 + w * shift)
    return w, rhs, fac, w * fac, fac * rhs + w * fac * residual, il, iu

  stage = slack(
      b.s, b.z_l, b.z_u, b.g_l, b.g_u, a.Hs, b.r_s, b.r_c_l, b.r_c_u, de.g, c.g
  )
  terminal = slack(
      b.sf,
      b.zf_l,
      b.zf_u,
      b.gf_l,
      b.gf_u,
      a.Hsf,
      b.r_sf,
      b.r_cf_l,
      b.r_cf_u,
      de.gf,
      c.gf,
  )
  w, bs, fac, weight, xi, il, iu = stage
  wf, bsf, facf, weightf, xif, ilf, iuf = terminal
  initial_lu = (
      jax.scipy.linalg.lu_factor(b.E0)
      if a._cache is None
      else a._cache.initial_lu
  )
  rhs = dict(
      q=b.r_x
      + jp.concatenate(
          (pd._matvec(b.Gx.swapaxes(-1, -2), xi), (b.Gf.T @ xif)[None])
      ),
      r=b.r_u + pd._matvec(b.Gu.swapaxes(-1, -2), xi),
      c=b.r_f,
      e=c.e,
      x0=-jax.scipy.linalg.lu_solve(initial_lu, b.r0),
  )
  if a._cache is None:
    gx, gu = b.Gx.swapaxes(-1, -2), b.Gu.swapaxes(-1, -2)
    reduced = lqr.EqualityConstrainedLQR.create(
        Q=b.Hxx
        + jp.concatenate((
            gx @ (weight[..., None] * b.Gx),
            (b.Gf.T @ (weightf[:, None] * b.Gf))[None],
        )),
        N=b.Hxu + gx @ (weight[..., None] * b.Gu),
        R=b.Huu + gu @ (weight[..., None] * b.Gu),
        A=b.A,
        B=b.B,
        Ex=b.Ex,
        Eu=b.Eu,
        delta_c=de.e + b.delta_c,
        **rhs,
    )
  else:
    reduced = a._cache.reduced.replace_rhs(**rhs)
  reduced = lqr.solve(reduced)
  dx, du = reduced.x, reduced.u
  ds = fac * (
      pd._matvec(b.Gx, dx[:-1]) + pd._matvec(b.Gu, du) + c.g - de.g * bs
  )
  dsf = facf * (b.Gf @ dx[-1] + c.gf - de.gf * bsf)
  dy, dyf = w * ds + bs, wf * dsf + bsf
  dlambda = reduced.lmbda.at[0].set(
      -jax.scipy.linalg.lu_solve(initial_lu, reduced.lmbda[0], trans=1)
  )
  base = replace(
      b,
      dx=dx,
      du=du,
      dlmbda=dlambda,
      dbeta=reduced.beta,
      ds=ds,
      dsf=dsf,
      dy=dy,
      dyf=dyf,
      dz_l=-il * (b.z_l * ds + b.r_c_l),
      dz_u=iu * (b.z_u * ds - b.r_c_u),
      dzf_l=-ilf * (b.zf_l * dsf + b.r_cf_l),
      dzf_u=iuf * (b.zf_u * dsf - b.r_cf_u),
  )
  dl = ElasticBlocks(reduced.beta, dy, dyf)
  dp = tree(lambda a, f, y: -a * (f + y), ap, fp, dl)
  dn = tree(lambda a, f, y: -a * (f - y), an, fn, dl)
  return replace(
      a,
      base=base,
      dp=dp,
      dn=dn,
      dzp=tree(lambda c, z, d, p: -(c + z * d) / p, a.r_cp, a.zp, dp, a.p),
      dzn=tree(lambda c, z, d, n: -(c + z * d) / n, a.r_cn, a.zn, dn, a.n),
      _cache=_Cache(reduced, initial_lu),
  )


def system_defect(a: RestorationSystem) -> dict[str, jax.Array | ElasticBlocks]:
  """Full regularized KKT defects, before any elastic/slack elimination."""
  d = pd.system_defect(a.base)
  d['r_s'] += a.Hs * a.base.ds
  d['r_sf'] += a.Hsf * a.base.dsf
  for name, p, n in zip(('r_e', 'r_g', 'r_gf'), a.dp, a.dn):
    d[name] += p - n
  dl = ElasticBlocks(a.base.dbeta, a.base.dy, a.base.dyf)
  d['r_p'] = tree(
      lambda r, p, y, z: r + a.delta_w * p + y - z, a.r_p, a.dp, dl, a.dzp
  )
  d['r_n'] = tree(
      lambda r, n, y, z: r + a.delta_w * n - y - z, a.r_n, a.dn, dl, a.dzn
  )
  d['r_cp'] = tree(
      lambda r, z, p, v, dz: r + z * p + v * dz, a.r_cp, a.zp, a.dp, a.p, a.dzp
  )
  d['r_cn'] = tree(
      lambda r, z, n, v, dz: r + z * n + v * dz, a.r_cn, a.zn, a.dn, a.n, a.dzn
  )
  return d


def iterative_refinement_solve(
    system: RestorationSystem, linear_ref_tol: float, max_refinement: int
) -> tuple[RestorationSystem, jax.Array]:
  """Refine the complete elastic equations using cached reduced factors."""
  solved = (
      solve(system) if system._cache is None or system.dp is None else system
  )
  names = system_defect(solved)
  rhs = {
      name: (
          getattr(system, name)
          if name.startswith(('r_p', 'r_n', 'r_cp', 'r_cn'))
          else getattr(system.base, name)
      )
      for name in names
  }
  threshold = linear_ref_tol * (1 + max_abs(*jax.tree.leaves(rhs)))

  def cond(carry):
    _, defect, count = carry
    return (count < max_refinement) & (
        max_abs(*jax.tree.leaves(defect)) > threshold
    )

  def body(carry):
    original, defect, count = carry
    correction = solve(original.replace_rhs(**defect))
    base = replace(
        original.base,
        **{
            name: getattr(original.base, name) + getattr(correction.base, name)
            for name in pd._STEPS
        },
    )
    improved = replace(
        original,
        base=base,
        **{
            name: tree(
                lambda x, y: x + y,
                getattr(original, name),
                getattr(correction, name),
            )
            for name in ('dp', 'dn', 'dzp', 'dzn')
        },
    )
    return improved, system_defect(improved), count + 1

  solved, defect, _ = jax.lax.while_loop(
      cond, body, (solved, names, jp.asarray(0, jp.int32))
  )
  return solved, all_finite(defect) & (
      max_abs(*jax.tree.leaves(defect)) <= threshold
  )


def refresh_barrier_terms(
    system: RestorationSystem,
    old_mu: jax.Array,
    mu: jax.Array,
    options: Options,
) -> RestorationSystem:
  """Change μ without re-evaluating proximity or constraint derivatives."""
  a, b = system, system.base
  change = mu - old_mu
  b = replace(
      b,
      r_s=b.r_s
      + change
      * options.kappa_d
      * (jp.isfinite(b.g_l).astype(b.s.dtype) - jp.isfinite(b.g_u)),
      r_sf=b.r_sf
      + change
      * options.kappa_d
      * (jp.isfinite(b.gf_l).astype(b.sf.dtype) - jp.isfinite(b.gf_u)),
      **{
          name: getattr(b, name) - jp.where(jp.isfinite(bound), change, 0.0)
          for name, bound in zip(
              ('r_c_l', 'r_c_u', 'r_cf_l', 'r_cf_u'),
              (b.g_l, b.g_u, b.gf_l, b.gf_u),
          )
      },
      _cache=None,
      **{name: None for name in pd._STEPS},
  )
  return replace(
      a,
      base=b,
      r_p=tree(lambda x: x + options.kappa_d * change, a.r_p),
      r_n=tree(lambda x: x + options.kappa_d * change, a.r_n),
      r_cp=tree(lambda x: x - change, a.r_cp),
      r_cn=tree(lambda x: x - change, a.r_cn),
      dp=None,
      dn=None,
      dzp=None,
      dzn=None,
      _cache=None,
  )
