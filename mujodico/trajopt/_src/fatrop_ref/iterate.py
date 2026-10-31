"""Primal-dual iterates, finite-side bound algebra and KKT residual metrics.

The distances are not additional optimization variables.  All missing sides
are sanitized before arithmetic so JIT/AD never evaluates inf/inf or log(inf).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


from .problem import Primal


class Iterate(NamedTuple):
  primal: object
  eq: object
  eqT: object
  zl: object
  zu: object


class Metrics(NamedTuple):
  objective: object
  theta: object
  primal: object
  dual: object
  complementarity: object
  error: object
  barrier_error: object
  finite: object


class BoundData(NamedTuple):
  lower: object
  upper: object
  dl: object
  du: object
  ml: object
  mu: object


def tree_map(fun, *trees):
  return jax.tree_util.tree_map(fun, *trees)


def tree_sum(tree):
  return sum(jnp.sum(x) for x in jax.tree_util.tree_leaves(tree))


def tree_max(tree):
  return jnp.max(
      jnp.stack([
          jnp.max(jnp.abs(x), initial=0.0)
          for x in jax.tree_util.tree_leaves(tree)
      ])
  )


def relax(lower, upper, factor, max_relax=jnp.inf):
  lo = tree_map(
      lambda l: jnp.where(
          jnp.isfinite(l),
          l - jnp.minimum(max_relax, factor * jnp.maximum(1.0, jnp.abs(l))),
          l,
      ),
      lower,
  )
  hi = tree_map(
      lambda u: jnp.where(
          jnp.isfinite(u),
          u + jnp.minimum(max_relax, factor * jnp.maximum(1.0, jnp.abs(u))),
          u,
      ),
      upper,
  )
  return lo, hi


def data(primal, lower, upper):
  ml = tree_map(jnp.isfinite, lower)
  mu = tree_map(jnp.isfinite, upper)
  dl = tree_map(
      lambda q, l, m: jnp.where(m, q - jnp.where(m, l, 0.0), 1.0),
      primal,
      lower,
      ml,
  )
  du = tree_map(
      lambda q, u, m: jnp.where(m, jnp.where(m, u, 0.0) - q, 1.0),
      primal,
      upper,
      mu,
  )
  return BoundData(lower, upper, dl, du, ml, mu)


def push(primal, lower, upper, push_amount=0.01, fraction=0.01):
  def one(q, l, u):
    finite_l, finite_u = jnp.isfinite(l), jnp.isfinite(u)
    ll, uu = jnp.where(finite_l, l, 0.0), jnp.where(finite_u, u, 0.0)
    width = jnp.where(finite_l & finite_u, uu - ll, jnp.inf)
    pl = jnp.minimum(
        push_amount * jnp.maximum(1.0, jnp.abs(ll)), fraction * width
    )
    pu = jnp.minimum(
        push_amount * jnp.maximum(1.0, jnp.abs(uu)), fraction * width
    )
    q = jnp.where(finite_l, jnp.maximum(q, ll + pl), q)
    return jnp.where(finite_u, jnp.minimum(q, uu - pu), q)

  return tree_map(one, primal, lower, upper)


def initialize_duals(bd):
  return (
      tree_map(lambda m, d: m.astype(d.dtype), bd.ml, bd.dl),
      tree_map(lambda m, d: m.astype(d.dtype), bd.mu, bd.du),
  )


def complementarity(bd, zl, zu, target=0.0):
  return (
      tree_map(
          lambda d, z, m: jnp.where(m, d * z - target, 0.0), bd.dl, zl, bd.ml
      ),
      tree_map(
          lambda d, z, m: jnp.where(m, d * z - target, 0.0), bd.du, zu, bd.mu
      ),
  )


def damping(bd, target, kappa):
  return tree_map(
      lambda ml, mu: target
      * kappa
      * (
          ml.astype(jnp.result_type(target))
          - mu.astype(jnp.result_type(target))
      ),
      bd.ml,
      bd.mu,
  )


def condense(bd, zl, zu, target, kappa=0.0):
  cl, cu = complementarity(bd, zl, zu, target)
  theta = tree_map(
      lambda l, u, zl, zu, ml, mu: jnp.where(ml, zl / l, 0.0)
      + jnp.where(mu, zu / u, 0.0),
      bd.dl,
      bd.du,
      zl,
      zu,
      bd.ml,
      bd.mu,
  )
  omega = tree_map(lambda l, u, cl, cu: cl / l - cu / u, bd.dl, bd.du, cl, cu)
  force = tree_map(
      lambda l, u, w, d: -l + u + w + d,
      zl,
      zu,
      omega,
      damping(bd, target, kappa),
  )
  return theta, force


def dual_step(bd, zl, zu, dq, target):
  cl, cu = complementarity(bd, zl, zu, target)
  dl = tree_map(
      lambda d, z, v, c, m: jnp.where(m, -(z * v + c) / d, 0.0),
      bd.dl,
      zl,
      dq,
      cl,
      bd.ml,
  )
  du = tree_map(
      lambda d, z, v, c, m: jnp.where(m, (z * v - c) / d, 0.0),
      bd.du,
      zu,
      dq,
      cu,
      bd.mu,
  )
  return dl, du


def maximum_step(bd, zl, zu, dq, dzl, dzu, tau):
  def limit(d, v, m):
    return jnp.min(
        jnp.where(m & (v < 0), -tau * d / jnp.where(v < 0, v, -1.0), 1.0),
        initial=1.0,
    )

  ap = jnp.min(
      jnp.stack(
          [
              limit(d, v, m)
              for d, v, m in zip(
                  jax.tree_util.tree_leaves(bd.dl),
                  jax.tree_util.tree_leaves(dq),
                  jax.tree_util.tree_leaves(bd.ml),
              )
          ]
          + [
              limit(d, -v, m)
              for d, v, m in zip(
                  jax.tree_util.tree_leaves(bd.du),
                  jax.tree_util.tree_leaves(dq),
                  jax.tree_util.tree_leaves(bd.mu),
              )
          ]
      )
  )
  ad = jnp.min(
      jnp.stack(
          [
              limit(z, dz, m)
              for z, dz, m in zip(
                  jax.tree_util.tree_leaves(zl),
                  jax.tree_util.tree_leaves(dzl),
                  jax.tree_util.tree_leaves(bd.ml),
              )
          ]
          + [
              limit(z, dz, m)
              for z, dz, m in zip(
                  jax.tree_util.tree_leaves(zu),
                  jax.tree_util.tree_leaves(dzu),
                  jax.tree_util.tree_leaves(bd.mu),
              )
          ]
      )
  )
  return jnp.clip(ap, 0.0, 1.0), jnp.clip(ad, 0.0, 1.0)


def barrier(bd, target, kappa=0.0):
  value = -target * (
      tree_sum(
          tree_map(lambda d, m: jnp.where(m, jnp.log(d), 0.0), bd.dl, bd.ml)
      )
      + tree_sum(
          tree_map(lambda d, m: jnp.where(m, jnp.log(d), 0.0), bd.du, bd.mu)
      )
  )
  value += (
      target
      * kappa
      * (
          tree_sum(
              tree_map(
                  lambda d, l, u: jnp.where(l & ~u, d, 0.0), bd.dl, bd.ml, bd.mu
              )
          )
          + tree_sum(
              tree_map(
                  lambda d, u, l: jnp.where(u & ~l, d, 0.0), bd.du, bd.mu, bd.ml
              )
          )
      )
  )
  valid = (
      tree_sum(
          tree_map(
              lambda d, m: jnp.sum(m & ((d <= 0) | ~jnp.isfinite(d))),
              bd.dl,
              bd.ml,
          )
      )
      == 0
  )
  valid &= (
      tree_sum(
          tree_map(
              lambda d, m: jnp.sum(m & ((d <= 0) | ~jnp.isfinite(d))),
              bd.du,
              bd.mu,
          )
      )
      == 0
  )
  return value, valid


def violation(primal, lower, upper):
  return tree_max(
      tree_map(
          lambda q, l, u: jnp.maximum(jnp.maximum(l - q, q - u), 0.0),
          primal,
          lower,
          upper,
      )
  )


def tangent_zero(problem, primal):
  return Primal(
      jnp.zeros((problem.N + 1, problem.nxt), primal.xs.dtype),
      jnp.zeros((problem.N, problem.nut), primal.us.dtype),
      jnp.zeros_like(primal.s),
      jnp.zeros_like(primal.st),
      jnp.zeros_like(primal.ut),
  )


def tangent_retract(problem, primal, d, params):
  xs = jax.vmap(lambda x, v: problem.state_manifold.retract(x, v, params))(
      primal.xs, d.xs
  )
  xs = xs.at[0].set(problem.initial_state(params))
  us = jax.vmap(lambda u, v: problem.control_manifold.retract(u, v, params))(
      primal.us, d.us
  )
  return Primal(xs, us, primal.s + d.s, primal.st + d.st, primal.ut + d.ut)


def gradient(problem, iterate, params):
  def lag(d):
    e = problem.evaluate(
        tangent_retract(problem, iterate.primal, d, params), params
    )
    return (
        e.objective + jnp.sum(e.eq * iterate.eq) + jnp.sum(e.eqT * iterate.eqT)
    )

  g = jax.grad(lag)(tangent_zero(problem, iterate.primal))
  f = tree_map(lambda l, u: -l + u, iterate.zl, iterate.zu)
  return g._replace(
      xs=(g.xs + f.xs @ problem.state_map.astype(g.xs.dtype)).at[0].set(0.0),
      us=g.us + f.us @ problem.control_map.astype(g.us.dtype),
      s=g.s + f.s,
      st=g.st + f.st,
      ut=g.ut + f.ut,
  )


def metrics(
    problem,
    iterate,
    params,
    mu,
    lower,
    upper,
    original_lower,
    original_upper,
    options,
):
  e = problem.evaluate(iterate.primal, params)
  g = gradient(problem, iterate, params)
  bd = data(iterate.primal, lower, upper)
  cl, cu = complementarity(bd, iterate.zl, iterate.zu)
  cml, cmu = complementarity(bd, iterate.zl, iterate.zu, mu)
  p = jnp.maximum(
      jnp.maximum(
          jnp.max(jnp.abs(e.eq), initial=0.0),
          jnp.max(jnp.abs(e.eqT), initial=0.0),
      ),
      violation(iterate.primal, original_lower, original_upper),
  )
  dual = tree_max(g)
  damping_force = damping(bd, mu, options.kappa_d)
  gb = g._replace(
      xs=(g.xs + damping_force.xs @ problem.state_map.astype(g.xs.dtype))
      .at[0]
      .set(0.0),
      us=g.us + damping_force.us @ problem.control_map.astype(g.us.dtype),
      s=g.s + damping_force.s,
      st=g.st + damping_force.st,
      ut=g.ut + damping_force.ut,
  )
  comp = jnp.maximum(tree_max(cl), tree_max(cu))
  mult = jnp.concatenate((iterate.eq.ravel(), iterate.eqT.ravel()))
  zcount = tree_sum(bd.ml) + tree_sum(bd.mu)
  zsum = tree_sum(iterate.zl) + tree_sum(iterate.zu)
  sd = jnp.maximum(
      1.0,
      (jnp.sum(jnp.abs(mult)) + zsum)
      / jnp.maximum(mult.size + zcount, 1)
      / options.smax,
  )
  sc = jnp.maximum(1.0, zsum / jnp.maximum(zcount, 1) / options.smax)
  error = jnp.maximum(p, jnp.maximum(dual / sd, comp / sc))
  berr = jnp.maximum(
      p,
      jnp.maximum(
          tree_max(gb) / sd,
          jnp.maximum(tree_max(cml), tree_max(cmu)) / sc,
      ),
  )
  finite = (
      problem.valid(iterate.primal, params)
      & jnp.isfinite(error)
      & jnp.isfinite(e.objective)
  )
  return Metrics(
      e.objective,
      jnp.sum(jnp.abs(e.eq)) + jnp.sum(jnp.abs(e.eqT)),
      p,
      dual,
      comp,
      jnp.where(finite, error, jnp.inf),
      jnp.where(finite, berr, jnp.inf),
      finite,
  )
