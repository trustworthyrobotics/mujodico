# Portions follow FATROP's Ipopt-derived interior-point policies.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
"""Elastic L1 feasibility restoration with manifold-local proximity.

The algorithmic policy follows FATROP/Ipopt's restoration phase.  See the
EPL-2.0 provenance notice shipped with this package. Dynamics are never
elasticized; p/n pairs apply to the other physical equality rows.
"""

from typing import NamedTuple
from dataclasses import replace
from . import iterate as bounds
import numpy as np
import jax
import jax.numpy as jnp
from . import filter as filters
from .problem import Problem, Primal
from .manifold import Euclidean, Manifold, ProductManifold


class RestorationParams(NamedTuple):
  original: object
  reference: object


class _OriginalParamsManifold(Manifold):
  """Keep user geometry callbacks on their original parameter PyTree."""

  def __init__(self, manifold):
    self.base = manifold
    self.storage_dim = manifold.storage_dim
    self.tangent_dim = manifold.tangent_dim
    self.native_map = manifold.native_map

  def retract(self, x, delta, params):
    return self.base.retract(x, delta, params.original)

  def local(self, reference, x, params):
    return self.base.local(reference, x, params.original)

  def valid(self, x, params):
    return self.base.valid(x, params.original)

  def chart_valid(self, reference, x, params):
    return self.base.chart_valid(reference, x, params.original)

  def identity(self, dtype):
    return self.base.identity(dtype)


def elastic_seed(c, mu, rho):
  t = mu / (2 * rho)
  p = 0.5 * (jnp.sqrt(c * c + 4 * t * t) - c) + t
  return p, p + c


class RestorationProblem(Problem):

  def __init__(self, base, primal, params, options):
    self.base = base
    self.nh = base.stage_h(primal.xs[0], primal.us[0], params).size
    self.ng = primal.s.shape[-1]
    self.nhT = base.terminal_h(primal.xs[-1], primal.ut, params).size
    self.ngT = primal.st.size
    self.nr = self.nh + self.ng
    self.nrT = self.nhT + self.ngT
    self.rho = options.rho
    self.zeta = options.zeta
    lo = np.concatenate((base.original_u_bounds[0], np.zeros(2 * self.nr)))
    hi = np.concatenate(
        (base.original_u_bounds[1], np.full(2 * self.nr, np.inf))
    )
    lot = np.concatenate((base.original_ut_bounds[0], np.zeros(2 * self.nrT)))
    hit = np.concatenate(
        (base.original_ut_bounds[1], np.full(2 * self.nrT, np.inf))
    )
    super().__init__(
        horizon=base.N,
        num_states=base.nx,
        num_controls=base.nu + 2 * self.nr,
        stage_cost=lambda x, u, p: 0.0,
        terminal_cost=lambda x, p: 0.0,
        initial_state=lambda p: base.initial_state(p.original),
        dynamics=lambda x, u, p: base.dynamics(x, u[: base.nu], p.original),
        state_manifold=_OriginalParamsManifold(base.state_manifold),
        control_manifold=ProductManifold((
            _OriginalParamsManifold(base.control_manifold),
            Euclidean(2 * self.nr),
        )),
        x_bounds=base.original_x_bounds,
        u_bounds=(lo, hi),
        terminal_control_dim=base.nt + 2 * self.nrT,
        terminal_control_bounds=(lot, hit),
    )

  def stage_objective(self, x, u, params):
    del x, params
    return self.rho * jnp.sum(u[self.base.nu :])

  def terminal_objective(self, x, ut, params):
    del x, params
    return self.rho * jnp.sum(ut[self.base.nt :])

  def stage_h(self, x, u, params):
    off = self.base.nu
    p, n = u[off : off + self.nr], u[off + self.nr :]
    return (
        self.base.stage_h(x, u[:off], params.original)
        + p[: self.nh]
        - n[: self.nh]
    )

  def stage_g(self, x, u, params):
    off = self.base.nu
    p, n = u[off : off + self.nr], u[off + self.nr :]
    return (
        self.base.stage_g(x, u[:off], params.original)
        + p[self.nh :]
        - n[self.nh :]
    )

  def terminal_h(self, x, ut, params):
    off = self.base.nt
    p, n = ut[off : off + self.nrT], ut[off + self.nrT :]
    return (
        self.base.terminal_h(x, ut[:off], params.original)
        + p[: self.nhT]
        - n[: self.nhT]
    )

  def terminal_g(self, x, ut, params):
    off = self.base.nt
    p, n = ut[off : off + self.nrT], ut[off + self.nrT :]
    return (
        self.base.terminal_g(x, ut[:off], params.original)
        + p[self.nhT :]
        - n[self.nhT :]
    )

  def _stage_g_bounds(self, x, u, params):
    return self.base._stage_g_bounds(x, u[: self.base.nu], params.original)

  def _terminal_g_bounds(self, x, ut, params):
    return self.base._terminal_g_bounds(x, ut[: self.base.nt], params.original)

  def project(self, primal):
    return Primal(
        primal.xs,
        primal.us[:, : self.base.nu],
        primal.s,
        primal.st,
        primal.ut[: self.base.nt],
    )

  def seed(self, primal, params, mu):
    e = self.base.evaluate(primal, params)
    p, n = elastic_seed(e.eq[:, self.base.nxt :], mu, self.rho)
    pt, nt = elastic_seed(e.eqT, mu, self.rho)
    return Primal(
        primal.xs,
        jnp.concatenate((primal.us, p, n), axis=-1),
        primal.s,
        primal.st,
        jnp.concatenate((primal.ut, pt, nt)),
    )

  def _proximity(self, primal, params):
    ref = params.reference
    dx = jax.vmap(
        lambda a, b: self.base.state_manifold.local(a, b, params.original)
    )(ref.xs, primal.xs)
    du = jax.vmap(
        lambda a, b: self.base.control_manifold.local(a, b, params.original)
    )(ref.us, primal.us[:, : self.base.nu])
    return (
        0.5
        * self.zeta
        * (
            jnp.sum(dx * dx)
            + jnp.sum(du * du)
            + jnp.sum((primal.s - ref.s) ** 2)
            + jnp.sum((primal.st - ref.st) ** 2)
            + jnp.sum((primal.ut[: self.base.nt] - ref.ut) ** 2)
        )
    )

  def valid(self, primal, params):
    states = jax.vmap(
        lambda a, b: self.base.state_manifold.chart_valid(a, b, params.original)
    )(params.reference.xs, primal.xs)
    controls = jax.vmap(
        lambda a, b: self.base.control_manifold.chart_valid(
            a, b, params.original
        )
    )(params.reference.us, primal.us[:, : self.base.nu])
    return super().valid(primal, params) & jnp.all(states) & jnp.all(controls)

  def evaluate(self, primal, params):
    result = super().evaluate(primal, params)
    return result._replace(
        objective=result.objective + self._proximity(primal, params)
    )

  def linearize(
      self,
      primal,
      eq,
      eqT,
      params,
      objective_scale=1.0,
      hessian_mode='exact',
      stagewise=False,
  ):
    system, rhs = super().linearize(
        primal, eq, eqT, params, objective_scale, hessian_mode, stagewise
    )

    def derivatives(manifold, ref, value):
      def prox(d):
        displacement = manifold.local(
            ref, manifold.retract(value, d, params.original), params.original
        )
        return (
            0.5
            * self.zeta
            * objective_scale
            * jnp.dot(displacement, displacement)
        )

      zero = jnp.zeros(manifold.tangent_dim, value.dtype)
      return jax.hessian(prox)(zero), jax.grad(prox)(zero)

    hx, gx = jax.vmap(lambda a, b: derivatives(self.base.state_manifold, a, b))(
        params.reference.xs, primal.xs
    )
    hu, gu = jax.vmap(
        lambda a, b: derivatives(self.base.control_manifold, a, b)
    )(params.reference.us, primal.us[:, : self.base.nu])
    nu = self.base.nut
    si = self.nut
    ns = primal.s.shape[-1]
    nt = self.base.nt
    st = self.nt
    z = self.zeta * objective_scale
    R = (
        system.R.at[:, :nu, :nu]
        .add(hu)
        .at[:, si : si + ns, si : si + ns]
        .add(z * jnp.eye(ns, dtype=system.R.dtype))
    )
    RT = (
        system.RT.at[:nt, :nt]
        .add(z * jnp.eye(nt, dtype=system.RT.dtype))
        .at[st:, st:]
        .add(z * jnp.eye(primal.st.size, dtype=system.RT.dtype))
    )
    gy = (
        rhs.gy.at[:, :nu]
        .add(gu)
        .at[:, si : si + ns]
        .add(z * (primal.s - params.reference.s))
    )
    gTy = (
        rhs.gTy.at[:nt]
        .add(z * (primal.ut[:nt] - params.reference.ut))
        .at[st:]
        .add(z * (primal.st - params.reference.st))
    )
    return system._replace(
        Q=system.Q + hx[:-1], QT=system.QT + hx[-1], R=R, RT=RT
    ), rhs._replace(gx=rhs.gx + gx[:-1], gTx=rhs.gTx + gx[-1], gy=gy, gTy=gTy)


def perform_restoration(problem, current, params, mu, filter_state, options):
  from .solver import _run
  from .initialization import initialize
  from .line_search import _merit
  from .iterate import Iterate

  rp = RestorationProblem(problem, current.primal, params, options)
  seed = rp.seed(current.primal, params, mu)
  rparams = RestorationParams(params, current.primal)
  reference = problem.evaluate(current.primal, params)
  theta0 = jnp.sum(jnp.abs(reference.eq)) + jnp.sum(jnp.abs(reference.eqT))
  original_lower, original_upper = problem.bounds(current.primal, params)
  lo, hi = bounds.relax(
      original_lower,
      original_upper,
      options.bound_relax_factor,
      options.constr_viol_tol,
  )

  def finish(iterate):
    primal = rp.project(iterate.primal)
    phi, theta, finite = _merit(
        problem, primal, params, mu, lo, hi, options.kappa_d
    )
    return (
        finite
        & (theta <= options.resto_reduction * theta0)
        & filters.acceptable(filter_state, theta, phi)
    )

  ro = replace(
      options,
      max_iter=options.max_resto_iter,
      enable_restoration=False,
      enable_watchdog=False,
      initialize_dual=False,
      bound_push=1e-8,
      bound_relax_factor=0.0,
      max_soft_resto_iters=0,
      hessian_mode='exact_lagrangian',
  )
  result = _run(
      rp,
      rparams,
      seed,
      ro,
      allow_restore=False,
      finish_test=finish,
      initial_mu=jnp.maximum(mu, jnp.minimum(theta0, 1.0)),
  )
  projected = rp.project(result.primal)
  # Reinitialize original duals after crossing between the two NLPs.
  seeded, _, _, _, _ = initialize(
      problem, params, projected, options, preserve_primal=True
  )
  okay = finish(
      Iterate(result.primal, result.eq, result.eqT, result.zl, result.zu)
  )
  return seeded, okay, result.iterations, result.status
