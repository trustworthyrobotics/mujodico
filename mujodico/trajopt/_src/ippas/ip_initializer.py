from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas.ip_iterate import Iterate, all_finite, max_abs
from mujodico.trajopt._src.ippas.ip_search_dir import dual_regularization
from mujodico.trajopt._src.ippas.options import Options
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel
from mujodico.trajopt._src.types import Params

if TYPE_CHECKING:
  from mujodico.trajopt._src.ippas.ip_algorithm import InitialGuess


def get_bounds(
    model: StagewiseOcpModel, options: Options, *, original: bool = False
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Return (stage lower, stage upper, terminal lower, terminal upper).

  Unless original=True, expand each finite bound [IPOPT, eq. (35)]:

    δ(b) = min(constr_viol_tol, bound_relax_factor × max(1, |b|)),
    lower_relaxed = lower − δ(lower),
    upper_relaxed = upper + δ(upper).
  """
  m = model
  lower, upper = (
      jp.broadcast_to(jp.asarray(v), (m.N, v.size))
      for v in m.stage_inequality_bounds
  )
  lower_f, upper_f = map(jp.asarray, m.terminal_inequality_bounds)
  bounds = lower, upper, lower_f, upper_f
  if original:
    return bounds
  result = []
  for i, bound in enumerate(bounds):
    finite = jp.isfinite(bound)
    change = jp.minimum(
        options.constr_viol_tol,
        options.bound_relax_factor
        * jp.maximum(1.0, jp.abs(jp.where(finite, bound, 0.0))),
    )
    result.append(jp.where(finite, bound + (-1 if i % 2 == 0 else 1) * change, bound))
  return tuple(result)


def push_interior(
    value: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    options: Options,
) -> jax.Array:
  """Clip values into a shrunken bound interval [IPOPT, sec. 3.6].

  Let κ₁ = options.bound_push and κ₂ = options.bound_frac. If both l and u
  are finite, choose margins

    pˡ = min(κ₁ max(1, |l|), κ₂ (u − l)),
    pᵘ = min(κ₁ max(1, |u|), κ₂ (u − l)),

  then clip value into [l + pˡ, u − pᵘ]. For a one-sided bound, use only
  κ₁ max(1, |bound|); no interval-width cap applies. An unbounded side imposes
  no clipping. Values already inside the shrunken interval remain unchanged.

  Guard finite endpoints with nextafter(l, +∞) and nextafter(u, −∞), so a
  push cannot round back onto its bound. Strict interiority requires that
  the bounds admit a representable interior point. initialize() applies
  this projection to inequality slacks, preserving the supplied trajectory.
  """
  ml, mu = jp.isfinite(lower), jp.isfinite(upper)
  # Form finite intervals before taking their difference to avoid inf-inf.
  width = jp.where(ml & mu, upper, 1.0) - jp.where(ml & mu, lower, 0.0)
  pl = jp.minimum(
      options.bound_push * jp.maximum(1.0, jp.abs(lower)),
      options.bound_frac * width,
  )
  pu = jp.minimum(
      options.bound_push * jp.maximum(1.0, jp.abs(upper)),
      options.bound_frac * width,
  )
  pl = jp.where(ml & mu, pl, options.bound_push * jp.maximum(1.0, jp.abs(lower)))
  pu = jp.where(ml & mu, pu, options.bound_push * jp.maximum(1.0, jp.abs(upper)))

  lo = jp.where(ml, jp.maximum(lower + pl, jp.nextafter(lower, jp.inf)), -jp.inf)
  hi = jp.where(mu, jp.minimum(upper - pu, jp.nextafter(upper, -jp.inf)), jp.inf)

  return jp.clip(value, lo, hi)


def initialize(
    model: StagewiseOcpModel,
    params: Params,
    guess: 'InitialGuess',
    bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    options: Options,
) -> Iterate:
  """Push cold-start slacks, or preserve a primal-dual warm start."""
  m, o = model, options
  dtype = m.evaluation_dtype
  lower, upper, lowerf, upperf = bounds

  xs, us = jp.asarray(guess.xs, dtype=dtype), jp.asarray(guess.us, dtype=dtype)
  if xs.shape != (m.N + 1, m.nx) or us.shape != (m.N, m.nu):
    raise ValueError(
        f'Initial guesses must have shapes {(m.N + 1, m.nx)} and {(m.N, m.nu)}.'
    )

  g = m.stage_inequality_residual(xs[:-1], us, params)
  gf = m.terminal_inequality_residual(xs[-1], params)
  z_l = jp.isfinite(lower).astype(dtype)
  z_u = jp.isfinite(upper).astype(dtype)
  zf_l = jp.isfinite(lowerf).astype(dtype)
  zf_u = jp.isfinite(upperf).astype(dtype)
  it = Iterate(
      x=xs,
      u=us,
      s=push_interior(g, lower, upper, o),
      sf=push_interior(gf, lowerf, upperf, o),
      lmbda=jp.zeros((m.N + 1, m.nx_tangent), dtype=dtype),
      beta=jp.zeros((m.N, m.constraint_sizes[0]), dtype=dtype),
      y=z_u - z_l,
      yf=zf_u - zf_l,
      z_l=z_l,
      z_u=z_u,
      zf_l=zf_l,
      zf_u=zf_u,
  )

  if guess.primal_dual is not None:
    warm = jax.tree.map(lambda v: jp.asarray(v, dtype=dtype), guess.primal_dual)
    for a, b in zip(jax.tree.leaves(it), jax.tree.leaves(warm)):
      if a.shape != b.shape:
        raise ValueError('Warm-start shapes must match this model.')
    # The supplied primal arrays take precedence over the saved trajectories.
    it = replace(warm, x=xs, u=us)

  return it


def initialize_multipliers(
    system: pd.IpPrimalDualSystem, options: Options
) -> tuple[jax.Array, jax.Array]:
  """Estimate cold-start λ/β using [IPOPT, sec. 3.6].

  Supply a complete system built at μ = 0 with zero initial λ/β. With J
  stacking initial, dynamics, and stage-equality Jacobians, the structured
  projection has the form

    [ I  Jᵀ ] [ w ] = −[ ∇ℓ + Gᵀ(zᵘ − zˡ) ],
    [ J  0  ] [ ν ]    [   0              ].   [IPOPT eq. (36)]
  """
  zeros = {
      name: jp.zeros_like(getattr(system, name))
      for name in (
          'z_l',
          'z_u',
          'zf_l',
          'zf_u',
          'Hxu',
          'r_f',
          'r_e',
          'r_g',
          'r_c_l',
          'r_c_u',
          'r_gf',
          'r_cf_l',
          'r_cf_u',
          'r0',
      )
  }
  system = replace(
      system,
      **zeros,
      delta_c=dual_regularization(system.Eu, jp.asarray(options.mu_init), options),
      _cache=None,
      Hxx=jp.broadcast_to(jp.eye(system.Hxx.shape[-1]), system.Hxx.shape),
      Huu=jp.broadcast_to(jp.eye(system.Huu.shape[-1]), system.Huu.shape),
  )
  solution = pd.solve(system)

  lmbda = solution.dlmbda
  beta = solution.dbeta
  valid = all_finite((lmbda, beta)) & (max_abs(lmbda, beta) <= options.lam_max)

  return (
      jp.where(valid, lmbda, jp.zeros_like(lmbda)),
      jp.where(valid, beta, jp.zeros_like(beta)),
  )
