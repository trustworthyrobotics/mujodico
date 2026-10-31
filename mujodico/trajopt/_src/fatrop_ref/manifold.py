"""Retractions and local coordinates used by every IPPAS trial step.

Quaternions use Hamilton ``[w, x, y, z]`` coordinates and a right/body-frame
rotation increment.  The implementations have finite first and second
derivatives at the identity; no differentiation through a zero norm is used.
"""

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np


def quat_mul(a, b):
  return jnp.concatenate((
      jnp.atleast_1d(a[0] * b[0] - jnp.dot(a[1:], b[1:])),
      a[0] * b[1:] + b[0] * a[1:] + jnp.cross(a[1:], b[1:]),
  ))


def quat_normalize(q):
  return q / jnp.sqrt(jnp.maximum(jnp.dot(q, q), jnp.finfo(q.dtype).tiny))


def quat_exp(v):
  """Rotation-vector exponential, including a differentiable identity limit."""
  t2 = jnp.dot(v, v)
  small = t2 < 1e-8
  t = jnp.sqrt(jnp.where(small, jnp.ones_like(t2), t2))
  w = jnp.where(small, 1 - t2 / 8 + t2 * t2 / 384, jnp.cos(t / 2))
  scale = jnp.where(small, 0.5 - t2 / 48 + t2 * t2 / 3840, jnp.sin(t / 2) / t)
  return quat_normalize(jnp.concatenate((jnp.atleast_1d(w), scale * v)))


def quat_log(q):
  """Shortest rotation vector; the usual pi cut locus is excluded."""
  q = quat_normalize(q)
  q = jnp.where(q[0] < 0, -q, q)
  w, v = q[0], q[1:]
  t2 = jnp.dot(v, v)
  small = t2 < 1e-8
  t = jnp.sqrt(jnp.where(small, jnp.ones_like(t2), t2))
  wsafe = jnp.where(small, jnp.maximum(w, 0.5), jnp.ones_like(w))
  z = t2 / (wsafe * wsafe)
  series = 2 / wsafe * (1 - z / 3 + z * z / 5)
  scale = jnp.where(small, series, 2 * jnp.arctan2(t, w) / t)
  return scale * v


class Manifold:
  """Protocol for a stored point, its tangent, and additive bound coordinates."""

  storage_dim: int
  tangent_dim: int
  native_map: np.ndarray

  def retract(self, x, delta, params=None):
    raise NotImplementedError

  def local(self, reference, x, params=None):
    raise NotImplementedError

  def valid(self, x, params=None):
    return jnp.all(jnp.isfinite(x))

  def chart_valid(self, reference, x, params=None):
    return self.valid(reference, params) & self.valid(x, params)

  def identity(self, dtype):
    raise NotImplementedError


@dataclass(frozen=True)
class Euclidean(Manifold):
  storage_dim: int

  @property
  def tangent_dim(self):
    return self.storage_dim

  @property
  def native_map(self):
    return np.eye(self.storage_dim)

  def retract(self, x, delta, params=None):
    del params
    return x + delta

  def local(self, reference, x, params=None):
    del params
    return x - reference

  def identity(self, dtype):
    return jnp.zeros(self.storage_dim, dtype)


@dataclass(frozen=True)
class Quaternion(Manifold):
  storage_dim: int = 4
  tangent_dim: int = 3

  @property
  def native_map(self):
    return np.zeros((4, 3))

  def retract(self, x, delta, params=None):
    del params
    return quat_normalize(quat_mul(quat_normalize(x), quat_exp(delta)))

  def local(self, reference, x, params=None):
    del params
    q = quat_normalize(reference)
    inv = q * jnp.asarray([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)
    return quat_log(quat_mul(inv, quat_normalize(x)))

  def valid(self, x, params=None):
    del params
    return jnp.all(jnp.isfinite(x)) & (jnp.dot(x, x) > 1e-24)

  def chart_valid(self, reference, x, params=None):
    dot = jnp.abs(jnp.dot(quat_normalize(reference), quat_normalize(x)))
    return self.valid(reference) & self.valid(x) & (dot > 1e-7)

  def identity(self, dtype):
    return jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype)


class ProductManifold(Manifold):
  """A static ordered product of Euclidean and non-Euclidean factors."""

  def __init__(self, *factors):
    if len(factors) == 1 and isinstance(factors[0], (tuple, list)):
      factors = tuple(factors[0])
    self.factors = tuple(factors)
    self.storage_dim = sum(f.storage_dim for f in self.factors)
    self.tangent_dim = sum(f.tangent_dim for f in self.factors)
    self._slices = []
    a = b = 0
    self.native_map = np.zeros((self.storage_dim, self.tangent_dim))
    for f in self.factors:
      si, ti = slice(a, a + f.storage_dim), slice(b, b + f.tangent_dim)
      self._slices.append((f, si, ti))
      self.native_map[si, ti] = f.native_map
      a, b = si.stop, ti.stop

  def retract(self, x, delta, params=None):
    return (
        jnp.concatenate(
            [f.retract(x[si], delta[ti], params) for f, si, ti in self._slices]
        )
        if self.factors
        else jnp.zeros(0, x.dtype)
    )

  def local(self, reference, x, params=None):
    return (
        jnp.concatenate(
            [f.local(reference[si], x[si], params) for f, si, _ in self._slices]
        )
        if self.factors
        else jnp.zeros(0, x.dtype)
    )

  def valid(self, x, params=None):
    return jnp.all(
        jnp.asarray([f.valid(x[si], params) for f, si, _ in self._slices])
    )

  def chart_valid(self, reference, x, params=None):
    return jnp.all(
        jnp.asarray([
            f.chart_valid(reference[si], x[si], params)
            for f, si, _ in self._slices
        ])
    )

  def identity(self, dtype):
    return (
        jnp.concatenate([f.identity(dtype) for f in self.factors])
        if self.factors
        else jnp.zeros(0, dtype)
    )


class CustomManifold(Manifold):
  """User-defined smooth operations, with optional additive bound coordinates.

  ``native_map`` must contain only genuinely additive storage-coordinate rows.
  Rows left at zero route finite storage bounds through nonlinear constraints.
  """

  def __init__(
      self,
      storage_dim: int,
      tangent_dim: int,
      retract: Callable,
      local: Callable,
      *,
      native_map=None,
      valid=None,
      identity=None,
  ):
    self.storage_dim = storage_dim
    self.tangent_dim = tangent_dim
    self.native_map = (
        np.zeros((storage_dim, tangent_dim))
        if native_map is None
        else np.asarray(native_map)
    )
    if self.native_map.shape != (storage_dim, tangent_dim):
      raise ValueError('native_map has an incompatible shape')
    self._retract, self._local = retract, local
    self._valid = valid
    self._identity = identity

  def retract(self, x, delta, params=None):
    return self._retract(x, delta, params)

  def local(self, reference, x, params=None):
    return self._local(reference, x, params)

  def valid(self, x, params=None):
    return (
        super().valid(x, params)
        if self._valid is None
        else self._valid(x, params)
    )

  def identity(self, dtype):
    return (
        jnp.zeros(self.storage_dim, dtype)
        if self._identity is None
        else jnp.asarray(self._identity, dtype)
    )


def quaternion_product(storage_dim, quaternion_indices):
  """Construct a state product from quaternion starting indices."""
  factors = []
  offset = 0
  for i in sorted(quaternion_indices):
    if i < offset or i + 4 > storage_dim:
      raise ValueError('Quaternion slices overlap or exceed the state size')
    if i > offset:
      factors.append(Euclidean(i - offset))
    factors.append(Quaternion())
    offset = i + 4
  if offset < storage_dim:
    factors.append(Euclidean(storage_dim - offset))
  return ProductManifold(*factors)
