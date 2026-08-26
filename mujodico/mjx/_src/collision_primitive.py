# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Collision primitives."""

from typing import Tuple

import jax
from jax import numpy as jp
from mujodico.mjx._src import math
# pylint: disable=g-importing-member
from mujodico.mjx._src.collision_types import Collision
from mujodico.mjx._src.collision_types import GeomInfo
from mujodico.mjx._src.types import Data
from mujodico.mjx._src.types import Model
# pylint: enable=g-importing-member


def collider(ncon: int):
  """Wraps collision functions for use by collision_driver."""

  def wrapper(func):
    def collide(m: Model, d: Data, _, geom: jax.Array) -> Collision:
      g1, g2 = geom.T
      info1 = GeomInfo(d.geom_xpos[g1], d.geom_xmat[g1], m.geom_size[g1])
      info2 = GeomInfo(d.geom_xpos[g2], d.geom_xmat[g2], m.geom_size[g2])
      dist, pos, frame = jax.vmap(func)(info1, info2)
      if ncon > 1:
        return jax.tree_util.tree_map(jp.concatenate, (dist, pos, frame))
      return dist, pos, frame

    collide.ncon = ncon
    return collide

  return wrapper


def _plane_sphere(
    plane_normal: jax.Array,
    plane_pos: jax.Array,
    sphere_pos: jax.Array,
    sphere_radius: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
  """Returns the distance and contact point between a plane and sphere."""
  dist = jp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@collider(ncon=1)
def plane_sphere(plane: GeomInfo, sphere: GeomInfo) -> Collision:
  """Calculates contact between a plane and a sphere."""
  n = plane.mat[:, 2]
  dist, pos = _plane_sphere(n, plane.pos, sphere.pos, sphere.size[0])
  return dist, pos, math.make_frame(n)


@collider(ncon=2)
def plane_capsule(plane: GeomInfo, cap: GeomInfo) -> Collision:
  """Calculates two contacts between a capsule and a plane."""
  n, axis = plane.mat[:, 2], cap.mat[:, 2]
  # align contact frames with capsule axis
  b, b_norm = math.normalize_with_norm(axis - n * jp.dot(n, axis))
  y, z = jp.array([0.0, 1.0, 0.0]), jp.array([0.0, 0.0, 1.0])
  b = jp.where(b_norm < 0.5, jp.where((-0.5 < n[1]) & (n[1] < 0.5), y, z), b)
  frame = jp.array([[n, b, jp.cross(n, b)]])
  segment = axis * cap.size[1]
  collisions = []
  for offset in [segment, -segment]:
    dist, pos = _plane_sphere(n, plane.pos, cap.pos + offset, cap.size[0])
    dist = jp.expand_dims(dist, axis=0)
    pos = jp.expand_dims(pos, axis=0)
    collisions.append((dist, pos, frame))
  return jax.tree_util.tree_map(lambda *x: jp.concatenate(x), *collisions)


@collider(ncon=1)
def plane_ellipsoid(plane: GeomInfo, ellipsoid: GeomInfo) -> Collision:
  """Calculates one contact between an ellipsoid and a plane."""
  n = plane.mat[:, 2]
  size = ellipsoid.size
  sphere_support = -math.normalize((ellipsoid.mat.T @ n) * size)
  pos = ellipsoid.pos + ellipsoid.mat @ (sphere_support * size)
  dist = jp.dot(n, pos - plane.pos)
  pos = pos - n * dist * 0.5
  return dist, pos, math.make_frame(n)


@collider(ncon=3)
def plane_cylinder(plane: GeomInfo, cylinder: GeomInfo) -> Collision:
  """Calculates three contacts between a cylinder and a plane."""
  n = plane.mat[:, 2]
  axis = cylinder.mat[:, 2]

  # make sure axis points towards plane
  prjaxis = jp.dot(n, axis)
  sign = -math.sign(prjaxis)
  axis, prjaxis = axis * sign, prjaxis * sign

  # compute normal distance to cylinder center
  dist0 = jp.dot(cylinder.pos - plane.pos, n)

  # remove component of -normal along axis, compute length
  vec = axis * prjaxis - n
  len_ = math.norm(vec)

  vec = jp.where(
      len_ < 1e-12,
      # disk parallel to plane: pick x-axis of cylinder, scale by radius
      cylinder.mat[:, 0] * cylinder.size[0],
      # general configuration: normalize vector, scale by radius
      math.safe_div(vec, len_) * cylinder.size[0],
  )

  # project vector on normal
  prjvec = jp.dot(vec, n)

  # scale axis by half-length
  axis *= cylinder.size[1]
  prjaxis *= cylinder.size[1]

  # compute sideways vector: vec1
  prjvec1 = -prjvec * 0.5
  vec1 = math.normalize(jp.cross(vec, axis)) * cylinder.size[0]
  vec1 *= jp.sqrt(3.0) * 0.5

  # disk parallel to plane
  d1 = dist0 + prjaxis + prjvec
  d2 = dist0 + prjaxis + prjvec1
  dist = jp.array([d1, d2, d2])
  pos = (
      cylinder.pos
      + axis
      + jp.array([
          vec - n * d1 * 0.5,
          vec1 + vec * -0.5 - n * d2 * 0.5,
          -vec1 + vec * -0.5 - n * d2 * 0.5,
      ])
  )

  # cylinder parallel to plane
  cond = jp.abs(prjaxis) < 1e-3
  d3 = dist0 - prjaxis + prjvec
  dist = jp.where(cond, dist.at[1].set(d3), dist)
  pos = jp.where(
      cond, pos.at[1].set(cylinder.pos + vec - axis - n * d3 * 0.5), pos
  )

  frame = jp.stack([math.make_frame(n)] * 3, axis=0)
  return dist, pos, frame


def _sphere_sphere(
    pos1: jax.Array, radius1: jax.Array, pos2: jax.Array, radius2: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Returns the penetration, contact point, and normal between two spheres."""
  n, dist = math.normalize_with_norm(pos2 - pos1)
  n = jp.where(dist == 0.0, jp.array([1.0, 0.0, 0.0]), n)
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + dist * 0.5)
  return dist, pos, n


@collider(ncon=1)
def sphere_sphere(s1: GeomInfo, s2: GeomInfo) -> Collision:
  """Calculates contact between two spheres."""
  dist, pos, n = _sphere_sphere(s1.pos, s1.size[0], s2.pos, s2.size[0])
  return dist, pos, math.make_frame(n)


@collider(ncon=1)
def sphere_capsule(sphere: GeomInfo, cap: GeomInfo) -> Collision:
  """Calculates one contact between a sphere and a capsule."""
  axis, length = cap.mat[:, 2], cap.size[1]
  segment = axis * length
  pt = math.closest_segment_point(
      cap.pos - segment, cap.pos + segment, sphere.pos
  )
  dist, pos, n = _sphere_sphere(sphere.pos, sphere.size[0], pt, cap.size[0])
  return dist, pos, math.make_frame(n)


def _sphere_box(
    sphere_pos: jax.Array,
    sphere_radius: jax.Array,
    box_pos: jax.Array,
    box_mat: jax.Array,
    box_half_size: jax.Array,
    fillet_radius: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Returns the penetration, contact point, and normal between sphere and
  filleted box."""

  # sphere center in box-local coordinates
  p = box_mat.T @ (sphere_pos - box_pos)

  # filleted box = Minkowski sum of a shrunken box and a sphere
  core_half_size = box_half_size - fillet_radius

  # closest point on/inside the shrunken box.
  q = jp.clip(p, -core_half_size, core_half_size)

  # vector from the point on/inside shrunken box to the sphere center
  delta = p - q
  delta_normalized, delta_norm = math.normalize_with_norm(delta)

  # candidate 1: sphere center outside shrunken box
  fillet_normal = delta_normalized
  fillet_pos = q + fillet_radius * fillet_normal

  # candidate 2: sphere center inside shrunken box
  face_distance = core_half_size - jp.abs(p)
  face_axis = jp.argmin(face_distance)

  face_sign = jp.where(p[face_axis] >= 0.0, 1.0, -1.0)

  face_normal = jp.zeros(3).at[face_axis].set(face_sign)
  face_pos = p.at[face_axis].set(face_sign * box_half_size[face_axis])

  # select surface normal and position
  use_outside = delta_norm > 0.0

  surface_normal = jp.where(
      use_outside,
      fillet_normal,
      face_normal,
  )

  surface_pos = jp.where(
      use_outside,
      fillet_pos,
      face_pos,
  )

  # collision distance and position
  dist = jp.dot(p - surface_pos, surface_normal) - sphere_radius
  n = box_mat @ -surface_normal
  pos = sphere_pos + n * (sphere_radius + 0.5 * dist)

  return dist, pos, n


@collider(ncon=1)
def sphere_box(
    sphere: GeomInfo,
    box: GeomInfo,
) -> Collision:
  """Calculates one contact between a sphere and a filleted box."""
  dist, pos, n = _sphere_box(
      sphere_pos=sphere.pos,
      sphere_radius=sphere.size[0],
      box_pos=box.pos,
      box_mat=box.mat,
      box_half_size=box.size[:3],
      fillet_radius=jp.min(box.size[:3]) * 0.001,
  )
  return dist, pos, math.make_frame(n)


def _sphere_cylinder(
    sphere_pos: jax.Array,
    sphere_radius: jax.Array,
    cylinder_pos: jax.Array,
    cylinder_axis: jax.Array,
    cylinder_radius: jax.Array,
    cylinder_half_length: jax.Array,
    fillet_radius: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Returns the penetration, contact point, and normal between sphere and
  filleted cylinder."""

  d = sphere_pos - cylinder_pos

  # cylinder-local coordinates
  z = jp.dot(d, cylinder_axis)
  radial = d - z * cylinder_axis
  radial_dir, rho = math.normalize_with_norm(radial)
  radial_dir = jp.where(rho == 0.0, jp.array([1.0, 0.0, 0.0]), radial_dir)

  # work with the positive half of the cylinder
  z_abs = jp.abs(z)
  z_sign = jp.where(z < 0.0, -1.0, 1.0)

  # center of the fillet in the (rho, |z|) plane
  rho_f = cylinder_radius - fillet_radius
  z_f = cylinder_half_length - fillet_radius

  # candidate 1: cylindrical side
  side_rho = cylinder_radius
  side_z = jp.clip(z_abs, 0.0, z_f)

  side_dr = rho - side_rho
  side_dz = z_abs - side_z
  side_dist2 = side_dr * side_dr + side_dz * side_dz

  # candidate 2: flat cap
  cap_rho = jp.clip(rho, 0.0, rho_f)
  cap_z = cylinder_half_length

  cap_dr = rho - cap_rho
  cap_dz = z_abs - cap_z
  cap_dist2 = cap_dr * cap_dr + cap_dz * cap_dz

  # candidate 3: filleted edge
  fillet_dr = rho - rho_f
  fillet_dz = z_abs - z_f

  fillet_norm = math.norm(jp.array([fillet_dr, fillet_dz]))

  fillet_dir_r = math.safe_div(fillet_dr, fillet_norm)
  fillet_dir_z = math.safe_div(fillet_dz, fillet_norm)

  fillet_rho = rho_f + fillet_radius * fillet_dir_r
  fillet_z = z_f + fillet_radius * fillet_dir_z

  fillet_dist2 = (rho - fillet_rho)**2 + (z_abs - fillet_z)**2

  fillet_valid = (rho >= rho_f) & (z_abs >= z_f)
  fillet_dist2 = jp.where(fillet_valid, fillet_dist2, jp.inf)

  # select surface coordinates
  use_fillet = (fillet_dist2 <= side_dist2) & (fillet_dist2 <= cap_dist2)

  use_side = (side_dist2 <= cap_dist2) & (~use_fillet)

  surface_rho = jp.where(
      use_fillet,
      fillet_rho,
      jp.where(use_side, side_rho, cap_rho),
  )

  surface_z = jp.where(
      use_fillet,
      fillet_z,
      jp.where(use_side, side_z, cap_z),
  )

  # surface normal and position
  side_normal = radial_dir

  cap_normal = z_sign * cylinder_axis

  fillet_normal = fillet_dir_r * radial_dir + fillet_dir_z * z_sign * cylinder_axis

  surface_normal = jp.where(
      use_fillet,
      fillet_normal,
      jp.where(use_side, side_normal, cap_normal),
  )

  surface_pos = (
      cylinder_pos
      + surface_rho * radial_dir
      + surface_z * z_sign * cylinder_axis
  )

  # collision distance and position
  n = -surface_normal
  dist = jp.dot(surface_pos - sphere_pos, n) - sphere_radius
  pos = sphere_pos + n * (sphere_radius + 0.5 * dist)

  return dist, pos, n


@collider(ncon=1)
def sphere_cylinder(sphere: GeomInfo, cylinder: GeomInfo) -> Collision:
  """Calculates one contact between a sphere and a filleted cylinder."""
  dist, pos, n = _sphere_cylinder(
      sphere_pos=sphere.pos,
      sphere_radius=sphere.size[0],
      cylinder_pos=cylinder.pos,
      cylinder_axis=cylinder.mat[:, 2],
      cylinder_radius=cylinder.size[0],
      cylinder_half_length=cylinder.size[1],
      fillet_radius=jp.min(cylinder.size[0:2]) * 0.001,
  )
  return dist, pos, math.make_frame(n)


@collider(ncon=1)
def capsule_capsule(cap1: GeomInfo, cap2: GeomInfo) -> Collision:
  """Calculates one contact between two capsules."""
  axis1, length1 = cap1.mat[:, 2], cap1.size[1]
  axis2, length2 = cap2.mat[:, 2], cap2.size[1]
  seg1, seg2 = axis1 * length1, axis2 * length2
  pt1, pt2 = math.closest_segment_to_segment_points(
      cap1.pos - seg1,
      cap1.pos + seg1,
      cap2.pos - seg2,
      cap2.pos + seg2,
  )
  radius1, radius2 = cap1.size[0], cap2.size[0]
  dist, pos, n = _sphere_sphere(pt1, radius1, pt2, radius2)
  return dist, pos, math.make_frame(n)
