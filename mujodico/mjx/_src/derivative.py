# Copyright 2025 DeepMind Technologies Limited
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
"""Derivative functions."""

from typing import Optional

import jax
from jax import numpy as jp
# pylint: disable=g-importing-member
from mujodico.mjx._src.types import BiasType
from mujodico.mjx._src.types import Data
from mujodico.mjx._src.types import DataJAX
from mujodico.mjx._src.types import DisableBit
from mujodico.mjx._src.types import DynType
from mujodico.mjx._src.types import GainType
from mujodico.mjx._src.types import Model
from mujodico.mjx._src.types import ModelJAX
from mujodico.mjx._src.types import OptionJAX
# pylint: enable=g-importing-member


def deriv_smooth_vel(m: Model, d: Data) -> Optional[jax.Array]:
  """Analytical derivative of smooth forces w.r.t. velocities."""
  if (
      not isinstance(m._impl, ModelJAX)
      or not isinstance(d._impl, DataJAX)
      or not isinstance(m.opt._impl, OptionJAX)
  ):
    raise ValueError('deriv_smooth_vel requires JAX MJX implementation.')

  qderiv = None

  # qDeriv += d qfrc_actuator / d qvel
  if not m.opt.disableflags & DisableBit.ACTUATION:
    affine_bias = m.actuator_biastype == BiasType.AFFINE
    bias_vel = m.actuator_biasprm[:, 2] * affine_bias
    affine_gain = m.actuator_gaintype == GainType.AFFINE
    gain_vel = m.actuator_gainprm[:, 2] * affine_gain
    ctrl = d.ctrl.at[m.actuator_dyntype != DynType.NONE].set(d.act)
    vel = bias_vel + gain_vel * ctrl
    qderiv = d._impl.actuator_moment.T @ jax.vmap(jp.multiply)(
        d._impl.actuator_moment, vel
    )

  # qDeriv += d qfrc_actuator / d qpos * timestep
  if not m.opt.disableflags & DisableBit.ACTUATION:
    affine_bias = m.actuator_biastype == BiasType.AFFINE
    bias_pos = m.actuator_biasprm[:, 1] * affine_bias
    affine_gain = m.actuator_gaintype == GainType.AFFINE
    gain_pos = m.actuator_gainprm[:, 1] * affine_gain
    ctrl = d.ctrl.at[m.actuator_dyntype != DynType.NONE].set(d.act)
    pos = bias_pos + gain_pos * ctrl
    qderiv += d._impl.actuator_moment.T @ jax.vmap(jp.multiply)(
        d._impl.actuator_moment, pos
    ) * m.opt.timestep

  # qDeriv += d qfrc_passive / d qvel
  if not m.opt.disableflags & DisableBit.DAMPER:
    if qderiv is None:
      qderiv = -jp.diag(m.dof_damping)
    else:
      qderiv -= jp.diag(m.dof_damping)
    if m.ntendon:
      qderiv -= d._impl.ten_J.T @ jp.diag(m.tendon_damping) @ d._impl.ten_J

  # qDeriv += d qfrc_passive / d qpos * timestep
  if not m.opt.disableflags & DisableBit.SPRING:
    adrs = m.jnt_dofadr
    ends = jp.concatenate([adrs[1:], jp.array([m.nv])])
    diff = jp.zeros(m.nv + 1)
    diff = diff.at[adrs].add(m.jnt_stiffness)
    diff = diff.at[ends].add(-m.jnt_stiffness)
    dof_stiffness = jp.cumsum(diff)[:-1]
    if qderiv is None:
      qderiv = -jp.diag(dof_stiffness) * m.opt.timestep
    else:
      qderiv -= jp.diag(dof_stiffness) * m.opt.timestep
    if m.ntendon:
      qderiv -= d._impl.ten_J.T @ jp.diag(m.tendon_stiffness) @ d._impl.ten_J * m.opt.timestep

  if not m.opt.disableflags & (DisableBit.DAMPER | DisableBit.SPRING):
    # TODO(robotics-simulation): fluid drag model
    if m.opt._impl.has_fluid_params:
      raise NotImplementedError('fluid drag not supported for implicitfast')

  # TODO(team): rne derivative

  return qderiv
