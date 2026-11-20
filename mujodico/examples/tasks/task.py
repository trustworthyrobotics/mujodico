import dataclasses
import jax
import jax.numpy as jp
import mujoco
from mujodico.examples.tasks.task_playback import TaskPlayback
from mujodico.mjx import ContactApproxType
from mujodico.trajopt._src import manifold
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclasses.dataclass(frozen=True)
class Task:
  model: mujoco.MjModel
  horizon: int
  x_0: np.ndarray
  x_goal: np.ndarray
  Q: np.ndarray
  Qf: np.ndarray
  u_ref: np.ndarray
  R: np.ndarray
  contact_smoothing: float
  contact_approx: ContactApproxType = ContactApproxType.HUNT_CROSSLEY
  planar_normal: tuple[float, float, float] | None = None

  @property
  def nx(self) -> int:
    return self.model.nq + self.model.nv + self.model.na

  @property
  def ndx(self) -> int:
    return 2 * self.model.nv + self.model.na

  @property
  def nu(self) -> int:
    return self.model.nu

  def __post_init__(self):
    nx, nu = self.nx, self.nu
    assert self.x_0.shape == (nx,)
    assert self.x_goal.shape == (nx,)
    assert self.Q.shape == (self.ndx, self.ndx)
    assert self.Qf.shape == (self.ndx, self.ndx)
    assert self.u_ref.shape == (nu,)
    assert self.R.shape == (nu, nu)

    x_lb, x_ub = self.x_bounds
    assert (self.x_0 >= x_lb).all()
    assert (self.x_0 <= x_ub).all()
    assert (self.x_goal >= x_lb).all()
    assert (self.x_goal <= x_ub).all()
    u_lb, u_ub = self.u_bounds
    assert (self.u_ref >= u_lb).all()
    assert (self.u_ref <= u_ub).all()

  @property
  def x_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    m = self.model

    q_lb = np.full(m.nq, -np.inf)
    q_ub = np.full(m.nq, np.inf)

    for j in range(m.njnt):
      if not m.jnt_limited[j]:
        continue

      # Only scalar joints have simple bounds
      if int(m.jnt_type[j]) in (
          mujoco.mjtJoint.mjJNT_HINGE,
          mujoco.mjtJoint.mjJNT_SLIDE,
      ):
        adr = m.jnt_qposadr[j]
        q_lb[adr] = m.jnt_range[j, 0]
        q_ub[adr] = m.jnt_range[j, 1]

    v_lb = np.full(m.nv, -np.inf)
    v_ub = np.full(m.nv, np.inf)

    a_lb = np.where(
        m.actuator_actlimited,
        m.actuator_actrange[:, 0],
        -np.inf,
    )[m.actuator_actnum > 0]
    a_ub = np.where(
        m.actuator_actlimited,
        m.actuator_actrange[:, 1],
        np.inf,
    )[m.actuator_actnum > 0]

    x_lb = np.concatenate([q_lb, v_lb, a_lb])
    x_ub = np.concatenate([q_ub, v_ub, a_ub])

    return x_lb, x_ub

  @property
  def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    m = self.model

    u_lb = np.where(
        m.actuator_ctrllimited,
        m.actuator_ctrlrange[:, 0],
        -np.inf,
    )
    u_ub = np.where(
        m.actuator_ctrllimited,
        m.actuator_ctrlrange[:, 1],
        np.inf,
    )
    return u_lb, u_ub

  @property
  def xs_init(self) -> np.ndarray:
    xs = np.linspace(self.x_0, self.x_goal, self.horizon + 1)

    mask = np.diag(self.Q) > 0
    for i in self.quaternion_addrs:
      # Expand each three-dimensional tangent mask to a full quaternion block.
      mask = np.concatenate(
          [mask[:i], np.full(4, np.any(mask[i : i + 3])), mask[i + 3 :]]
      )
    xs[:, ~mask] = self.x_0[~mask]

    for i in self.quaternion_addrs:
      if np.any(mask[i : i + 4]):
        # SciPy uses [x, y, z, w]; interpolate the complete quaternion block.
        endpoints = np.stack([self.x_0[i : i + 4], self.x_goal[i : i + 4]])
        rotations = Rotation.from_quat(np.roll(endpoints, -1, axis=1))
        quaternions = Slerp([0, 1], rotations)(
            np.linspace(0, 1, self.horizon + 1)
        ).as_quat()
        xs[:, i : i + 4] = np.roll(quaternions, 1, axis=1)
      else:
        xs[:, i : i + 4] = self.x_0[i : i + 4]

    return xs

  @property
  def us_init(self) -> np.ndarray:
    return np.tile(self.u_ref, (self.horizon, 1))

  @property
  def quaternion_addrs(self) -> tuple[int, ...]:
    idxs = []
    for j in range(self.model.njnt):
      typ = self.model.jnt_type[j]
      adr = int(self.model.jnt_qposadr[j])
      if typ == mujoco.mjtJoint.mjJNT_FREE:
        idxs.append(adr + 3)
      elif typ == mujoco.mjtJoint.mjJNT_BALL:
        idxs.append(adr)
    return tuple(idxs)

  def print_summary(self):
    x_lb, x_ub = self.x_bounds
    u_lb, u_ub = self.u_bounds
    qw_idxs = self.quaternion_addrs
    print(f"""
Task: {self.__class__.__name__}
  horizon = {self.horizon}
  nx = {self.nx} (nq={self.model.nq}, nv={self.model.nv}, na={self.model.na})
  ndx = {self.ndx}
  nu = {self.nu}
  quaternion_addrs={qw_idxs} ({len(qw_idxs) if qw_idxs else 'no'} quaternion{'s' if len(qw_idxs) > 1 else ''} in state)
  x_0 = {self.x_0}
  x_goal = {self.x_goal}
  diag(Q) = {np.diag(self.Q)}
  diag(Qf) = {np.diag(self.Qf)}
  x_lb = {x_lb}
  x_ub = {x_ub}
  u_ref = {self.u_ref}
  diag(R) = {np.diag(self.R)}
  u_lb = {u_lb}
  u_ub = {u_ub}
  contact_smoothing = {self.contact_smoothing}
""")

  def update_visualization(self, visualizer: TaskPlayback, traj: dict, k: int):
    model = self.model
    visualizer.update_scene(traj['xs'][k, :model.nq])
    visualizer.set_text(str(traj.get('info', '')))


def quat_error(
    q_target: jax.Array | np.ndarray,
    q: jax.Array | np.ndarray,
) -> jax.Array:
  """Return Log(q_target**-1 * q), a shortest rotation vector in radians.

  Quaternions use scalar-first [w, x, y, z] order. Inputs have shape (..., 4)
  with broadcastable batch dimensions; the result has shape (..., 3).
  """
  q_target, q = jp.broadcast_arrays(jp.asarray(q_target), jp.asarray(q))
  assert q.shape[-1] == 4, f'Expected quaternion shape (..., 4), got {q.shape}.'

  def error(target, current):
    return manifold.quat_log(
        manifold.quat_multiply(manifold.quat_inverse(target), current)
    )

  if q.ndim == 1:
    return error(q_target, q)
  return jax.vmap(error)(q_target.reshape(-1, 4), q.reshape(-1, 4)).reshape(
      q.shape[:-1] + (3,)
  )


def state_error(
    x_goal: jax.Array | np.ndarray,
    x: jax.Array | np.ndarray,
    quaternion_addrs: tuple[int, ...],
) -> jax.Array:
  """Return x - x_goal with each quaternion block replaced by quat_error.

  The tangent error retains state order, with three rotation coordinates per
  quaternion followed by the remaining position, velocity and activation errors.
  """
  error = jp.asarray(x) - jp.asarray(x_goal)
  parts = []
  start = 0
  for i in sorted(quaternion_addrs):
    parts.extend((error[start:i], quat_error(x_goal[i : i + 4], x[i : i + 4])))
    start = i + 4
  parts.append(error[start:])
  return jp.concatenate(parts)


def state_cost(
    Q: jax.Array | np.ndarray,
    x_goal: jax.Array | np.ndarray,
    x: jax.Array | np.ndarray,
    quaternion_addrs: tuple[int, ...],
):
  error = state_error(x_goal, x, quaternion_addrs)
  return error.T @ Q @ error


def control_cost(
    R: jax.Array | np.ndarray,
    u_ref: jax.Array | np.ndarray,
    u: jax.Array | np.ndarray,
):
  return (u - u_ref).T @ R @ (u - u_ref)
