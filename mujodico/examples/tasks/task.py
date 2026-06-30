import dataclasses
import time
import mujoco
import numpy as np

try:
  import mjviser
  import plotly
  import viser
except ImportError as e:
  raise ImportError(
      "Optional visualization dependencies are not installed. "
      "Install them with:\n\n"
      "    pip install mjviser plotly"
  ) from e


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

  def __post_init__(self):
    nx = self.model.nq + self.model.nv
    nu = self.model.nu
    assert self.x_0.shape == (nx,)
    assert self.x_goal.shape == (nx,)
    assert self.Q.shape == (nx, nx)
    assert self.Qf.shape == (nx, nx)
    assert self.u_ref.shape == (nu,)
    assert self.R.shape == (nu, nu)

  @property
  def x_bounds(self):
    q_lb = np.full(self.model.nq, -np.inf)
    q_ub = np.full(self.model.nq, np.inf)

    for j in range(self.model.njnt):
      if not self.model.jnt_limited[j]:
        continue

        # Only scalar joints have simple bounds
      if self.model.jnt_type[j] in (
          mujoco.mjtJoint.mjJNT_HINGE,
          mujoco.mjtJoint.mjJNT_SLIDE,
      ):
        adr = self.model.jnt_qposadr[j]
        q_lb[adr] = self.model.jnt_range[j, 0]
        q_ub[adr] = self.model.jnt_range[j, 1]

    v_lb = np.full(self.model.nv, -np.inf)
    v_ub = np.full(self.model.nv, np.inf)

    x_lb = np.concatenate([q_lb, v_lb])
    x_ub = np.concatenate([q_ub, v_ub])

    return x_lb, x_ub

  @property
  def u_bounds(self):
    u_lb = np.where(
        self.model.actuator_ctrllimited,
        self.model.actuator_ctrlrange[:, 0],
        -np.inf,
    )
    u_ub = np.where(
        self.model.actuator_ctrllimited,
        self.model.actuator_ctrlrange[:, 1],
        np.inf,
    )
    return u_lb, u_ub

  def print_summary(self):
    print("Task:", self.__class__.__name__)
    print("  horizon =", self.horizon)
    print("  x_0 =", self.x_0)
    print("  x_goal =", self.x_goal)
    print("  Q =", str(self.Q).replace("\n", "\n" + " " * 6))
    print("  Qf =", str(self.Qf).replace("\n", "\n" + " " * 7))
    print("  u_ref =", self.u_ref)
    print("  R =", str(self.R).replace("\n", "\n" + " " * 6))
    x_lb, x_ub = self.x_bounds
    print("  x_lb =", x_lb)
    print("  x_ub =", x_ub)
    u_lb, u_ub = self.u_bounds
    print("  u_lb =", u_lb)
    print("  u_ub =", u_ub)
    print("  contact_smoothing =", self.contact_smoothing)

  def visualize_trajectory(self, xs: np.ndarray):
    model = self.model
    data = mujoco.MjData(model)

    server = viser.ViserServer()
    scene = mjviser.ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = False

    while True:
      for k in range(xs.shape[0]):
        tic = time.time()

        data.qpos = xs[k, : model.nq]
        mujoco.mj_kinematics(model, data)
        scene.update_from_mjdata(data)

        self._update_visualization(server, k, xs)

        remain = model.opt.timestep - (time.time() - tic)
        if remain > 0:
          time.sleep(remain)
      time.sleep(3.0)

  def _update_visualization(self, server: viser.ViserServer, k: int, xs: np.ndarray):
    pass
