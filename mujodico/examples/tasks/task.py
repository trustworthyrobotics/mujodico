import dataclasses
import time
import mujoco
from mujodico.mjx import ContactApproxType
import numpy as np

try:
  import mjviser
  import viser
except ImportError as e:
  raise ImportError(
      "Optional visualization dependencies are not installed. "
      "Install them with:\n\n"
      "    pip install mjviser"
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
  contact_approx: ContactApproxType = ContactApproxType.HUNT_CROSSLEY

  @property
  def nx(self):
    return self.model.nq + self.model.nv + self.model.na

  @property
  def nu(self):
    return self.model.nu

  def __post_init__(self):
    nx, nu = self.nx, self.nu
    assert self.x_0.shape == (nx,)
    assert self.x_goal.shape == (nx,)
    assert self.Q.shape == (nx, nx)
    assert self.Qf.shape == (nx, nx)
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
  def x_bounds(self):
    m = self.model

    q_lb = np.full(m.nq, -np.inf)
    q_ub = np.full(m.nq, np.inf)

    for j in range(m.njnt):
      if not m.jnt_limited[j]:
        continue

      # Only scalar joints have simple bounds
      if m.jnt_type[j] in (
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
  def u_bounds(self):
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
  def xs_init(self):
    xs = np.linspace(self.x_0, self.x_goal, self.horizon + 1)

    mask = np.diag(self.Q) > 0
    xs[:, ~mask] = self.x_0[~mask]

    return xs

  @property
  def us_init(self):
    return np.zeros((self.horizon, self.nu))

  @property
  def quaternion_w_idxs(self):
    idxs = []
    for j in range(self.model.njnt):
      typ = self.model.jnt_type[j]
      adr = int(self.model.jnt_qposadr[j])
      if typ == mujoco.mjtJoint.mjJNT_FREE:
        idxs.append(adr + 3)
      elif typ == mujoco.mjtJoint.mjJNT_BALL:
        idxs.append(adr)
    return idxs

  def print_summary(self):
    x_lb, x_ub = self.x_bounds
    u_lb, u_ub = self.u_bounds
    qw_idxs = self.quaternion_w_idxs
    print(f'''
Task: {self.__class__.__name__}
  horizon = {self.horizon}
  nx = {self.nx} (nq={self.model.nq}, nv={self.model.nv}, na={self.model.na})
  nu = {self.nu}
  quaternion_w_idxs={qw_idxs} ({len(qw_idxs) if qw_idxs else 'no'} quaternion{'s' if len(qw_idxs) > 1 else ''} in state)
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
''')

  def update_visualization(self, server: viser.ViserServer, traj: dict, k: int):
    model = self.model
    data = mujoco.MjData(model)

    if not hasattr(server, 'mj_scene'):
      mj_scene = mjviser.ViserMujocoScene(server, model, num_envs=1)
      mj_scene.camera_tracking_enabled = False
      server.mj_scene = mj_scene

    data.qpos = traj['xs'][k, :model.nq]
    mujoco.mj_kinematics(model, data)
    server.mj_scene.update_from_mjdata(data)

    if not hasattr(server, 'info_panel'):
      server.info_panel = server.gui.add_markdown('')
    server.info_panel.content = str(traj.get('info', ''))

