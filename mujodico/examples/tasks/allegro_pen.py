from pathlib import Path
import tempfile
import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks import plotly_util
import numpy as np
import plotly.graph_objects as go
import plotly.subplots
import viser


_XML = """
<mujoco model="right_allegro_hand scene">

  <option timestep="0.02"  />
  <option integrator="implicitfast" />
  <option iterations="40" tolerance="1e-6" />

  <include file="right_hand2.xml"/>

  <worldbody>
    <body name="object" pos="-0.025 0 0.026">
      <joint name="object_joint" type="free" damping="0.05" armature="0.002"/>
      <geom type="capsule" size="0.015" fromto="0 -0.08 0 0 0.08 0" rgba="0.5 0.7 0.5 1"
            friction="0.7" density="600" solref="0.02 1" solimp="0.9 0.95 0.001"/>
      <geom name="angle_indicator" type="sphere" size="0.005" pos="0 0 0.011" rgba="0 0 0 1"
            contype="0" conaffinity="0" />
    </body>
  </worldbody>

</mujoco>
"""


class AllegroPen(Task):

  def __init__(self, seed=0):
    dir = Path(__file__).resolve().parent
    allegro_xml_path = dir / 'models' / 'wonik_allegro' / 'right_hand2.xml'

    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.xml',
        dir=allegro_xml_path.parent,
        delete=False,
    ) as f:
      f.write(_XML)

    scene_path = f.name
    model = mujoco.MjModel.from_xml_path(scene_path)
    Path(scene_path).unlink()
    data = mujoco.MjData(model)

    horizon = 200

    x_0 = np.concatenate((data.qpos, data.qvel))
    x_0[0:4] = np.deg2rad([0, 70, 80, 60])
    x_0[4:8] = np.deg2rad([0, 40, 60, 30])
    x_0[8:12] = np.deg2rad([0, 70, 80, 60])
    x_0[12:16] = np.deg2rad([50, 110, 80, 60])

    x_goal = x_0.copy()
    x_goal[16:19] += np.array([0.051, -0.022, 0.074])
    x_goal[19:23] = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0, 0.0])

    Q = np.diag([0.0] * 16 + [100] * 3 + [0.01] * 4 + [0.0] * 22)
    Qf = Q * 40.0

    u_ref = x_0[:16]
    R = np.eye(16) * 0.01

    contact_smoothing = 0.002

    super().__init__(
        model=model,
        horizon=horizon,
        x_0=x_0,
        x_goal=x_goal,
        Q=Q,
        Qf=Qf,
        u_ref=u_ref,
        R=R,
        contact_smoothing=contact_smoothing,
    )

  def _plot_object_dof(self, fig: go.Figure, k: int, xs: np.ndarray, object_dof: int):
    ts = np.arange(xs.shape[0]) * self.model.opt.timestep
    t_range = [min(ts), max(ts)]

    dof = 16 + object_dof
    if object_dof < 3:
      values = xs[:, dof]
      goal = self.x_goal[dof]
    else:
      values = 1.0 - (xs[:, dof:dof+4] @ self.x_goal[dof:dof+4])**2
      goal = 0.0

    y_min = min(np.min(values), goal)
    y_max = max(np.max(values), goal)
    y_pad = max(y_max - y_min, 1e-5) * 0.05
    y_range = [y_min - y_pad, y_max + y_pad]

    plotly_util.add_line(
        fig=fig,
        x=ts[:k+1],
        y=values[:k+1],
        color='blue',
        row=object_dof + 1,
        col=1,
    )
    plotly_util.add_line(
        fig=fig,
        x=t_range,
        y=[goal, goal],
        color='black',
        linestyle='dash',
        row=object_dof + 1,
        col=1,
    )
    fig.update_xaxes(
        title_text='Time (sec)' if object_dof == 3 else None,
        range=t_range,
        row=object_dof + 1,
        col=1,
    )
    fig.update_yaxes(
        title_text=['X Position (m)', 'Y Position (m)', 'Z Position (m)', 'Quaternion error'][object_dof],
        range=y_range,
        row=object_dof + 1,
        col=1,
    )

  def update_visualization(self, server: viser.ViserServer, traj: dict, k: int):
    super().update_visualization(server, traj, k)

    if '/grid' not in server.scene._handle_from_node_name:
      server.scene.add_grid('/grid')

    fig = plotly.subplots.make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )
    for object_dof in (0, 1, 2, 3):
      self._plot_object_dof(fig, k, traj['xs'], object_dof)

    if not hasattr(server, 'plotly'):
      server.plotly = server.gui.add_plotly(fig)
    else:
      server.plotly.figure = fig


if __name__ == '__main__':
  task = AllegroPen()
  task.print_summary()

  import mjviser
  model = task.model
  server = viser.ViserServer()
  scene = mjviser.ViserMujocoScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False

  data = mujoco.MjData(model)
  data.qpos = task.x_0[:model.nq]
  mujoco.mj_kinematics(model, data)
  scene.update_from_mjdata(data)
  server.sleep_forever()
