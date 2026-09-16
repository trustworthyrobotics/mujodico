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

  <option timestep="0.05"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <default>
    <geom condim="1" />
    <default class="corner">
      <geom type="sphere" size="0.0001" rgba="1 0 0 1" contype="2" mass="0"/>
    </default>
  </default>

  <include file="right_hand3.xml"/>

  <worldbody>
    <body name="object" pos="-0.035 0 0.041">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.03" rgba="0.5 0.7 0.5 1" density="200" />
      <geom class="corner" pos="0.03 0.03 -0.03" />
      <geom class="corner" pos="0.03 -0.03 -0.03" />
      <geom class="corner" pos="-0.03 0.03 -0.03" />
      <geom class="corner" pos="-0.03 -0.03 -0.03" />
    </body>
  </worldbody>

</mujoco>
"""


class AllegroCube(Task):

  def __init__(self, seed=0):
    dir = Path(__file__).resolve().parent
    allegro_xml_path = dir / 'models' / 'wonik_allegro' / 'right_hand3.xml'

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

    horizon = 10

    x_0 = np.concatenate((data.qpos, data.qvel))
    x_0[0:4] = np.deg2rad([0, 60, 60, 40])
    x_0[4:8] = np.deg2rad([0, 10, 60, 50])
    x_0[8:12] = np.deg2rad([10, 65, 50, 60])
    x_0[12:16] = np.deg2rad([21, 95, 90, 70])

    rng = np.random.default_rng(seed)
    # goal_pos = rng.uniform([-0.03, -0.03, 0.0], [0.03, 0.03, 0.0])
    # goal_angle = np.deg2rad(rng.uniform(-30, 30))
    goal_pos = np.array([0, 0, 0])
    goal_angle = np.deg2rad(90)

    x_goal = x_0.copy()
    x_goal[16:19] += goal_pos
    x_goal[19:23] = np.array([np.cos(goal_angle/2), 0.0, 0.0, np.sin(goal_angle/2)])

    quaternion_starts = (19,)

    Q = np.diag([0.0] * 16 + [100] * 3 + [0.1] * 4 + [0.0] * 22)
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
        quaternion_starts=quaternion_starts,
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

  def _update_visualization(
      self, server: viser.ViserServer, k: int, xs: np.ndarray
  ):
    fig = plotly.subplots.make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )
    for object_dof in (0, 1, 2, 3):
      self._plot_object_dof(fig, k, xs, object_dof)

    if not hasattr(self, 'plot'):
      server.scene.add_grid('grid')
      self.plot = server.gui.add_plotly(fig, aspect=1.5)
    else:
      self.plot.figure = fig


if __name__ == '__main__':
  task = AllegroCube()
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
