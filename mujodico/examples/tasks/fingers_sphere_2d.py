import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks import plotly_util
import numpy as np
import plotly.graph_objects as go
import plotly.subplots
import viser


_XML = """
<mujoco model="fingers_sphere_2d">

  <option timestep="0.02"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <default>
    <geom condim="3" friction="0.2" />
  </default>

  <worldbody>

    <body name="left_link1" pos="-0.15 0 0">
      <joint name="left_joint1" type="hinge" range="-90 90" />
      <geom type="capsule" size="0.03" fromto="0 0 0 0 0.15 0" />

      <body name="left_link2" pos="0 0.15 0">
        <joint name="left_joint2" type="hinge" range="-100 100" />
        <geom type="capsule" size="0.03" fromto="0 0.06 0 0 0.15 0" />
      </body>
    </body>

    <body name="right_link1" pos="0.15 0 0">
      <joint name="right_joint1" type="hinge" range="-90 90" />
      <geom type="capsule" size="0.03" fromto="0 0 0 0 0.15 0" />

      <body name="right_link2" pos="0 0.15 0">
        <joint name="right_joint2" type="hinge" range="-100 100" />
        <geom type="capsule" size="0.03" fromto="0 0.06 0 0 0.15 0" />
      </body>
    </body>

    <body name="object" pos="0 0 0">
      <joint name="object_translate_x" type="slide" axis="1 0 0" damping="1" range="-0.5 0.5" />
      <joint name="object_translate_y" type="slide" axis="0 1 0" damping="1" range="-0.5 0.5" />
      <joint name="object_rotate" type="hinge" damping="0.025" />

      <geom type="sphere" size="0.08" rgba="0 0 1 1" />
      <geom name="angle_indicator" type="sphere" size="0.012" pos="0.05 0 0.05" rgba="0 0 0 1"
            contype="0" conaffinity="0" />
    </body>

  </worldbody>

  <actuator>
    <velocity name="left_joint1_vel" joint="left_joint1" kv="1" ctrlrange="-1 1" />
    <velocity name="left_joint2_vel" joint="left_joint2" kv="1" ctrlrange="-1 1" />
    <velocity name="right_joint1_vel" joint="right_joint1" kv="1" ctrlrange="-1 1" />
    <velocity name="right_joint2_vel" joint="right_joint2" kv="1" ctrlrange="-1 1" />
  </actuator>

</mujoco>
"""


class FingersSphere2d(Task):

  def __init__(self, seed=0):
    model = mujoco.MjModel.from_xml_string(_XML)
    horizon = 100

    x_0 = np.concatenate((
        np.deg2rad([30, -60, -30, 60]),
        [0.0, 0.25, 0.0],
        [0.0] * 7,
    ))
    x_goal = np.concatenate((
        [0.0] * 4,
        [0.0, 0.25, np.pi/2],
        [0.0] * 7,
    ))
    Q = np.diag(np.concatenate((
        [0.0] * 4,
        [5.0, 5.0, 0.1],
        [0.0] * 4,
        [0.01, 0.01, 0.001],
    )))
    Qf = Q * 100

    u_ref = np.array([0.0] * 4)
    R = np.eye(4) * 0.001

    contact_smoothing = 0.01

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

    dof = 4 + object_dof
    y_min = min(np.min(xs[:, dof]), self.x_goal[dof])
    y_max = max(np.max(xs[:, dof]), self.x_goal[dof])
    y_pad = max(y_max - y_min, 1e-5) * 0.05
    y_range = [y_min - y_pad, y_max + y_pad]

    convert = np.rad2deg if object_dof == 2 else (lambda x: x)

    plotly_util.add_line(
        fig=fig,
        x=ts[:k+1],
        y=convert(xs[:k+1, dof]),
        color='blue',
        row=object_dof + 1,
        col=1,
    )
    plotly_util.add_line(
        fig=fig,
        x=t_range,
        y=convert([self.x_goal[dof]] * 2),
        color='black',
        linestyle='dash',
        row=object_dof + 1,
        col=1,
    )
    fig.update_xaxes(
        title_text='Time (sec)' if object_dof == 2 else None,
        range=t_range,
        row=object_dof + 1,
        col=1,
    )
    fig.update_yaxes(
        title_text=['X Position (m)', 'Y Position (m)', 'Angle (deg)'][object_dof],
        range=convert(y_range),
        row=object_dof + 1,
        col=1,
    )

  def _update_visualization(
      self, server: viser.ViserServer, k: int, xs: np.ndarray
  ):
    fig = plotly.subplots.make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )
    for object_dof in (0, 1, 2):
      self._plot_object_dof(fig, k, xs, object_dof)

    if not hasattr(self, 'plot'):
      server.scene.add_grid('grid')
      self.plot = server.gui.add_plotly(fig, aspect=1.2)
    else:
      self.plot.figure = fig


if __name__ == '__main__':
  task = FingersSphere2d()
  task.print_summary()
