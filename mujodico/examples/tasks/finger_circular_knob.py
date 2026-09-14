import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks import plotly_util
import numpy as np
import plotly.graph_objects as go
import viser


_XML = """
<mujoco model="finger_circular_knob">

  <option timestep="0.02"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <default>
    <geom condim="3" friction="0.3" />
  </default>

  <worldbody>

    <body name="finger_link1" pos="-0.15 -0.25 0">
      <joint name="finger_joint1" type="hinge" range="-90 90" />
      <geom type="capsule" size="0.03" fromto="0 0 0 0 0.15 0" />

      <body name="finger_link2" pos="0 0.15 0">
        <joint name="finger_joint2" type="hinge" range="-90 90" />
        <geom type="capsule" size="0.03" fromto="0 0.06 0 0 0.15 0" />
      </body>
    </body>

    <body name="knob" pos="0 0 0">
      <joint name="knob_rotate" type="hinge" damping="0.05" />

      <geom type="sphere" size="0.08" rgba="0 0 1 1" />
      <geom name="angle_indicator" type="sphere" size="0.012" pos="0.05 0 0.05" rgba="0 0 0 1"
            contype="0" conaffinity="0" />
    </body>

  </worldbody>

  <actuator>
    <velocity name="finger_joint1_vel" joint="finger_joint1" kv="1" ctrlrange="-2 2" />
    <velocity name="finger_joint2_vel" joint="finger_joint2" kv="1" ctrlrange="-2 2" />
  </actuator>

</mujoco>
"""


class FingerCircularKnob(Task):

  def __init__(self, seed=0):
    model = mujoco.MjModel.from_xml_string(_XML)
    horizon = 400

    x_0 = np.concatenate((
      np.deg2rad([30, -60]),
      np.deg2rad([0]),
      [0.0] * 3,
    ))
    x_goal = np.concatenate((
      np.deg2rad([0, 0]),
      np.deg2rad([360]),
      [0.0] * 3,
    ))
    Q = np.diag([0.0, 0.0, 1.0, 0.0, 0.0, 0.001])
    Qf = Q * 400

    u_ref = np.array([0.0, 0.0])
    R = np.eye(2) * 0.01

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

  def _update_visualization(
      self, server: viser.ViserServer, k: int, xs: np.ndarray
  ):
    ts = np.arange(xs.shape[0]) * self.model.opt.timestep
    t_range = [min(ts), max(ts)]

    y_min = min(np.min(xs[:, 2]), self.x_goal[2])
    y_max = max(np.max(xs[:, 2]), self.x_goal[2])
    y_pad = max(y_max - y_min, 1e-5) * 0.05
    y_range = np.rad2deg([y_min - y_pad, y_max + y_pad])

    fig1 = go.Figure()
    plotly_util.add_line(
        fig=fig1,
        x=ts[:k+1],
        y=np.rad2deg(xs[:k+1, 2]),
        name='Angle',
        color='blue',
    )
    plotly_util.add_line(
        fig=fig1,
        x=t_range,
        y=np.rad2deg([self.x_goal[2]] * 2),
        name='Angle goal',
        color='black',
        linestyle='dash',
    )
    fig1.update_layout(
        xaxis_title='Time (sec)',
        yaxis_title='Angle (deg)',
        xaxis={'range': t_range},
        yaxis={'range': y_range, 'dtick': 45},
    )
    if not hasattr(self, 'plot1'):
      server.scene.add_grid('grid')
      self.plot1 = server.gui.add_plotly(fig1, aspect=0.8)
    else:
      self.plot1.figure = fig1


if __name__ == '__main__':
  task = FingerCircularKnob()
  task.print_summary()
