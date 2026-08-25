import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks import plotly_util
import numpy as np
import plotly.graph_objects as go
import viser


_XML = """
<mujoco model="turn_square_knob">

  <option timestep="0.02"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <default>
    <geom condim="3" />
  </default>

  <worldbody>

    <body name="finger" pos="0 0 0">
      <joint name="finger_slide_x" type="slide" axis="1 0 0" range="-0.5 0.5" />
      <joint name="finger_slide_y" type="slide" axis="0 1 0" range="-0.5 0.5" />

      <geom name="finger_geom"
            type="sphere"
            size="0.02"
            rgba="1 0 0 1" />
    </body>

    <body name="knob" pos="0 0 0">
      <joint name="knob_rotate" type="hinge" damping="0.05" />

      <geom name="knob_geom"
            type="cylinder"
            size="0.08"
            fromto="-0.08 0 0 0.08 0 0"
            rgba="0 0 1 1" />

      <geom name="angle_indicator"
            type="box"
            pos="0.04 0 0"
            size="0.04 0.003 0.08"
            rgba="0 0 0 1"
            contype="0"
            conaffinity="0" />
    </body>

  </worldbody>

  <actuator>
    <velocity name="finger_vel_x" joint="finger_slide_x" kv="10" ctrlrange="-0.2 0.2" />
    <velocity name="finger_vel_y" joint="finger_slide_y" kv="10" ctrlrange="-0.2 0.2" />
  </actuator>

</mujoco>
"""


class TurnSquareKnob(Task):

  def __init__(self):
    model = mujoco.MjModel.from_xml_string(_XML)
    horizon = 400

    x_0 = np.array([-0.2, -0.2, 0.0] + [0.0] * 3)
    x_goal = np.array([0.0, 0.0, np.pi * 2.0] + [0.0] * 3)
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

    fig1.add_trace(go.Scatter(x=ts[:k], y=np.rad2deg(xs[:k, 2]), name='Angle',
                              mode='lines', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=t_range, y=np.rad2deg([self.x_goal[2]] * 2), name='Angle goal',
                              mode='lines', line=dict(color='black', dash='dash')))
    fig1.update_layout(
        xaxis_title='Time (sec)',
        yaxis_title='Angle (deg)',
        xaxis=dict(range=t_range),
        yaxis=dict(range=y_range, dtick=45),
    )

    if not hasattr(self, 'plot1'):
      server.scene.add_grid('grid')
      self.plot1 = server.gui.add_plotly(fig1, aspect=0.8)
    else:
      self.plot1.figure = fig1

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=xs[:k, 0], y=xs[:k, 1], name='Finger',
                              mode='lines', line=dict(color='red')))
    plotly_util.add_circle(fig2, x=xs[k-1, 0], y=xs[k-1, 1], r=0.02, color='red')

    plotly_util.add_rectangle(fig2, x=0.0, y=0.0, theta=xs[k-1, 2],
                              w=0.16, h=0.16, color='blue',
                              name='Knob')

    a = xs[k-1, 2]
    plotly_util.add_rectangle(fig2, x=0.04 * np.cos(a), y=0.04 * np.sin(a), theta=a,
                              w=0.08, h=0.003, color='black')

    fig2.update_layout(
        xaxis_title='x',
        yaxis_title='y',
        xaxis=dict(range=[-0.5, 0.5], scaleanchor='y', scaleratio=1),
        yaxis=dict(range=[-0.5, 0.5]),
    )

    if not hasattr(self, 'plot2'):
      server.scene.add_grid('grid')
      self.plot2 = server.gui.add_plotly(fig2)
    else:
      self.plot2.figure = fig2


if __name__ == '__main__':
  task = TurnSquareKnob()
  task.print_summary()
