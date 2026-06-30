import mujoco
from mujodico.examples.tasks.task import Task
import numpy as np
import plotly.graph_objects as go
import viser


_XML = """
<mujoco model="push_shpere_1d">

  <option timestep="0.02"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <default>
    <joint damping="5"/>
    <geom condim="1" />
  </default>

  <worldbody>

    <body name="pusher" pos="0 0 0">
      <joint name="pusher_slide" type="slide" axis="1 0 0" />

      <geom name="pusher_geom"
            type="sphere"
            size="0.05"
            mass="1"
            rgba="1 0 0 1" />
    </body>

    <body name="object" pos="0 0 0">
      <joint name="object_slide" type="slide" axis="1 0 0" />

      <geom name="object_geom"
            type="sphere"
            size="0.05"
            mass="1"
            rgba="0 0 1 1" />
    </body>

  </worldbody>

  <actuator>
    <velocity name="pusher_vel"
        joint="pusher_slide"
        kv="100"
        ctrlrange="-1 1" />
  </actuator>

</mujoco>
"""


class PushSphere1d(Task):

  def __init__(self):
    model = mujoco.MjModel.from_xml_string(_XML)
    horizon = 100

    x_0 = np.array([-0.2, 0.0, 0.0, 0.0])
    x_goal = np.array([0.0, 1.0, 0.0, 0.0])
    Q = np.diag([0.0, 1.0, 0.0, 0.1])
    Qf = Q * 100.0
    u_ref = np.array([0.0])
    R = np.eye(1) * 0.01

    super().__init__(
        model=model,
        horizon=horizon,
        x_0=x_0,
        x_goal=x_goal,
        Q=Q,
        Qf=Qf,
        u_ref=u_ref,
        R=R,
    )

  def _update_visualization(
      self, server: viser.ViserServer, k: int, xs: np.ndarray
  ):
    ts = np.arange(xs.shape[0]) * self.model.opt.timestep
    t_range = [min(ts), max(ts)]

    y_min = min(np.min(xs[:, :2]), self.x_goal[1])
    y_max = max(np.max(xs[:, :2]), self.x_goal[1])
    y_pad = max(y_max - y_min, 1e-5) * 0.05
    y_range = [y_min - y_pad, y_max + y_pad]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ts[:k], y=xs[:k, 0], name='Pusher',
                             mode='lines', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=ts[:k], y=xs[:k, 1], name='Object',
                             mode='lines', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t_range, y=[self.x_goal[1]] * 2, name='Object goal',
                             mode='lines', line=dict(color='black', dash='dash')))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Position',
        xaxis=dict(range=t_range),
        yaxis=dict(range=y_range),
    )

    if not hasattr(self, 'plot'):
      server.scene.add_grid('grid')
      self.plot = server.gui.add_plotly(fig)
    else:
      self.plot.figure = fig


if __name__ == '__main__':
  task = PushSphere1d()
  task.print_summary()
