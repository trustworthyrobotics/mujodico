import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks import plotly_util
import numpy as np
import plotly.graph_objects as go
import viser


_XML = """
<mujoco model="push_shpere_2d">

  <option timestep="0.02"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <default>
    <joint damping="5"/>
    <geom condim="1" />
  </default>

  <worldbody>

    <body name="pusher" pos="0 0 0">
      <joint name="pusher_slide_x" type="slide" axis="1 0 0" range="-1 1" />
      <joint name="pusher_slide_y" type="slide" axis="0 1 0" range="-1 1" />

      <geom name="pusher_geom"
            type="sphere"
            size="0.05"
            mass="1"
            rgba="1 0 0 1" />
    </body>

    <body name="object" pos="0 0 0">
      <joint name="object_slide_x" type="slide" axis="1 0 0" range="-1 1" />
      <joint name="object_slide_y" type="slide" axis="0 1 0" range="-1 1" />

      <geom name="object_geom"
            type="sphere"
            size="0.08"
            mass="1"
            rgba="0 0 1 1" />
    </body>

  </worldbody>

  <actuator>
    <velocity name="pusher_vel_x" joint="pusher_slide_x" kv="100" ctrlrange="-1 1" />
    <velocity name="pusher_vel_y" joint="pusher_slide_y" kv="100" ctrlrange="-1 1" />
  </actuator>

</mujoco>
"""


class PushSphere2d(Task):

  def __init__(self):
    model = mujoco.MjModel.from_xml_string(_XML)
    horizon = 100

    x_0 = np.array([-0.1, -0.15, 0.1, 0.15] + [0.0] * 4)
    x_goal = np.array([0.0, 0.0, 0.0, 0.0] + [0.0] * 4)
    Q = np.diag([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.1, 0.1])
    Qf = Q * 100.0

    u_ref = np.array([0.0, 0.0])
    R = np.eye(2) * 0.01

    contact_smoothing = 1.0

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
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=xs[:k, 0], y=xs[:k, 1], name='Pusher',
                             mode='lines', line=dict(color='red')))
    plotly_util.add_circle(fig, x=xs[k-1, 0], y=xs[k-1, 1], r=0.05, color='red')

    fig.add_trace(go.Scatter(x=xs[:k, 2], y=xs[:k, 3], name='Object',
                             mode='lines', line=dict(color='blue')))
    plotly_util.add_circle(fig, x=xs[k-1, 2], y=xs[k-1, 3], r=0.08, color='blue')

    plotly_util.add_circle(fig, x=self.x_goal[2], y=self.x_goal[3], r=0.08, color='blue',
                           name='Object goal', opacity=0.2)

    fig.update_layout(
        xaxis_title='x',
        yaxis_title='y',
        xaxis=dict(range=[-1, 1], scaleanchor='y', scaleratio=1),
        yaxis=dict(range=[-1, 1]),
    )

    if not hasattr(self, 'plot'):
      server.scene.add_grid('grid')
      self.plot = server.gui.add_plotly(fig)
    else:
      self.plot.figure = fig


if __name__ == '__main__':
  task = PushSphere2d()
  task.print_summary()
