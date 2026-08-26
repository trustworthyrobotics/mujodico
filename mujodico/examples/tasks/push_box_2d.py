import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks import plotly_util
import numpy as np
import plotly.graph_objects as go
import viser


_XML = """
<mujoco model="push_box_2d">

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
            size="0.03"
            mass="1"
            rgba="1 0 0 1" />
    </body>

    <body name="object" pos="0 0 0">
      <joint name="object_slide_x" type="slide" axis="1 0 0" range="-1 1" />
      <joint name="object_slide_y" type="slide" axis="0 1 0" range="-1 1" />
      <joint name="object_rotate" type="hinge" damping="0.08" />

      <geom name="object_geom"
            type="box"
            size="0.15 0.08 0.05"
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


class PushBox2d(Task):

  def __init__(self):
    model = mujoco.MjModel.from_xml_string(_XML)
    horizon = 200

    x_0 = np.array([-0.1, -0.15, 0.1, 0.15, 1.0] + [0.0] * 5)
    x_goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 5)
    Q = np.diag([0.0, 0.0, 5.0, 5.0, 0.1, 0.0, 0.0, 0.01, 0.01, 0.001])
    Qf = Q * 200

    u_ref = np.array([0.0, 0.0])
    R = np.eye(2) * 0.01

    contact_smoothing = 0.02

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

    plotly_util.add_line(
        fig=fig,
        x=xs[:k+1, 0],
        y=xs[:k+1, 1],
        name='Pusher',
        color='red',
    )
    plotly_util.add_circle(
        fig=fig,
        x=xs[k, 0],
        y=xs[k, 1],
        r=0.03,
        color='red',
    )
    plotly_util.add_line(
        fig=fig,
        x=xs[:k+1, 2],
        y=xs[:k+1, 3],
        name='Object',
        color='blue',
    )
    plotly_util.add_rectangle(
        fig=fig,
        x=xs[k, 2],
        y=xs[k, 3],
        theta=xs[k, 4],
        w=0.3, h=0.16,
        color='blue',
    )
    plotly_util.add_rectangle(
        fig=fig,
        x=self.x_goal[2],
        y=self.x_goal[3],
        theta=self.x_goal[4],
        w=0.3, h=0.16,
        name='Object goal',
        color='blue',
        opacity=0.2,
    )
    fig.update_layout(
        xaxis_title='x',
        yaxis_title='y',
        xaxis={'range': [-1, 1]},
        yaxis={'range': [-1, 1], 'scaleanchor': 'x', 'scaleratio': 1},
    )
    if not hasattr(self, 'plot'):
      server.scene.add_grid('grid')
      self.plot = server.gui.add_plotly(fig)
    else:
      self.plot.figure = fig


if __name__ == '__main__':
  task = PushBox2d()
  task.print_summary()
