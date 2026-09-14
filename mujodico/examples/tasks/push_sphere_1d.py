import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks import plotly_util
import numpy as np
import plotly.graph_objects as go
import viser


_XML = """
<mujoco model="push_shpere_1d">

  <option timestep="0.02"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <default>
    <geom condim="1" />
  </default>

  <worldbody>

    <body name="pusher" pos="0 0 0">
      <joint name="pusher_slide" type="slide" axis="1 0 0" range="-2 2" />
      <geom type="sphere" size="0.05" mass="0.1" rgba="1 0 0 1" />
    </body>

    <body name="object" pos="0 0 0">
      <joint name="object_slide" type="slide" axis="1 0 0" range="-2 2" damping="10" />
      <geom type="sphere" size="0.05" mass="1" rgba="0 0 1 1" />
    </body>

  </worldbody>

  <actuator>
    <velocity name="pusher_vel" joint="pusher_slide" kv="10" ctrlrange="-2 2" />
  </actuator>

</mujoco>
"""


class PushSphere1d(Task):

  def __init__(self, seed=0):
    model = mujoco.MjModel.from_xml_string(_XML)
    horizon = 100

    x_0 = np.array([-0.2, 0.0, 0.0, 0.0])
    x_goal = np.array([0.0, 1.0, 0.0, 0.0])
    Q = np.diag([0.0, 1.0, 0.0, 0.1])
    Qf = Q * 100.0

    u_ref = np.array([0.0])
    R = np.eye(1) * 0.01

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

    y_min = min(np.min(xs[:, :2]), self.x_goal[1])
    y_max = max(np.max(xs[:, :2]), self.x_goal[1])
    y_pad = max(y_max - y_min, 1e-5) * 0.05
    y_range = [y_min - y_pad, y_max + y_pad]

    fig = go.Figure()
    plotly_util.add_line(
        fig=fig,
        x=ts[:k+1],
        y=xs[:k+1, 0],
        name='Pusher',
        color='red',
    )
    plotly_util.add_line(
        fig=fig,
        x=ts[:k+1],
        y=xs[:k+1, 1],
        name='Object',
        color='blue',
    )
    plotly_util.add_line(
        fig=fig,
        x=t_range,
        y=[self.x_goal[1]] * 2,
        name='Object goal',
        color='black',
        linestyle='dash',
    )
    fig.update_layout(
        xaxis_title='Time (sec)',
        yaxis_title='Position (m)',
        xaxis={'range': t_range},
        yaxis={'range': y_range},
    )
    if not hasattr(self, 'plot'):
      server.scene.add_grid('grid')
      self.plot = server.gui.add_plotly(fig)
    else:
      self.plot.figure = fig


if __name__ == '__main__':
  task = PushSphere1d()
  task.print_summary()
