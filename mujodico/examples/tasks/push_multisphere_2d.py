import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks.task_playback import TaskPlayback
from mujodico.examples.tasks import util_plotly
import numpy as np


_XML = """
<mujoco model="push_multisphere_2d">

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

    <body name="distractor" pos="0 0 0">
      <joint name="distractor_slide_x" type="slide" axis="1 0 0" />
      <joint name="distractor_slide_y" type="slide" axis="0 1 0" />

      <geom name="distractor_geom"
            type="sphere"
            size="0.05"
            mass="1"
            rgba="0 1 0 1" />
    </body>

  </worldbody>

  <contact>
    <exclude body1="object" body2="distractor"/>
  </contact>

  <actuator>
    <velocity name="pusher_vel_x" joint="pusher_slide_x" kv="100" ctrlrange="-1 1" />
    <velocity name="pusher_vel_y" joint="pusher_slide_y" kv="100" ctrlrange="-1 1" />
  </actuator>

</mujoco>
"""


class PushMultiSphere2d(Task):

  def __init__(self, seed=0):
    model = mujoco.MjModel.from_xml_string(_XML)
    horizon = 100

    x_0 = np.array([-0.1, -0.15, 0.1, 0.15, 0.5, -0.8] + [0.0] * 6)
    x_goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 6)
    Q = np.diag([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0])
    Qf = Q * 100.0

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

  def update_visualization(self, visualizer: TaskPlayback, traj: dict, k: int):
    super().update_visualization(visualizer, traj, k)

    xs = traj['xs']

    fig = util_plotly.new_fig()
    util_plotly.add_line(
        fig=fig,
        x=xs[:k+1, 0],
        y=xs[:k+1, 1],
        name='Pusher',
        color='red',
    )
    util_plotly.add_circle(
        fig=fig,
        x=xs[k, 0],
        y=xs[k, 1],
        r=0.05,
        color='red',
    )
    util_plotly.add_line(
        fig=fig,
        x=xs[:k+1, 2],
        y=xs[:k+1, 3],
        name='Object',
        color='blue',
    )
    util_plotly.add_circle(
        fig=fig,
        x=xs[k, 2],
        y=xs[k, 3],
        r=0.08,
        color='blue',
    )
    util_plotly.add_circle(
        fig=fig,
        x=xs[k, 4],
        y=xs[k, 5],
        r=0.05,
        color='rgb(100,255,100)',
    )
    util_plotly.add_circle(
        fig=fig,
        x=self.x_goal[2],
        y=self.x_goal[3],
        r=0.08,
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
    visualizer.set_plot('visualization', fig)


if __name__ == '__main__':
  PushMultiSphere2d().print_summary()
