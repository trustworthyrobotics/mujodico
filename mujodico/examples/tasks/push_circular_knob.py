import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks.task_playback import TaskPlayback
from mujodico.examples.tasks import util_plotly
from mujodico.examples.tasks import util_uplot
import numpy as np


_XML = """
<mujoco model="push_circular_knob">

  <option timestep="0.02"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <default>
    <geom condim="3" />
  </default>

  <worldbody>

    <body name="pusher" pos="0 0 0">
      <joint name="pusher_slide_x" type="slide" axis="1 0 0" range="-0.5 0.5" />
      <joint name="pusher_slide_y" type="slide" axis="0 1 0" range="-0.5 0.5" />

      <geom name="pusher_geom"
            type="sphere"
            size="0.02"
            rgba="1 0 0 1" />
    </body>

    <body name="knob" pos="0 0 0">
      <joint name="knob_rotate" type="hinge" damping="0.05" />

      <geom name="knob_geom"
            type="cylinder"
            size="0.08"
            fromto="0 0 -0.05 0 0 0.05"
            rgba="0 0 1 1" />

      <geom name="angle_indicator"
            type="box"
            pos="0.04 0 0"
            size="0.04 0.003 0.05001"
            rgba="0 0 0 1"
            contype="0"
            conaffinity="0" />
    </body>

  </worldbody>

  <actuator>
    <velocity name="pusher_vel_x" joint="pusher_slide_x" kv="10" ctrlrange="-0.2 0.2" />
    <velocity name="pusher_vel_y" joint="pusher_slide_y" kv="10" ctrlrange="-0.2 0.2" />
  </actuator>

</mujoco>
"""


class PushCircularKnob(Task):

  def __init__(self, seed=0):
    model = mujoco.MjModel.from_xml_string(_XML)
    horizon = 400

    rng = np.random.default_rng(seed)

    x_0 = np.concatenate((
        [-0.2, -0.2],
        [-2 * np.pi + rng.uniform(-np.pi / 4, np.pi / 4)],
        [0.0] * 3
    ))
    x_goal = np.array([0.0, 0.0, 0.0] + [0.0] * 3)
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

  def update_visualization(self, visualizer: TaskPlayback, traj: dict, k: int):
    super().update_visualization(visualizer, traj, k)

    xs = traj['xs']
    ts = np.arange(xs.shape[0]) * self.model.opt.timestep
    visualizer.set_plot(
        'trajectory',
        util_uplot.UplotFigure((
            util_uplot.time_series_panel(
                ts,
                np.rad2deg(xs[:, 2]),
                k + 1,
                ylabel='Angle (deg)',
                labels=('Angle',),
                colors=('blue',),
                references=(
                    util_uplot.ReferenceLine(
                        np.rad2deg(self.x_goal[2]), 'Angle goal'
                    ),
                ),
                tick_step=45,
            ),
        )),
    )

    fig = util_plotly.new_fig()
    util_plotly.add_line(
        fig=fig,
        x=xs[: k + 1, 0],
        y=xs[: k + 1, 1],
        name='Pusher',
        color='red',
    )
    util_plotly.add_circle(
        fig=fig,
        x=xs[k, 0],
        y=xs[k, 1],
        r=0.02,
        color='red',
    )
    util_plotly.add_circle(
        fig=fig,
        x=0.0,
        y=0.0,
        r=0.08,
        name='Knob',
        color='blue',
    )
    util_plotly.add_rectangle(
        fig=fig,
        x=0.04 * np.cos(xs[k, 2]),
        y=0.04 * np.sin(xs[k, 2]),
        theta=xs[k, 2],
        w=0.08,
        h=0.003,
        color='black',
    )
    fig.update_layout(
        xaxis_title='x',
        yaxis_title='y',
        xaxis={'range': [-0.5, 0.5]},
        yaxis={'range': [-0.5, 0.5], 'scaleanchor': 'x', 'scaleratio': 1},
    )
    visualizer.set_plot('visualization', fig)


if __name__ == '__main__':
  PushCircularKnob().print_summary()
