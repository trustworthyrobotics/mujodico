import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks.task_playback import TaskPlayback
from mujodico.examples.tasks import util_uplot
import numpy as np


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

  def update_visualization(self, visualizer: TaskPlayback, traj: dict, k: int):
    super().update_visualization(visualizer, traj, k)

    xs = traj['xs']
    ts = np.arange(xs.shape[0]) * self.model.opt.timestep

    visualizer.set_plot(
        'trajectory',
        util_uplot.UplotFigure((
            util_uplot.time_series_panel(
                ts,
                xs[:, :2],
                k + 1,
                ylabel='Position (m)',
                labels=('Pusher', 'Object'),
                colors=('red', 'blue'),
                references=(
                    util_uplot.ReferenceLine(self.x_goal[1], 'Object goal'),
                ),
            ),
        )),
    )


if __name__ == '__main__':
  PushSphere1d().print_summary()
