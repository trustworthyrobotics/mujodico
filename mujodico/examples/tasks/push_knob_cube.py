import mujoco
from mujodico.examples.tasks import util_uplot
from mujodico.examples.tasks.task import Task, quat_error
from mujodico.examples.tasks.task_playback import TaskPlayback
import numpy as np
from scipy.spatial.transform import Rotation


_XML = """
<mujoco model="push_knob_cube">

  <option timestep="0.02" gravity="0 0 0" />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <default>
    <geom condim="3" />
  </default>

  <worldbody>

    <body name="pusher" pos="0 0 0">
      <joint name="pusher_slide_x" type="slide" axis="1 0 0" range="-0.5 0.5" />
      <joint name="pusher_slide_y" type="slide" axis="0 1 0" range="-0.5 0.5" />
      <joint name="pusher_slide_z" type="slide" axis="0 0 1" range="-0.5 0.5" />

      <geom name="pusher_geom"
            type="sphere"
            size="0.02"
            rgba="1 0 1 1" />
    </body>

    <body name="knob" pos="0 0 0">
      <joint name="knob_rotate" type="ball" damping="0.05" />

      <geom name="knob_geom"
            type="box"
            size="0.08 0.08 0.08"
            rgba="0 1 1 1" />

      <!-- Distinct markers show all three rotation axes without contact. -->
      <geom name="x_indicator" type="sphere" pos="0.08 0 0" size="0.008"
            rgba="1 0 0 1" mass="0" contype="0" conaffinity="0" />
      <geom name="y_indicator" type="sphere" pos="0 0.08 0" size="0.008"
            rgba="0 1 0 1" mass="0" contype="0" conaffinity="0" />
      <geom name="z_indicator" type="sphere" pos="0 0 0.08" size="0.008"
            rgba="0 0 1 1" mass="0" contype="0" conaffinity="0" />
    </body>

  </worldbody>

  <actuator>
    <velocity name="pusher_vel_x" joint="pusher_slide_x" kv="10" ctrlrange="-0.2 0.2" />
    <velocity name="pusher_vel_y" joint="pusher_slide_y" kv="10" ctrlrange="-0.2 0.2" />
    <velocity name="pusher_vel_z" joint="pusher_slide_z" kv="10" ctrlrange="-0.2 0.2" />
  </actuator>

</mujoco>
"""


class PushKnobCube(Task):

  def __init__(self, seed=0):
    model = mujoco.MjModel.from_xml_string(_XML)

    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, 3)
    quaternion = np.roll(Rotation.from_euler('xyz', angles).as_quat(), 1)

    x_0 = np.concatenate(([-0.2, -0.2, -0.2], quaternion, np.zeros(6)))
    x_goal = np.concatenate((np.zeros(3), [1.0, 0.0, 0.0, 0.0], np.zeros(6)))
    Q = np.diag([0.0] * 3 + [1.0] * 3 + [0.0] * 3 + [0.001] * 3)

    super().__init__(
        model=model,
        horizon=400,
        x_0=x_0,
        x_goal=x_goal,
        Q=Q,
        Qf=Q * 400,
        u_ref=np.zeros(3),
        R=np.eye(3) * 0.01,
        contact_smoothing=0.01,
    )

  def update_visualization(self, visualizer: TaskPlayback, traj: dict, k: int):
    super().update_visualization(visualizer, traj, k)

    xs = np.asarray(traj['xs'])
    ts = np.arange(len(xs)) * self.model.opt.timestep
    attitude_error = np.rad2deg(
        np.linalg.norm(quat_error(self.x_goal[3:7], xs[:, 3:7]), axis=1)
    )
    visualizer.set_plot(
        'trajectory',
        util_uplot.UplotFigure((
            util_uplot.time_series_panel(
                ts,
                attitude_error,
                k + 1,
                ylabel='Attitude error (deg)',
                colors=('black',),
                references=(util_uplot.ReferenceLine(0.0),),
                tick_step=45,
            ),
        )),
    )


if __name__ == '__main__':
  PushKnobCube().print_summary()
