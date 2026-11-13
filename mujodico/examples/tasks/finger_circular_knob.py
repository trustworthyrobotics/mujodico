import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks.task_playback import TaskPlayback
from mujodico.examples.tasks import util_uplot
import numpy as np


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


if __name__ == '__main__':
  FingerCircularKnob().print_summary()
