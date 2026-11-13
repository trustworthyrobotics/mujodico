import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks.task_playback import TaskPlayback
from mujodico.examples.tasks import util_uplot
import numpy as np


_XML = """
<mujoco model="fingers_sphere_2d">

  <option timestep="0.02"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <default>
    <geom condim="3" friction="0.2" />
  </default>

  <worldbody>

    <body name="left_link1" pos="-0.15 0 0">
      <joint name="left_joint1" type="hinge" range="-90 90" />
      <geom type="capsule" size="0.03" fromto="0 0 0 0 0.15 0" />

      <body name="left_link2" pos="0 0.15 0">
        <joint name="left_joint2" type="hinge" range="-100 100" />
        <geom type="capsule" size="0.03" fromto="0 0.06 0 0 0.15 0" />
      </body>
    </body>

    <body name="right_link1" pos="0.15 0 0">
      <joint name="right_joint1" type="hinge" range="-90 90" />
      <geom type="capsule" size="0.03" fromto="0 0 0 0 0.15 0" />

      <body name="right_link2" pos="0 0.15 0">
        <joint name="right_joint2" type="hinge" range="-100 100" />
        <geom type="capsule" size="0.03" fromto="0 0.06 0 0 0.15 0" />
      </body>
    </body>

    <body name="object" pos="0 0 0">
      <joint name="object_translate_x" type="slide" axis="1 0 0" damping="1" range="-0.5 0.5" />
      <joint name="object_translate_y" type="slide" axis="0 1 0" damping="1" range="-0.5 0.5" />
      <joint name="object_rotate" type="hinge" damping="0.025" />

      <geom type="sphere" size="0.08" rgba="0 0 1 1" />
      <geom name="angle_indicator" type="sphere" size="0.012" pos="0.05 0 0.05" rgba="0 0 0 1"
            contype="0" conaffinity="0" />
    </body>

  </worldbody>

  <actuator>
    <velocity name="left_joint1_vel" joint="left_joint1" kv="1" ctrlrange="-1 1" />
    <velocity name="left_joint2_vel" joint="left_joint2" kv="1" ctrlrange="-1 1" />
    <velocity name="right_joint1_vel" joint="right_joint1" kv="1" ctrlrange="-1 1" />
    <velocity name="right_joint2_vel" joint="right_joint2" kv="1" ctrlrange="-1 1" />
  </actuator>

</mujoco>
"""


class FingersSphere2d(Task):

  def __init__(self, seed=0):
    model = mujoco.MjModel.from_xml_string(_XML)
    horizon = 100

    x_0 = np.concatenate((
        np.deg2rad([30, -60, -30, 60]),
        [0.0, 0.25, 0.0],
        [0.0] * 7,
    ))
    x_goal = np.concatenate((
        [0.0] * 4,
        [0.0, 0.25, np.pi/2],
        [0.0] * 7,
    ))
    Q = np.diag(np.concatenate((
        [0.0] * 4,
        [5.0, 5.0, 0.1],
        [0.0] * 4,
        [0.01, 0.01, 0.001],
    )))
    Qf = Q * 100

    u_ref = np.array([0.0] * 4)
    R = np.eye(4) * 0.001

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

    labels = ('X Position (m)', 'Y Position (m)', 'Angle (deg)')
    panels = []
    for object_dof, label in enumerate(labels):
      dof = 4 + object_dof
      values = xs[:, dof]
      goal = self.x_goal[dof]
      if object_dof == 2:
        values = np.rad2deg(values)
        goal = np.rad2deg(goal)
      panels.append(
          util_uplot.time_series_panel(
              ts,
              values,
              k + 1,
              ylabel=label,
              references=(util_uplot.ReferenceLine(goal),),
              xlabel='Time (sec)' if object_dof == 2 else '',
          )
      )
    visualizer.set_plot('trajectory', util_uplot.UplotFigure(tuple(panels)))


if __name__ == '__main__':
  FingersSphere2d().print_summary()
