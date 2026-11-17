import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujodico.examples.tasks import util_uplot
from mujodico.examples.tasks.task import Task, quat_error
from mujodico.examples.tasks.task_playback import TaskPlayback


_XML = """
<mujoco model="rocket_landing">
  <compiler angle="radian" />
  <option timestep="0.02" integrator="implicitfast" gravity="0 0 -9.81"
          iterations="20" tolerance="1e-5" />
  <default>
    <geom contype="0" conaffinity="0" />
    <motor ctrllimited="true" />
  </default>
  <worldbody>
    <light pos="0 -3 8" />
    <geom name="ground" type="plane" size="5 5 0.1"
          rgba="0.8 0.8 0.8 1" />
    <site name="landing_pad" type="cylinder" pos="0 0 0.002"
          size="0.7 0.002" rgba="0 0.6 0 0.4" />
    <!-- The landing pose places the feet on the visual ground at z = 0.8 m.
         Collisions are disabled for a smooth powered-descent problem. -->
    <body name="rocket" pos="0 0 0.8">
      <freejoint name="root" />
      <inertial pos="0 0 0" mass="2" diaginertia="0.18 0.18 0.02" />
      <geom name="fuselage" type="cylinder"
            size="0.12 0.5" rgba="0.9 0.9 0.95 1" />
      <geom type="capsule" fromto="0 0 0.4 0 0 0.55" size="0.12"
            rgba="0.8 0.15 0.1 1" />
      <geom type="capsule" fromto="0.1 0 -0.35 0.38 0 -0.76"
            size="0.02" rgba="0.3 0.3 0.3 1" />
      <geom type="capsule" fromto="-0.1 0 -0.35 -0.38 0 -0.76"
            size="0.02" rgba="0.3 0.3 0.3 1" />
      <geom type="capsule" fromto="0 0.1 -0.35 0 0.38 -0.76"
            size="0.02" rgba="0.3 0.3 0.3 1" />
      <geom type="capsule" fromto="0 -0.1 -0.35 0 -0.38 -0.76"
            size="0.02" rgba="0.3 0.3 0.3 1" />
      <geom name="foot_front" type="sphere"
            pos="0.38 0 -0.76" size="0.04" rgba="0.2 0.2 0.2 1" />
      <geom name="foot_back" type="sphere"
            pos="-0.38 0 -0.76" size="0.04" rgba="0.2 0.2 0.2 1" />
      <geom name="foot_left" type="sphere"
            pos="0 0.38 -0.76" size="0.04" rgba="0.2 0.2 0.2 1" />
      <geom name="foot_right" type="sphere"
            pos="0 -0.38 -0.76" size="0.04" rgba="0.2 0.2 0.2 1" />
      <body name="gimbal_x" pos="0 0 -0.58">
        <joint name="gimbal_x" type="hinge" axis="1 0 0"
               limited="false" damping="0.02" />
        <inertial pos="0 0 0" mass="0.01" diaginertia="0.0002 0.0002 0.0002" />
        <body name="engine">
          <joint name="gimbal_y" type="hinge" axis="0 1 0"
                 limited="false" damping="0.02" />
          <inertial pos="0 0 0" mass="0.04" diaginertia="0.0002 0.0002 0.0001" />
          <geom type="cylinder" size="0.07 0.08" rgba="0.2 0.2 0.2 1" />
          <site name="engine" pos="0 0 -0.08" size="0.01" />
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- Thrust follows engine +z. Gimbal servos vector it about two axes. -->
    <motor name="thrust" site="engine" gear="0 0 1 0 0 0" ctrlrange="5 40" />
    <position name="gimbal_x" joint="gimbal_x" kp="5" kv="0.2"
              ctrlrange="-0.3490658503988659 0.3490658503988659" />
    <position name="gimbal_y" joint="gimbal_y" kp="5" kv="0.2"
              ctrlrange="-0.3490658503988659 0.3490658503988659" />
  </actuator>
</mujoco>
"""


class RocketLanding(Task):

  def __init__(self, seed=0):
    model = mujoco.MjModel.from_xml_string(_XML)

    hover_thrust = model.body_mass.sum() * abs(model.opt.gravity[2])

    thrust_min, thrust_max = model.actuator_ctrlrange[0]
    if not thrust_min <= hover_thrust <= thrust_max:
      raise ValueError(
          f'Thrust limits must include the hover thrust ({hover_thrust:.4f} N).'
      )

    rng = np.random.default_rng(seed)

    x0 = np.r_[model.qpos0, np.zeros(model.nv)]
    x0[:3] = rng.uniform([-0.8, -0.8, 4.0], [0.8, 0.8, 6.0])
    # A two-axis gimbal has no direct authority about the rocket's long axis.
    angles = rng.uniform([-0.15, -0.15, 0.0], [0.15, 0.15, 0.0])
    x0[3:7] = np.roll(Rotation.from_euler('xyz', angles).as_quat(), 1)
    x0[model.nq : model.nq + 3] = rng.uniform(
        [-0.2, -0.2, -1.5], [0.2, 0.2, -0.5]
    )

    goal = np.r_[model.qpos0, np.zeros(model.nv)]
    Q = np.diag(
        [5.0, 5.0, 10.0]
        + [5.0] * 3
        + [0.05] * 2
        + [1.0] * 3
        + [0.5] * 3
        + [0.001] * 2
    )

    super().__init__(
        model=model,
        horizon=200,
        x_0=x0,
        x_goal=goal,
        Q=Q,
        Qf=50 * Q,
        u_ref=np.array([hover_thrust, 0.0, 0.0]),
        R=np.diag([0.002, 0.05, 0.05]),
        contact_smoothing=0.0,
    )

  @property
  def x_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = super().x_bounds
    lower[2] = self.x_goal[2]
    # Keep joint limits out of the dynamics; enforce gimbal travel as bounds.
    control_lower, control_upper = self.u_bounds
    lower[7:9], upper[7:9] = control_lower[1:], control_upper[1:]
    return lower, upper

  def update_visualization(self, visualizer: TaskPlayback, traj: dict, k: int):
    super().update_visualization(visualizer, traj, k)

    xs = np.asarray(traj['xs'])
    ts = np.arange(len(xs)) * self.model.opt.timestep
    attitude_error = np.rad2deg(
        np.linalg.norm(quat_error(self.x_goal[3:7], xs[:, 3:7]), axis=1)
    )
    values = np.column_stack(
        [xs[:, :3], xs[:, self.model.nq + 2], attitude_error]
    )
    goals = np.r_[self.x_goal[:3], 0.0, 0.0]
    labels = (
        'X Position (m)',
        'Y Position (m)',
        'Z Position (m)',
        'Vertical velocity (m/s)',
        'Attitude error (deg)',
    )
    visualizer.set_plot(
        'trajectory',
        util_uplot.UplotFigure(
            tuple(
                util_uplot.time_series_panel(
                    ts,
                    values[:, i],
                    k + 1,
                    ylabel=label,
                    references=(util_uplot.ReferenceLine(goals[i]),),
                    xlabel='Time (sec)' if i == len(labels) - 1 else '',
                )
                for i, label in enumerate(labels)
            )
        ),
    )

    if 'us' in traj:
      us = np.asarray(traj['us'])
      lower, upper = self.u_bounds
      labels = ('Thrust (N)', 'Gimbal X target (deg)', 'Gimbal Y target (deg)')
      scales = np.array([1.0, 180.0 / np.pi, 180.0 / np.pi])
      visualizer.set_plot(
          'controls',
          util_uplot.UplotFigure(
              tuple(
                  util_uplot.time_series_panel(
                      ts,
                      us[:, i] * scales[i],
                      k,
                      ylabel=label,
                      references=(
                          util_uplot.ReferenceLine(
                              self.u_ref[i] * scales[i], 'Reference'
                          ),
                          util_uplot.ReferenceLine(
                              lower[i] * scales[i], 'Lower bound', 'red', (2, 3)
                          ),
                          util_uplot.ReferenceLine(
                              upper[i] * scales[i], 'Upper bound', 'red', (2, 3)
                          ),
                      ),
                      xlabel='Time (sec)' if i == len(labels) - 1 else '',
                  )
                  for i, label in enumerate(labels)
              )
          ),
      )


if __name__ == '__main__':
  RocketLanding().print_summary()
