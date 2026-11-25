import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujodico.examples.tasks import util_uplot
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks.task_playback import TaskPlayback


_XML = """
<mujoco model="quadrotor_pole">
  <option timestep="0.02" integrator="implicitfast" gravity="0 0 -9.81" />
  <default>
    <geom contype="0" conaffinity="0" />
    <motor ctrllimited="true" ctrlrange="0 5" />
  </default>
  <worldbody>
    <light pos="0 0 4" />
    <geom name="ground" type="plane" pos="0 0 -0.55" size="0 0 1"
          rgba="0.8 0.8 0.8 1" />
    <site name="goal" pos="0 0 1" size="0.04" rgba="0 0.8 0 0.5" />
    <body name="quadrotor" pos="0 0 1">
      <freejoint name="root" />
      <inertial pos="0 0 0" mass="1" diaginertia="0.02 0.02 0.04" />
      <geom name="body" type="box" size="0.08 0.06 0.025"
            rgba="0.2 0.2 0.2 1" />
      <geom type="capsule" fromto="-0.2 0 0 0.2 0 0" size="0.012"
            rgba="0.3 0.3 0.3 1" />
      <geom type="capsule" fromto="0 -0.2 0 0 0.2 0" size="0.012"
            rgba="0.3 0.3 0.3 1" />
      <geom type="cylinder" pos="0.2 0 0.01" size="0.07 0.005"
            rgba="0.8 0.1 0.1 0.7" />
      <geom type="cylinder" pos="0 0.2 0.01" size="0.07 0.005"
            rgba="0.1 0.3 0.8 0.7" />
      <geom type="cylinder" pos="-0.2 0 0.01" size="0.07 0.005"
            rgba="0.1 0.3 0.8 0.7" />
      <geom type="cylinder" pos="0 -0.2 0.01" size="0.07 0.005"
            rgba="0.1 0.3 0.8 0.7" />
      <site name="rotor_front" pos="0.2 0 0" size="0.01" />
      <site name="rotor_left" pos="0 0.2 0" size="0.01" />
      <site name="rotor_back" pos="-0.2 0 0" size="0.01" />
      <site name="rotor_right" pos="0 -0.2 0" size="0.01" />
      <body name="pole">
        <joint name="pole_joint" type="ball" damping="0.003" />
        <geom name="pole" type="capsule" fromto="0 0 0 0 0 0.6"
              size="0.02" mass="0.15" rgba="0.9 0.4 0.05 1" />
        <site name="pole_tip" pos="0 0 0.6" size="0.025"
              rgba="0.9 0.4 0.05 1" />
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- Thrust acts along body +z; alternating reaction torques control yaw. -->
    <motor name="front" site="rotor_front" gear="0 0 1 0 0 0.02" />
    <motor name="left" site="rotor_left" gear="0 0 1 0 0 -0.02" />
    <motor name="back" site="rotor_back" gear="0 0 1 0 0 0.02" />
    <motor name="right" site="rotor_right" gear="0 0 1 0 0 -0.02" />
  </actuator>
</mujoco>
"""


class QuadrotorPole(Task):
  """Fly to the target while swinging a passive pole upright."""

  def __init__(self, seed=0):
    model = mujoco.MjModel.from_xml_string(_XML)
    rng = np.random.default_rng(seed)
    pole_qpos = model.joint('pole_joint').qposadr[0]

    goal = np.r_[model.qpos0, np.zeros(model.nv)]
    goal[:3] = model.site('goal').pos
    x0 = goal.copy()
    x0[:3] = rng.uniform([-5.0, -5.0, 0.1], [5.0, 5.0, 0.1])
    angles = rng.uniform([-0.1, -0.1, -0.2], [0.1, 0.1, 0.2])
    x0[3:7] = np.roll(Rotation.from_euler('xyz', angles).as_quat(), 1)
    tilt = np.pi - rng.uniform(0.05, 0.15)
    x0[pole_qpos : pole_qpos + 4] = [np.cos(tilt / 2), 0, np.sin(tilt / 2), 0]

    # The pole can take any twist angle about its long axis at the goal.
    position_weights = [3.0, 3.0, 10.0] + [1.0] * 3 + [0.5, 0.5, 0.0]
    velocity_weights = [0.2] * 3 + [0.1] * 3 + [0.025] * 3
    Q = np.diag(position_weights + velocity_weights)
    hover_thrust = model.body_mass.sum() * abs(model.opt.gravity[2]) / model.nu

    super().__init__(
        model=model,
        horizon=200,
        x_0=x0,
        x_goal=goal,
        Q=Q,
        Qf=100 * Q,
        u_ref=np.full(model.nu, hover_thrust),
        R=0.025 * np.eye(model.nu),
    )

  @property
  def x_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = super().x_bounds
    # Leave room for the hanging pole above the visual ground while keeping
    # the initial state strictly inside the altitude bound.
    lower[2] = 0.08
    return lower, upper

  def update_visualization(self, visualizer: TaskPlayback, traj: dict, k: int):
    super().update_visualization(visualizer, traj, k)

    xs = np.asarray(traj['xs'])
    ts = np.arange(len(xs)) * self.model.opt.timestep
    pole_qpos = self.model.joint('pole_joint').qposadr[0]
    pole_rotation = Rotation.from_quat(
        np.roll(xs[:, pole_qpos : pole_qpos + 4], -1, axis=1)
    )
    attitude = Rotation.from_quat(np.roll(xs[:, 3:7], -1, axis=1))
    world_z = (attitude * pole_rotation).apply([0, 0, 1])[:, 2]
    inclination = np.rad2deg(np.arccos(np.clip(world_z, -1.0, 1.0)))

    state_panels = tuple(
        util_uplot.time_series_panel(
            ts,
            xs[:, i],
            k + 1,
            ylabel=f'{axis} Position (m)',
            references=(util_uplot.ReferenceLine(self.x_goal[i]),),
            xlabel='',
        )
        for i, axis in enumerate('XYZ')
    )
    pole_panel = util_uplot.time_series_panel(
        ts,
        inclination,
        k + 1,
        ylabel='Pole angle from upright (deg)',
        references=(util_uplot.ReferenceLine(0.0, 'Upright'),),
    )
    visualizer.set_plot(
        'trajectory', util_uplot.UplotFigure((*state_panels, pole_panel))
    )

    if 'us' in traj:
      lower, upper = self.u_bounds
      thrust_panel = util_uplot.time_series_panel(
          ts,
          traj['us'],
          k,
          ylabel='Rotor thrust (N)',
          labels=('Front', 'Left', 'Back', 'Right'),
          colors=('red', 'blue', 'green', 'orange'),
          references=(
              util_uplot.ReferenceLine(self.u_ref[0], 'Hover'),
              util_uplot.ReferenceLine(lower[0], 'Lower bound'),
              util_uplot.ReferenceLine(upper[0], 'Upper bound'),
          ),
      )
      visualizer.set_plot('controls', util_uplot.UplotFigure((thrust_panel,)))


if __name__ == '__main__':
  QuadrotorPole().print_summary()
