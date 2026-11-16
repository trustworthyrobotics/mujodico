import mujoco
import numpy as np

from mujodico.examples.tasks import util_uplot
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks.task_playback import TaskPlayback


_XML = """
<mujoco model="double_cart_pole">
  <compiler angle="radian" />
  <option timestep="0.02" integrator="implicitfast" gravity="0 0 -9.81" />
  <default>
    <geom contype="0" conaffinity="0" />
    <joint limited="false" damping="0.01" />
  </default>
  <worldbody>
    <light pos="0 -3 5" />
    <geom name="ground" type="plane" size="4 2 0.1"
          rgba="0.8 0.8 0.8 1" />
    <geom name="rail" type="capsule" fromto="-2.6 0 1.05 2.6 0 1.05"
          size="0.025" rgba="0.3 0.3 0.3 1" />
    <geom type="capsule" fromto="-2.5 0 0 -2.5 0 1.05"
          size="0.04" rgba="0.3 0.3 0.3 1" />
    <geom type="capsule" fromto="2.5 0 0 2.5 0 1.05"
          size="0.04" rgba="0.3 0.3 0.3 1" />
    <body name="cart" pos="0 0 1.2">
      <!-- Travel is constrained by the optimizer, without rail impacts. -->
      <joint name="cart_slide" type="slide" axis="1 0 0"
             range="-2.4 2.4" damping="0.1" />
      <geom name="cart" type="box" size="0.2 0.12 0.075" mass="1"
            rgba="0.25 0.25 0.3 1" />
      <body name="pole_1" pos="0 0 0.1">
        <joint name="pole_1" type="hinge" axis="0 1 0" />
        <geom name="pole_1" type="capsule" fromto="0 0 0 0 0 0.6"
              size="0.025" mass="0.2" rgba="0.9 0.4 0.05 1" />
        <body name="pole_2" pos="0 0 0.6">
          <joint name="pole_2" type="hinge" axis="0 1 0" />
          <geom name="pole_2" type="capsule" fromto="0 0 0 0 0 0.6"
                size="0.025" mass="0.2" rgba="0.1 0.35 0.85 1" />
          <site name="tip" pos="0 0 0.6" size="0.04" />
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="cart_force" joint="cart_slide" gear="1"
           ctrllimited="true" ctrlrange="-20 20" />
  </actuator>
</mujoco>
"""


class DoubleCartPole(Task):

  def __init__(self, seed=0):
    model = mujoco.MjModel.from_xml_string(_XML)
    rng = np.random.default_rng(seed)
    x0 = np.r_[
        rng.uniform([-0.2, np.pi - 0.1, -0.1], [0.2, np.pi + 0.1, 0.1]),
        np.zeros(model.nv),
    ]
    Q = np.diag([1.0, 5.0, 5.0, 0.1, 0.1, 0.1])

    super().__init__(
        model=model,
        horizon=250,
        x_0=x0,
        x_goal=np.zeros(model.nq + model.nv),
        Q=Q,
        Qf=100 * Q,
        u_ref=np.zeros(model.nu),
        R=np.eye(model.nu) * 0.005,
        contact_smoothing=0.0,
    )

  @property
  def x_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = super().x_bounds
    lower[0], upper[0] = self.model.joint('cart_slide').range
    return lower, upper

  @property
  def xs_init(self) -> np.ndarray:
    # Seed a cart excursion and elbow bend before settling upright. Include
    # the corresponding velocities so the guess follows the link kinematics.
    duration = 3.0
    t = np.minimum(
        np.arange(self.horizon + 1) * self.model.opt.timestep / duration, 1.0
    )
    progress = 3 * t**2 - 2 * t**3
    progress_rate = (6 * t - 6 * t**2) / duration
    bend = np.sin(np.pi * t) ** 2
    bend_rate = np.pi * np.sin(2 * np.pi * t) / duration
    bend[t == 1.0] = 0.0
    bend_rate[t == 1.0] = 0.0
    excursion = np.array([-1.5, 0.0, 1.5])
    delta = self.x_goal[: self.model.nq] - self.x_0[: self.model.nq]
    positions = (
        self.x_0[: self.model.nq]
        + progress[:, None] * delta
        + bend[:, None] * excursion
    )
    velocities = progress_rate[:, None] * delta + bend_rate[:, None] * excursion
    return np.column_stack([positions, velocities])

  def update_visualization(self, visualizer: TaskPlayback, traj: dict, k: int):
    super().update_visualization(visualizer, traj, k)

    xs = np.asarray(traj['xs'])
    ts = np.arange(len(xs)) * self.model.opt.timestep
    lower, upper = self.x_bounds
    pole_angles = np.rad2deg(np.column_stack([xs[:, 1], xs[:, 1] + xs[:, 2]]))
    visualizer.set_plot(
        'trajectory',
        util_uplot.UplotFigure((
            util_uplot.time_series_panel(
                ts,
                xs[:, 0],
                k + 1,
                ylabel='Cart position (m)',
                references=(
                    util_uplot.ReferenceLine(self.x_goal[0]),
                    util_uplot.ReferenceLine(
                        lower[0], 'Lower bound', 'red', (2, 3)
                    ),
                    util_uplot.ReferenceLine(
                        upper[0], 'Upper bound', 'red', (2, 3)
                    ),
                ),
                xlabel='',
            ),
            util_uplot.time_series_panel(
                ts,
                pole_angles,
                k + 1,
                ylabel='Angle from upright (deg)',
                labels=('First pole', 'Second pole'),
                colors=('orange', 'blue'),
                references=(util_uplot.ReferenceLine(0.0, 'Upright'),),
                xlabel='',
            ),
            util_uplot.time_series_panel(
                ts,
                xs[:, self.model.nq],
                k + 1,
                ylabel='Cart velocity (m/s)',
                references=(
                    util_uplot.ReferenceLine(self.x_goal[self.model.nq]),
                ),
            ),
        )),
    )

    if 'us' in traj:
      us = np.asarray(traj['us'])
      lower, upper = self.u_bounds
      visualizer.set_plot(
          'controls',
          util_uplot.UplotFigure((
              util_uplot.time_series_panel(
                  ts,
                  us[:, 0],
                  k,
                  ylabel='Cart force (N)',
                  references=(
                      util_uplot.ReferenceLine(self.u_ref[0], 'Reference'),
                      util_uplot.ReferenceLine(
                          lower[0], 'Lower bound', 'red', (2, 3)
                      ),
                      util_uplot.ReferenceLine(
                          upper[0], 'Upper bound', 'red', (2, 3)
                      ),
                  ),
              ),
          )),
      )


if __name__ == '__main__':
  DoubleCartPole().print_summary()
