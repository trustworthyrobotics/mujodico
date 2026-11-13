"""Point-to-point quadrotor flight with four individual rotor thrusts."""

import mujoco
import numpy as np
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

from mujodico.examples.tasks import plotly_util
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks.task_playback import TaskPlayback


_XML = """
<mujoco model="quadrotor">
  <option timestep="0.02" integrator="implicitfast" gravity="0 0 -9.81" />
  <default>
    <geom contype="0" conaffinity="0" />
    <motor ctrllimited="true" ctrlrange="0 5" />
  </default>
  <worldbody>
    <light pos="0 0 4" />
    <geom name="ground" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1" />
    <site name="goal" pos="0.0 0.0 1.0" size="0.04" rgba="0 0.8 0 0.5" />
    <body name="quadrotor" pos="0 0 1">
      <freejoint name="root" />
      <inertial pos="0 0 0" mass="1" diaginertia="0.02 0.02 0.04" />
      <geom name="body" type="box" size="0.08 0.06 0.025" rgba="0.2 0.2 0.2 1" />
      <geom type="capsule" fromto="-0.2 0 0 0.2 0 0" size="0.012" rgba="0.3 0.3 0.3 1" />
      <geom type="capsule" fromto="0 -0.2 0 0 0.2 0" size="0.012" rgba="0.3 0.3 0.3 1" />
      <geom type="cylinder" pos="0.2 0 0.01" size="0.07 0.005" rgba="0.8 0.1 0.1 0.7" />
      <geom type="cylinder" pos="0 0.2 0.01" size="0.07 0.005" rgba="0.1 0.3 0.8 0.7" />
      <geom type="cylinder" pos="-0.2 0 0.01" size="0.07 0.005" rgba="0.1 0.3 0.8 0.7" />
      <geom type="cylinder" pos="0 -0.2 0.01" size="0.07 0.005" rgba="0.1 0.3 0.8 0.7" />
      <site name="rotor_front" pos="0.2 0 0" size="0.01" />
      <site name="rotor_left" pos="0 0.2 0" size="0.01" />
      <site name="rotor_back" pos="-0.2 0 0" size="0.01" />
      <site name="rotor_right" pos="0 -0.2 0" size="0.01" />
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


class Quadrotor(Task):

  def __init__(self, seed=0):
    model = mujoco.MjModel.from_xml_string(_XML)

    rng = np.random.default_rng(seed)

    x0 = np.r_[model.qpos0, np.zeros(model.nv)]
    x0[:3] = rng.uniform([-5.0, -5.0, 0.1], [5.0, 5.0, 0.1])
    angles = rng.uniform([-0.1, -0.1, -0.2], [0.1, 0.1, 0.2])
    x0[3:7] = np.roll(Rotation.from_euler("xyz", angles).as_quat(), 1)

    goal = np.r_[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, np.zeros(6)]

    Q = np.diag([10.0, 10.0, 20.0] + [2.0] * 4 + [0.5] * 3 + [1.0] * 3)
    hover_thrust = model.body_mass.sum() * abs(model.opt.gravity[2]) / model.nu

    super().__init__(
        model=model,
        horizon=100,
        x_0=x0,
        x_goal=goal,
        Q=Q,
        Qf=20 * Q,
        u_ref=np.full(4, hover_thrust),
        R=np.eye(4),
        contact_smoothing=0.0,
    )

  def update_visualization(self, visualizer: TaskPlayback, traj: dict, k: int):
    super().update_visualization(visualizer, traj, k)
    xs = np.asarray(traj["xs"])
    ts = np.arange(len(xs)) * self.model.opt.timestep
    rotations = Rotation.from_quat(np.roll(xs[:, 3:7], -1, axis=1))
    goal_rotation = Rotation.from_quat(np.roll(self.x_goal[3:7], -1))
    attitude_error = np.rad2deg(
        np.linalg.norm((goal_rotation.inv() * rotations).as_rotvec(), axis=1)
    )
    values = np.column_stack([xs[:, :3], attitude_error])
    goals = np.r_[self.x_goal[:3], 0.0]
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02
    )
    for row, label in enumerate(
        ["X Position (m)", "Y Position (m)", "Z Position (m)", "Attitude error (deg)"],
        start=1,
    ):
      data, goal = values[:, row - 1], goals[row - 1]
      lower, upper = min(data.min(), goal), max(data.max(), goal)
      padding = max(upper - lower, 1e-5) * 0.05
      plotly_util.add_line(
          fig, ts[: k + 1], data[: k + 1], color="blue", row=row, col=1
      )
      plotly_util.add_line(
          fig,
          [ts[0], ts[-1]],
          [goal, goal],
          color="black",
          linestyle="dash",
          row=row,
          col=1,
      )
      fig.update_xaxes(
          range=[ts[0], ts[-1]],
          title_text="Time (sec)" if row == 4 else None,
          row=row,
          col=1,
      )
      fig.update_yaxes(
          title_text=label,
          range=[lower - padding, upper + padding],
          row=row,
          col=1,
      )
    visualizer.set_plot("trajectory", fig)

    if "us" in traj:
      us = np.asarray(traj["us"])
      fig = make_subplots(
          rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02
      )
      lower, upper = self.u_bounds
      for i, name in enumerate(("Front", "Left", "Back", "Right")):
        plotly_util.add_line(
            fig, ts[:k], us[:k, i], color="blue", row=i + 1, col=1
        )
        for value, color, style in (
            (self.u_ref[i], "black", "dash"),
            (lower[i], "red", "dot"),
            (upper[i], "red", "dot"),
        ):
          plotly_util.add_line(
              fig,
              [ts[0], ts[-1]],
              [value, value],
              color=color,
              linestyle=style,
              row=i + 1,
              col=1,
          )
        fig.update_xaxes(
            range=[ts[0], ts[-1]],
            title_text="Time (sec)" if i == 3 else None,
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(title_text=f"{name} thrust (N)", row=i + 1, col=1)
      visualizer.set_plot("controls", fig)


if __name__ == "__main__":
  Quadrotor().print_summary()
