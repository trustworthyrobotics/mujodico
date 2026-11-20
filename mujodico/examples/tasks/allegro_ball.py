from pathlib import Path
import tempfile
import mujoco
from mujodico.examples.tasks.task import Task, quat_error
from mujodico.examples.tasks.task_playback import TaskPlayback
from mujodico.examples.tasks import util_uplot
import numpy as np
import viser


_XML = """
<mujoco model="right_allegro_hand scene">
  <custom>
    <text name="contact_approx" data="hunt-crossley"/>
    <numeric name="contact_smoothing" data="0.002"/>
  </custom>

  <option timestep="0.05"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <include file="right_hand.xml"/>

  <worldbody>
    <geom name="grid" type="plane" size="0 0 1"
          contype="0" conaffinity="0" rgba="0.8 0.8 0.8 1" />
    <body name="object" pos="-0.035 0 0.06">
      <freejoint/>
      <geom type="sphere" size="0.05" rgba="0.5 0.7 0.5 1" friction="0.7"/>

      <geom name="x_indicator" type="sphere" pos="0.044 0 0" size="0.008"
            rgba="1 0 0 1" mass="0" contype="0" conaffinity="0" />
      <geom name="y_indicator" type="sphere" pos="0 0.044 0" size="0.008"
            rgba="0 1 0 1" mass="0" contype="0" conaffinity="0" />
      <geom name="z_indicator" type="sphere" pos="0 0 0.044" size="0.008"
            rgba="0 0 1 1" mass="0" contype="0" conaffinity="0" />
    </body>
  </worldbody>

</mujoco>
"""


class AllegroBall(Task):

  def __init__(self, seed=0):
    dir = Path(__file__).resolve().parent
    allegro_xml_path = dir / 'models' / 'wonik_allegro' / 'right_hand.xml'

    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.xml',
        dir=allegro_xml_path.parent,
        delete=False,
    ) as f:
      f.write(_XML)

    scene_path = f.name
    model = mujoco.MjModel.from_xml_path(scene_path)
    Path(scene_path).unlink()
    data = mujoco.MjData(model)

    horizon = 60

    x_0 = np.concatenate((data.qpos, data.qvel))
    x_0[0:4] = np.deg2rad([0, 50, 60, 30])
    x_0[4:8] = np.deg2rad([0, 40, 60, 30])
    x_0[8:12] = np.deg2rad([0, 50, 60, 30])
    x_0[12:16] = np.deg2rad([30, 120, 80, 60])

    rng = np.random.default_rng(seed)
    goal_pos = rng.uniform([-0.03, -0.03, 0.0], [0.03, 0.03, 0.0])
    goal_angle = np.deg2rad(-180)

    x_goal = x_0.copy()
    x_goal[16:19] += goal_pos
    x_goal[19:23] = np.array([np.cos(goal_angle/2), 0.0, 0.0, np.sin(goal_angle/2)])

    Q = np.diag([0.0] * 16 + [100] * 3 + [0.1] * 3 + [0.0] * 22)
    Qf = Q * 40.0

    u_ref = x_0[:16]
    R = np.eye(16) * 0.01


    super().__init__(
        model=model,
        horizon=horizon,
        x_0=x_0,
        x_goal=x_goal,
        Q=Q,
        Qf=Qf,
        u_ref=u_ref,
        R=R,
    )

  def update_visualization(self, visualizer: TaskPlayback, traj: dict, k: int):
    super().update_visualization(visualizer, traj, k)
    xs = np.asarray(traj['xs'])
    ts = np.arange(len(xs)) * self.model.opt.timestep

    attitude_error = np.rad2deg(
        np.linalg.norm(quat_error(self.x_goal[19:23], xs[:, 19:23]), axis=1)
    )

    values = np.column_stack([xs[:, 16:18], attitude_error])
    goals = np.r_[self.x_goal[16:18], 0.0]

    labels = [
        'X Position (m)',
        'Y Position (m)',
        'Attitude error (deg)',
    ]
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


if __name__ == '__main__':
  task = AllegroBall(0)
  task.print_summary()

  import mjviser
  model = task.model
  server = viser.ViserServer()
  scene = mjviser.ViserMujocoScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False

  data = mujoco.MjData(model)
  data.qpos = task.x_0[:model.nq]
  mujoco.mj_kinematics(model, data)
  scene.update_from_mjdata(data)
  server.sleep_forever()
