from pathlib import Path
import tempfile
import mujoco
from mujodico.examples.tasks.task import Task
from mujodico.examples.tasks import plotly_util
import numpy as np
import plotly.graph_objects as go
import viser


_XML = """
<mujoco model="right_allegro_hand scene">

  <option timestep="0.05"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />

  <include file="right_hand.xml"/>

  <worldbody>
    <body name="object" pos="-0.035 0 0.06">
      <freejoint/>
      <geom type="sphere" size="0.05" rgba="0.5 0.7 0.5 1" priority="1" friction="0.7"/>
      <geom name="angle_indicator" type="sphere" size="0.008" pos="0.03 0 0.03" rgba="0 0 0 1"
            contype="0" conaffinity="0" />
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

    horizon = 10

    x_0 = np.concatenate((data.qpos, data.qvel))
    x_0[0:4] = np.deg2rad([0, 50, 60, 30])
    x_0[4:8] = np.deg2rad([0, 40, 60, 30])
    x_0[8:12] = np.deg2rad([0, 50, 60, 30])
    x_0[12:16] = np.deg2rad([30, 120, 80, 60])

    x_goal = x_0.copy()

    Q = np.diag([0.0] * 16 + [1, 1, 1] + [0.1, 0.1, 0.1, 0.1] + [0.0] * 22)
    Qf = Q * 100.0

    u_ref = x_0[:16]
    R = np.eye(16) * 0.01

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
        quaternion_starts=(19,),
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
