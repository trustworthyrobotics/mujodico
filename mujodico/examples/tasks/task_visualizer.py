import mujoco
import numpy as np

try:
  import mjviser
  import viser
  import plotly.graph_objects as go
except ImportError as e:
  raise ImportError(
      'Optional visualization dependencies are not installed. '
      'Install them with:\n\n'
      '    pip install mjviser plotly'
  ) from e


class TaskVisualizer:

  def __init__(self, server: viser.ViserServer, model: mujoco.MjModel):
    self._server = server
    server.scene.add_grid('/grid')

    self._model = model
    self._data = mujoco.MjData(model)

    self._mj_scene = mjviser.ViserMujocoScene(server, model, num_envs=1)
    self._mj_scene.camera_tracking_enabled = False

    self._folder = server.gui.add_folder('Task', expand_by_default=True)
    with self._folder:
      self._info_panel = server.gui.add_markdown('')
    self._plots: dict[str, viser.GuiPlotlyHandle] = {}

    self._cleaned_up = False

  def __del__(self):
    try:
      self.cleanup()
    except Exception:
      pass

  def update_scene(self, qpos: np.ndarray) -> None:
    self._data.qpos = qpos
    mujoco.mj_kinematics(self._model, self._data)
    self._mj_scene.update_from_mjdata(self._data)

  def set_text(self, text: str) -> None:
    self._info_panel.content = text

  def set_plot(self, name: str, figure: go.Figure) -> None:
    figure.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    if name not in self._plots:
      with self._folder:
        self._plots[name] = self._server.gui.add_plotly(figure, aspect=3/4)
    else:
      self._plots[name].figure = figure

  def cleanup(self) -> None:
    if self._cleaned_up:
      return
    self._folder.remove()
    self._server.scene.reset()
    self._plots.clear()
    self._cleaned_up = True
