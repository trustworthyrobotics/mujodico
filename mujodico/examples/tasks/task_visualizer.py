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
    self._on_select_iteration = None
    with self._folder:
      self._iteration = server.gui.add_number(
          'Iteration',
          initial_value=-1,
          min=-1,
          max=-1,
          step=1,
          disabled=True,
          hint='-1 plays the final result; 0 and above select recorded '
          'iterations by their zero-based position in history.',
      )
      self._info_panel = server.gui.add_markdown('')

    @self._iteration.on_update
    def _select_iteration(event):
      # Programmatic updates synchronize the control without selecting again.
      if event.client is not None and self._on_select_iteration is not None:
        self._iteration.value = self._on_select_iteration(event.target.value)

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

  def set_iteration_control(
      self, history_count, iteration, on_select, *, disabled=False
  ) -> None:
    self._on_select_iteration = on_select
    self._iteration.max = history_count - 1
    self._iteration.value = iteration
    self._iteration.disabled = disabled or history_count == 0

  def set_plot(self, name: str, figure: go.Figure) -> None:
    figure.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    if name not in self._plots:
      with self._folder:
        self._plots[name] = self._server.gui.add_plotly(figure, aspect=3/4)
    else:
      self._plots[name].figure = figure

  def get_plot_frames(self) -> dict[str, bytes]:
    frames = {}
    for name, handle in self._plots.items():
      figure, aspect = handle.figure, handle.aspect
      width, height = figure.layout.width, figure.layout.height
      if width is None:
        width = round(height / aspect) if height is not None else 400
      if height is None:
        height = round(width * aspect)
      frames[name] = figure.to_image(
          format='png',
          width=width,
          height=height,
          scale=3,
      )
    return frames

  def cleanup(self) -> None:
    if self._cleaned_up:
      return
    self._on_select_iteration = None
    self._folder.remove()
    self._server.scene.reset()
    self._plots.clear()
    self._cleaned_up = True
