from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from typing_extensions import override

import mujoco
import numpy as np

from mujodico.examples.visualizer import Playback

if TYPE_CHECKING:
  from mujodico.examples.tasks.task import Task

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


class TaskPlayback(Playback):

  def __init__(self, task: Task, traj: dict):
    timestep = float(task.model.opt.timestep)
    if task.horizon < 0 or not np.isfinite(timestep) or timestep <= 0:
      raise ValueError('The task must have a valid horizon and timestep.')
    if not isinstance(traj, dict):
      raise ValueError('The trajectory must be a dictionary.')
    history = traj.get('history', [])
    if not isinstance(history, list):
      raise ValueError('Trajectory history must be a list.')
    for trajectory in [traj, *history]:
      if not isinstance(trajectory, dict) or getattr(
          trajectory.get('xs'), 'shape', None
      ) != (task.horizon + 1, task.nx):
        raise ValueError(
            'Trajectory states must match the task horizon and state size.'
        )

    self._task = task
    self._traj = dict(traj)
    self._timestep = timestep
    self._lock = threading.RLock()
    self._selected_iteration = -1
    self._frame = 0
    self._interactive = True
    self._server = None
    self._data = None
    self._mj_scene = None
    self._folder = None
    self._iteration = None
    self._info_panel = None
    self._plots: dict[str, viser.GuiPlotlyHandle] = {}

  def __getstate__(self):
    with self._lock:
      return {'task': self._task, 'traj': self._traj}

  def __setstate__(self, state):
    self.__init__(state['task'], state['traj'])

  def __del__(self):
    try:
      self.cleanup()
    except Exception:
      pass

  @property
  @override
  def timestep(self) -> float:
    return self._timestep

  @override
  def attach(self, server: viser.ViserServer) -> None:
    with self._lock:
      if self._server is server:
        return
      self.cleanup()
      self._server = server
      self._selected_iteration = -1
      self._frame = 0
      try:
        server.scene.add_grid('/grid')
        model = self._task.model
        self._data = mujoco.MjData(model)
        self._mj_scene = mjviser.ViserMujocoScene(server, model, num_envs=1)
        self._mj_scene.camera_tracking_enabled = False
        self._folder = server.gui.add_folder('Task', expand_by_default=True)
        history_count = len(self._traj.get('history', []))
        with self._folder:
          self._iteration = server.gui.add_number(
              'Iteration',
              initial_value=-1,
              min=-1,
              max=history_count - 1,
              step=1,
              disabled=not self._interactive or history_count == 0,
              hint=(
                  '-1 plays the final result; 0 and above select recorded '
                  'iterations by their zero-based position in history.'
              ),
          )
          self._info_panel = server.gui.add_markdown('')
        self._iteration.on_update(self._select_iteration)
      except Exception:
        self.cleanup()
        raise

  def _select_iteration(self, event) -> None:
    # Server-side control updates must not trigger another selection.
    if event.client is None:
      return
    with self._lock:
      if self._iteration is None or event.target is not self._iteration:
        return
      if self._interactive:
        count = len(self._traj.get('history', []))
        iteration = min(max(int(event.target.value), -1), count - 1)
        if iteration != self._selected_iteration:
          self._selected_iteration = iteration
          self.reset()
      self._iteration.value = self._selected_iteration

  @override
  def set_interactive(self, enabled: bool) -> None:
    with self._lock:
      self._interactive = enabled
      if self._iteration is not None:
        self._iteration.disabled = not enabled or not self._traj.get('history')

  @override
  def play_next_frame(self) -> bool:
    with self._lock:
      if self._server is None:
        raise RuntimeError('Attach the playback before rendering frames.')
      if self._frame > self._task.horizon:
        return False
      traj = self._traj
      if self._selected_iteration >= 0:
        traj = {**traj, **traj['history'][self._selected_iteration]}
      self._task.update_visualization(self, traj, self._frame)
      self._frame += 1
      return True

  @override
  def reset(self) -> None:
    with self._lock:
      self._frame = 0

  def update_scene(self, qpos: np.ndarray) -> None:
    self._data.qpos = qpos
    mujoco.mj_kinematics(self._task.model, self._data)
    self._mj_scene.update_from_mjdata(self._data)

  def set_text(self, text: str) -> None:
    self._info_panel.content = text

  def set_plot(self, name: str, figure: go.Figure) -> None:
    figure.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    if name not in self._plots:
      with self._folder:
        self._plots[name] = self._server.gui.add_plotly(figure, aspect=3 / 4)
    else:
      self._plots[name].figure = figure

  @override
  def get_plot_frames(self) -> dict[str, bytes]:
    with self._lock:
      frames = {}
      for name, handle in self._plots.items():
        figure, aspect = handle.figure, handle.aspect
        width, height = figure.layout.width, figure.layout.height
        if width is None:
          width = round(height / aspect) if height is not None else 400
        if height is None:
          height = round(width * aspect)
        frames[name] = figure.to_image(
            format='png', width=width, height=height, scale=3
        )
      return frames

  @override
  def cleanup(self) -> None:
    with self._lock:
      if self._server is None:
        return
      if self._folder is not None:
        self._folder.remove()
      self._server.scene.reset()
      self._server = None
      self._data = None
      self._mj_scene = None
      self._folder = None
      self._iteration = None
      self._info_panel = None
      self._plots.clear()
