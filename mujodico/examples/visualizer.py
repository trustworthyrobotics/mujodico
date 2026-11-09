import contextlib
import logging
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import tempfile
import threading
import time


class Visualizer:

  def __init__(self):
    self._condition = threading.Condition()
    self._playbacks = []
    self._selected = 0
    self._selected_iteration = -1
    self._recording = None
    self._visualized_task = None
    self._version = 0
    self._thread = None

  def run(self):
    with self._condition:
      if self._thread is not None and self._thread.is_alive():
        return
      self._thread = threading.Thread(
          target=self._visualization_loop, daemon=True
      )
      self._thread.start()

  def add(self, task, traj):
    """Save a playback, selecting it unless a recording is in progress."""
    with self._condition:
      self._playbacks.append((task, dict(traj)))
      if self._recording is None:
        self._selected = len(self._playbacks) - 1
        self._selected_iteration = -1
      self._version += 1
      self._condition.notify_all()

  @staticmethod
  def _iteration_trajectory(traj, iteration):
    """Resolve a snapshot while retaining the playback's information."""
    if iteration < 0:
      return traj
    return {**traj, **traj['history'][iteration]}

  def _select_iteration(self, index, iteration):
    with self._condition:
      if index != self._selected or self._recording is not None:
        return self._selected_iteration
      _, traj = self._playbacks[index]
      iteration = min(max(int(iteration), -1), len(traj.get('history', [])) - 1)
      if iteration != self._selected_iteration:
        self._selected_iteration = iteration
        self._version += 1
        self._condition.notify_all()
      return iteration

  def _configure_iteration_control(self, server):
    visualizer = getattr(server, 'task_visualizer', None)
    if visualizer is None:
      return
    with self._condition:
      index = self._selected
      _, traj = self._playbacks[index]
      visualizer.set_iteration_control(
          len(traj.get('history', [])),
          self._selected_iteration,
          lambda iteration: self._select_iteration(index, iteration),
          disabled=self._recording is not None,
      )

  def _download_pickle(self, client, index):
    with self._condition:
      task, traj = self._playbacks[index]
    client.send_file_download(
        filename=f'playback_{index}.pkl',
        content=pickle.dumps(
            {'task': task, 'traj': traj}, protocol=pickle.HIGHEST_PROTOCOL
        ),
        save_immediately=True,
    )

  def _load_pickle(self, content):
    from mujodico.examples.tasks.task import Task

    saved = pickle.loads(content)
    if not isinstance(saved, dict) or not {'task', 'traj'} <= saved.keys():
      raise ValueError('The file must contain a task and trajectory.')
    task, traj = saved['task'], saved['traj']
    if not isinstance(task, Task) or not isinstance(traj, dict):
      raise ValueError(
          'The file must contain a Task and a trajectory dictionary.'
      )
    if task.horizon < 0 or task.model.opt.timestep <= 0:
      raise ValueError('The task must have a valid horizon and timestep.')
    if getattr(traj.get('xs'), 'shape', None) != (task.horizon + 1, task.nx):
      raise ValueError(
          'The trajectory states must match the task horizon and state size.'
      )
    self.add(task, traj)

  def _prepare_visualization(self, server, task):
    if self._visualized_task is task:
      return
    visualizer = getattr(server, 'task_visualizer', None)
    if visualizer is not None:
      visualizer.cleanup()
      del server.task_visualizer
    self._visualized_task = task

  @contextlib.contextmanager
  def _plot_renderer(self, directory):
    try:
      import kaleido
      logging.getLogger('kaleido').setLevel(logging.WARNING)
      logging.getLogger('choreographer').setLevel(logging.WARNING)
    except ImportError as error:
      raise ImportError(
          'Plot recording requires Kaleido 1.1 or newer. Install it with pip'
          ' install "kaleido>=1.1" and install Chrome with plotly_get_chrome.'
      ) from error
    if not hasattr(kaleido, 'start_sync_server'):
      raise ImportError('Upgrade Kaleido with pip install "kaleido>=1.1".')

    import plotly.graph_objects as go

    # Probe before starting the persistent server: a browser startup failure
    # in Kaleido's background server can otherwise leave a sync call waiting.
    options = dict(
        page_generator=kaleido.PageGenerator(mathjax=False), timeout=30
    )
    try:
      kaleido.calc_fig_sync(
          go.Figure(),
          opts=dict(format='png', width=64, height=64),
          kopts=options,
      )
    except Exception as error:
      raise RuntimeError(
          'Could not start Plotly image export. Install Kaleido 1.1 or newer '
          'and Chrome/Chromium (run plotly_get_chrome). '
          f'Details: {error}'
      ) from error
    try:
      kaleido.start_sync_server(**options)
      yield
    finally:
      kaleido.stop_sync_server()

  def _skip_plot_recording(self, client, error):
    client.add_notification(
        title='Plot recording unavailable',
        body=f'Plots will not be recorded. {error}',
        color='red',
    )

  def _capture_frames(
      self,
      server,
      client,
      task,
      traj,
      directory,
      *,
      video=False,
      plots=False,
      serializer=None,
  ):
    if video:
      import imageio.v3 as iio

      camera = client.camera
      height, width = camera.image_height, camera.image_width
      if height <= 0 or width <= 0:
        raise ValueError('The browser viewport is not ready for recording.')

      view = dict(
          height=height,
          width=width,
          wxyz=tuple(camera.wxyz),
          position=tuple(camera.position),
          fov=camera.fov,
          transport_format='jpeg',
      )
      scene_directory = directory / 'scene'
      scene_directory.mkdir()
    plot_directories = {}
    for frame in range(task.horizon + 1):
      with server.atomic():
        task.update_visualization(server, traj, frame)
        if frame == 0:
          self._configure_iteration_control(server)
      if serializer is not None:
        serializer.insert_sleep(task.model.opt.timestep)
      if plots:
        task_visualizer = getattr(server, 'task_visualizer', None)
        if task_visualizer is None:
          raise RuntimeError('The selected task has no Plotly plots to record.')
        try:
          plot_frames = task_visualizer.get_plot_frames()
        except ImportError as error:
          self._skip_plot_recording(client, error)
          plots = False
          plot_directories.clear()
          if not video and serializer is None:
            return {}
        else:
          if not plot_frames:
            raise RuntimeError(
                'The selected task has no Plotly plots to record.'
            )
          for ordinal, (name, image) in enumerate(plot_frames.items()):
            path = directory / 'plots' / f'plot_{ordinal}'
            path.mkdir(parents=True, exist_ok=True)
            (path / f'frame_{frame:06d}.png').write_bytes(image)
            plot_directories[name] = path
      if video:
        server.flush()
        image = client.get_render(**view)
        iio.imwrite(scene_directory / f'frame_{frame:06d}.jpg', image)
    return plot_directories

  def _encode_video(self, ffmpeg, directory, output, timestep, *, extension):
    result = subprocess.run(
        [
            ffmpeg,
            '-nostdin',
            '-hide_banner',
            '-loglevel',
            'error',
            '-framerate',
            f'{1 / timestep:.12g}',
            '-start_number',
            '0',
            '-i',
            f'frame_%06d.{extension}',
            '-vf',
            'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-c:v',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart',
            str(output),
        ],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
      raise RuntimeError(f'ffmpeg failed: {result.stderr.strip()}')

  def _record_playback(
      self,
      server,
      client,
      index,
      *,
      iteration=-1,
      video=False,
      plots=False,
      viser_recording=False,
  ):
    if not (video or plots or viser_recording):
      raise ValueError('Select at least one recording type.')
    with self._condition:
      task, traj = self._playbacks[index]
      traj = self._iteration_trajectory(traj, iteration)

    if video or plots:
      ffmpeg = shutil.which('ffmpeg')
      if ffmpeg is None:
        error = RuntimeError('Install ffmpeg to download playback as a video.')
        if video:
          client.add_notification(
              title='Video recording unavailable',
              body=f'3D scene and plots will not be recorded. {error}',
              color='red',
          )
        else:
          self._skip_plot_recording(client, error)
        video = False
        plots = False

    with tempfile.TemporaryDirectory(prefix='mujodico-playback-') as directory:
      path = Path(directory)
      with contextlib.ExitStack() as resources:
        if plots:
          try:
            resources.enter_context(self._plot_renderer(path))
          except Exception as error:
            self._skip_plot_recording(client, error)
            plots = False
        if not (video or plots or viser_recording):
          return
        self._prepare_visualization(server, task)
        serializer = server.get_scene_serializer() if viser_recording else None
        try:
          plot_directories = self._capture_frames(
              server,
              client,
              task,
              traj,
              path,
              video=video,
              plots=plots,
              serializer=serializer,
          )
        finally:
          if serializer is not None:
            content = serializer.serialize()

      outputs = []
      if video:
        output = path / f'playback_{index}.mp4'
        self._encode_video(
            ffmpeg,
            path / 'scene',
            output,
            task.model.opt.timestep,
            extension='jpg',
        )
        outputs.append(output)
      used_names = set()
      for name, frames_directory in plot_directories.items():
        safe_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', name).strip('_') or 'plot'
        stem = f'playback_{index}_{safe_name}'
        filename = f'{stem}.mp4'
        suffix = 2
        while filename in used_names:
          filename = f'{stem}_{suffix}.mp4'
          suffix += 1
        used_names.add(filename)
        output = path / filename
        self._encode_video(
            ffmpeg,
            frames_directory,
            output,
            task.model.opt.timestep,
            extension='png',
        )
        outputs.append(output)
      for output in outputs:
        client.send_file_download(
            filename=output.name,
            content=output.read_bytes(),
            save_immediately=True,
        )

      if serializer is not None:
        client.send_file_download(
            filename=f'playback_{index}.viser',
            content=content,
            save_immediately=True,
        )

  def _visualization_loop(self):
    import viser

    server = viser.ViserServer()
    with server.gui.add_folder('Playback controls', expand_by_default=True):
      playback = server.gui.add_number(
          'Playback',
          initial_value=0,
          min=0,
          max=0,
          step=1,
          disabled=True,
          hint='Zero-based index of the saved task and trajectory to play.',
      )
      with server.gui.add_folder('Record', expand_by_default=True):
        record_scene = server.gui.add_checkbox(
            '3D scene',
            initial_value=True,
            hint='Download the MuJoCo camera view as an MP4.',
        )
        record_plots = server.gui.add_checkbox(
            'plot',
            initial_value=True,
            hint='Download a separate MP4 for each Plotly plot.',
        )
        record_viser = server.gui.add_checkbox(
            'viser',
            initial_value=False,
            hint='Record the selected playback and download it as a .viser file.',
        )
        download_recording = server.gui.add_button(
            'Download recording',
            disabled=True,
            hint='Download the selected recording formats in one playback pass.',
        )
      with server.gui.add_folder('Pickle', expand_by_default=True):
        download_pickle = server.gui.add_button(
            'Download pickle',
            disabled=True,
            hint='Save the selected task and trajectory as a .pkl file for playback.',
        )
        upload_playback = server.gui.add_upload_button(
            'Upload pickle',
            mime_type='.pkl,.pickle',
            hint='Load a saved .pkl file for playback.',
        )
    record_controls = (
        record_scene,
        record_plots,
        record_viser,
        download_recording,
    )

    @upload_playback.on_upload
    def _upload_playback(event):
      if event.client is None:
        return
      try:
        self._load_pickle(event.target.value.content)
      except Exception as error:
        event.client.add_notification(
            title='Upload failed', body=str(error), color='red'
        )

    @playback.on_update
    def _select_playback(event):
      with self._condition:
        if not self._playbacks:
          return
        if self._recording is not None:
          event.target.value = self._selected
          return
        index = min(max(int(event.target.value), 0), len(self._playbacks) - 1)
        event.target.value = index
        if index != self._selected:
          self._selected = index
          self._selected_iteration = -1
          self._version += 1
          self._condition.notify_all()

    @download_recording.on_click
    def _request_recording(event):
      if event.client is None:
        return
      with self._condition:
        if not self._playbacks or self._recording is not None:
          return
        video = record_scene.value
        plots = record_plots.value
        viser_recording = record_viser.value
        if not (video or plots or viser_recording):
          event.client.add_notification(
              title='No recording selected',
              body='Select at least one recording type.',
              color='yellow',
          )
          return
        self._recording = (
            event.client,
            self._selected,
            self._selected_iteration,
            video,
            plots,
            viser_recording,
        )
        playback.disabled = True
        self._configure_iteration_control(server)
        for control in record_controls:
          control.disabled = True
        download_recording.label = 'Recording...'
        self._condition.notify_all()

    @download_pickle.on_click
    def _request_playback_download(event):
      if event.client is None:
        return
      with self._condition:
        if not self._playbacks:
          return
        index = self._selected
      try:
        self._download_pickle(event.client, index)
      except Exception as error:
        event.client.add_notification(
            title='Download failed', body=str(error), color='red'
        )

    version = -1
    frame = 0

    while True:
      with self._condition:
        while not self._playbacks:
          self._condition.wait()
        recording = self._recording
        if recording is None and self._version != version:
          task, traj = self._playbacks[self._selected]
          traj = self._iteration_trajectory(traj, self._selected_iteration)
          playback.max = len(self._playbacks) - 1
          playback.value = self._selected
          playback.disabled = False
          download_pickle.disabled = False
          for control in record_controls:
            control.disabled = False
          version = self._version
          frame = 0
        elif recording is None and frame > task.horizon:
          self._condition.wait_for(
              lambda: self._version != version or self._recording is not None,
              timeout=1.0,
          )
          if self._version != version or self._recording is not None:
            continue
          frame = 0

      if recording is not None:
        client, index, iteration, video, plots, viser_recording = recording
        try:
          self._record_playback(
              server,
              client,
              index,
              iteration=iteration,
              video=video,
              plots=plots,
              viser_recording=viser_recording,
          )
        except Exception as error:
          client.add_notification(
              title='Download failed', body=str(error), color='red'
          )
        finally:
          with self._condition:
            self._recording = None
            self._configure_iteration_control(server)
            playback.max = len(self._playbacks) - 1
            playback.disabled = False
            for control in record_controls:
              control.disabled = False
            download_recording.label = 'Download recording'
            version = -1
        continue

      tic = time.perf_counter()
      self._prepare_visualization(server, task)
      task.update_visualization(server, traj, frame)
      self._configure_iteration_control(server)
      remaining = task.model.opt.timestep - (time.perf_counter() - tic)
      if remaining > 0:
        with self._condition:
          if self._version == version and self._recording is None:
            self._condition.wait(timeout=remaining)
      frame += 1
