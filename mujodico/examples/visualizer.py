from abc import ABC, abstractmethod
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import viser


class Playback(ABC):
  """A playback owns its frame cursor, controls, and attached resources.

  Instances must be pickleable without their live server resources. A cleaned
  up instance can be attached again, including after loading it from a pickle.
  """

  @property
  @abstractmethod
  def timestep(self) -> float:
    """Positive, constant duration of each frame in seconds."""

  @abstractmethod
  def attach(self, server: 'viser.ViserServer') -> None:
    """Attach resources, doing nothing if already attached to this server."""

  @abstractmethod
  def play_next_frame(self) -> bool:
    """Render and advance one frame, returning False when exhausted."""

  @abstractmethod
  def reset(self) -> None:
    """Rewind to the first frame, preserving the selected content."""

  @abstractmethod
  def get_plot_frames(self) -> dict[str, bytes]:
    """Return PNG images of the plots for the most recently rendered frame."""

  @abstractmethod
  def cleanup(self) -> None:
    """Release attached resources; repeated calls must be safe."""

  def set_interactive(self, enabled: bool) -> None:
    """Enable or freeze controls that can change playback during recording."""


class Visualizer:

  def __init__(self):
    self._condition = threading.Condition()
    self._playbacks = []
    self._selected = 0
    self._recording = None
    self._attached_playback = None
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

  def add(self, playback: Playback):
    """Save a playback, selecting it unless a recording is in progress."""
    if not isinstance(playback, Playback):
      raise TypeError('Expected a Playback instance.')
    with self._condition:
      self._playbacks.append(playback)
      if self._recording is None:
        self._selected = len(self._playbacks) - 1
      self._version += 1
      self._condition.notify_all()

  def _download_pickle(self, client, index):
    with self._condition:
      playback = self._playbacks[index]
    client.send_file_download(
        filename=f'playback_{index}.pkl',
        content=pickle.dumps(playback, protocol=pickle.HIGHEST_PROTOCOL),
        save_immediately=True,
    )

  def _load_pickle(self, content):
    saved = pickle.loads(content)
    if not isinstance(saved, Playback):
      raise ValueError('The file must contain a Playback instance.')
    self.add(saved)

  def _prepare_visualization(self, server, playback):
    if self._attached_playback is playback:
      return
    if self._attached_playback is not None:
      self._attached_playback.cleanup()
      self._attached_playback = None
    playback.attach(server)
    self._attached_playback = playback

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
      playback,
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
    frame = 0
    while True:
      with server.atomic():
        if not playback.play_next_frame():
          break
      if serializer is not None:
        serializer.insert_sleep(playback.timestep)
      if plots:
        try:
          plot_frames = playback.get_plot_frames()
        except ImportError as error:
          self._skip_plot_recording(client, error)
          plots = False
          plot_directories.clear()
          if not video and serializer is None:
            return {}
        else:
          if not plot_frames:
            raise RuntimeError('The selected playback has no plots to record.')
          for ordinal, (name, image) in enumerate(plot_frames.items()):
            path = directory / 'plots' / f'plot_{ordinal}'
            path.mkdir(parents=True, exist_ok=True)
            (path / f'frame_{frame:06d}.png').write_bytes(image)
            plot_directories[name] = path
      if video:
        server.flush()
        image = client.get_render(**view)
        iio.imwrite(scene_directory / f'frame_{frame:06d}.jpg', image)
      frame += 1
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
      video=False,
      plots=False,
      viser_recording=False,
  ):
    if not (video or plots or viser_recording):
      raise ValueError('Select at least one recording type.')
    with self._condition:
      playback = self._playbacks[index]
    try:
      playback.set_interactive(False)
      self._prepare_visualization(server, playback)
      playback.reset()
      self._download_recording(
          server,
          client,
          playback,
          index,
          video=video,
          plots=plots,
          viser_recording=viser_recording,
      )
    finally:
      try:
        playback.reset()
      finally:
        playback.set_interactive(True)

  def _download_recording(
      self, server, client, playback, index, *, video, plots, viser_recording
  ):

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
        serializer = server.get_scene_serializer() if viser_recording else None
        try:
          plot_directories = self._capture_frames(
              server,
              client,
              playback,
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
            playback.timestep,
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
            playback.timestep,
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
          hint='Zero-based index of the saved playback to play.',
      )
      with server.gui.add_folder('Record', expand_by_default=False):
        record_scene = server.gui.add_checkbox(
            '3D scene',
            initial_value=True,
            hint='Download the camera view as an MP4.',
        )
        record_plots = server.gui.add_checkbox(
            'plot',
            initial_value=True,
            hint='Download a separate MP4 for each Plotly plot.',
        )
        record_viser = server.gui.add_checkbox(
            'viser',
            initial_value=False,
            hint=(
                'Record the selected playback and download it as a .viser file.'
            ),
        )
        download_recording = server.gui.add_button(
            'Download recording',
            disabled=True,
            hint=(
                'Download the selected recording formats in one playback pass.'
            ),
        )
      with server.gui.add_folder('Pickle', expand_by_default=False):
        download_pickle = server.gui.add_button(
            'Download pickle',
            disabled=True,
            hint='Save the selected playback object as a .pkl file.',
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
            video,
            plots,
            viser_recording,
        )
        playback.disabled = True
        self._playbacks[self._selected].set_interactive(False)
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
    exhausted_since = None

    try:
      while True:
        with self._condition:
          while not self._playbacks:
            self._condition.wait()
          recording = self._recording
          if recording is None and self._version != version:
            current = self._playbacks[self._selected]
            self._prepare_visualization(server, current)
            current.reset()
            playback.max = len(self._playbacks) - 1
            playback.value = self._selected
            playback.disabled = False
            download_pickle.disabled = False
            for control in record_controls:
              control.disabled = False
            version = self._version
            exhausted_since = None

        if recording is not None:
          client, index, video, plots, viser_recording = recording
          try:
            self._record_playback(
                server,
                client,
                index,
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
              playback.max = len(self._playbacks) - 1
              playback.disabled = False
              for control in record_controls:
                control.disabled = False
              download_recording.label = 'Download recording'
              version = -1
          continue

        tic = time.perf_counter()
        with self._condition:
          if self._version != version or self._recording is not None:
            continue
          with server.atomic():
            rendered = current.play_next_frame()
          if rendered:
            exhausted_since = None
            remaining = current.timestep - (time.perf_counter() - tic)
          else:
            now = time.perf_counter()
            if exhausted_since is None:
              exhausted_since = now
            elapsed = now - exhausted_since
            if elapsed >= 1.0:
              current.reset()
              exhausted_since = None
              continue
            # A playback can reset itself through its own controls. Poll for
            # that without knowing what content or selection it manages.
            remaining = min(0.05, 1.0 - elapsed)
          if remaining > 0:
            self._condition.wait(timeout=remaining)
    finally:
      if self._attached_playback is not None:
        self._attached_playback.cleanup()
        self._attached_playback = None
      server.stop()
