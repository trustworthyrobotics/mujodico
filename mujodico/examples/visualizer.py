from pathlib import Path
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
    self._recording = None
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
      self._version += 1
      self._condition.notify_all()

  def _capture_frames(
      self, server, client, task, traj, *, video=True, serializer=None
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
    for frame in range(task.horizon + 1):
      with server.atomic():
        task.update_visualization(server, traj, frame)
      if serializer is not None:
        serializer.insert_sleep(task.model.opt.timestep)
      if video:
        server.flush()
        image = client.get_render(**view)
        yield iio.imwrite('<bytes>', image, extension='.jpg')
      else:
        yield None

  def _record_playback(
      self, server, client, index, *, video=False, viser_recording=False
  ):
    with self._condition:
      task, traj = self._playbacks[index]

    if video:
      ffmpeg = shutil.which('ffmpeg')
      if ffmpeg is None:
        raise RuntimeError('Install ffmpeg to download playback as a video.')

    serializer = server.get_scene_serializer() if viser_recording else None
    frames = self._capture_frames(
        server, client, task, traj, video=video, serializer=serializer
    )
    try:
      if video:
        filename = f'playback_{index}.mp4'
        with tempfile.TemporaryDirectory(
            prefix='mujodico-playback-'
        ) as directory:
          path = Path(directory)
          for frame, image in enumerate(frames):
            (path / f'frame_{frame:06d}.jpg').write_bytes(image)
          result = subprocess.run(
              [
                  ffmpeg,
                  '-nostdin',
                  '-hide_banner',
                  '-loglevel',
                  'error',
                  '-framerate',
                  f'{1 / task.model.opt.timestep:.12g}',
                  '-start_number',
                  '0',
                  '-i',
                  'frame_%06d.jpg',
                  '-vf',
                  'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                  '-c:v',
                  'libx264',
                  '-pix_fmt',
                  'yuv420p',
                  '-movflags',
                  '+faststart',
                  filename,
              ],
              cwd=directory,
              capture_output=True,
              text=True,
          )
          if result.returncode != 0:
            raise RuntimeError(f'ffmpeg failed: {result.stderr.strip()}')
          client.send_file_download(
              filename=filename,
              content=(path / filename).read_bytes(),
              save_immediately=True,
          )
      else:
        for _ in frames:
          pass
    finally:
      if serializer is not None:
        content = serializer.serialize()

    if serializer is not None:
      client.send_file_download(
          filename=f'playback_{index}.viser',
          content=content,
          save_immediately=True,
      )

  def _visualization_loop(self):
    import viser

    server = viser.ViserServer()
    playback = server.gui.add_number(
        'Playback',
        initial_value=0,
        min=0,
        max=0,
        step=1,
        disabled=True,
        hint='Zero-based index of the saved task and trajectory to play.',
    )
    record_video = server.gui.add_button(
        'Download playback video',
        disabled=True,
        hint='Record the selected playback and download it as an MP4.',
    )
    record_viser = server.gui.add_button(
        'Download viser recording',
        disabled=True,
        hint='Record the selected playback and download it as a .viser file.',
    )
    record_all = server.gui.add_button(
        'Download all',
        disabled=True,
        hint='Record the selected playback as both an MP4 and a .viser file.',
    )
    record_buttons = (record_video, record_viser, record_all)

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

    def _request_recording(event, *, video=False, viser_recording=False):
      if event.client is None:
        return
      with self._condition:
        if not self._playbacks or self._recording is not None:
          return
        self._recording = (event.client, self._selected, video, viser_recording)
        playback.disabled = True
        for button in record_buttons:
          button.disabled = True
        event.target.label = 'Recording...'
        self._condition.notify_all()

    @record_video.on_click
    def _request_video_recording(event):
      _request_recording(event, video=True)

    @record_viser.on_click
    def _request_viser_recording(event):
      _request_recording(event, viser_recording=True)

    @record_all.on_click
    def _request_all_recordings(event):
      _request_recording(event, video=True, viser_recording=True)

    version = -1
    frame = 0

    while True:
      with self._condition:
        while not self._playbacks:
          self._condition.wait()
        recording = self._recording
        if recording is None and self._version != version:
          task, traj = self._playbacks[self._selected]
          playback.max = len(self._playbacks) - 1
          playback.value = self._selected
          playback.disabled = False
          for button in record_buttons:
            button.disabled = False
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
        client, index, video, viser_recording = recording
        try:
          self._record_playback(
              server,
              client,
              index,
              video=video,
              viser_recording=viser_recording,
          )
        except Exception as error:
          client.add_notification(
              title='Recording failed', body=str(error), color='red'
          )
        finally:
          with self._condition:
            self._recording = None
            playback.max = len(self._playbacks) - 1
            playback.disabled = False
            for button in record_buttons:
              button.disabled = False
            record_video.label = 'Download playback video'
            record_viser.label = 'Download viser recording'
            record_all.label = 'Download all'
            version = -1
        continue

      tic = time.perf_counter()
      task.update_visualization(server, traj, frame)
      remaining = task.model.opt.timestep - (time.perf_counter() - tic)
      if remaining > 0:
        with self._condition:
          if self._version == version and self._recording is None:
            self._condition.wait(timeout=remaining)
      frame += 1
