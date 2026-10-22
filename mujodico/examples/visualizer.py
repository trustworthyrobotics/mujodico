import io
import threading
import time
import zipfile


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

  def _record_playback(self, server, client, index):
    import imageio.v3 as iio

    with self._condition:
      task, traj = self._playbacks[index]

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
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
      for frame in range(task.horizon + 1):
        with server.atomic():
          task.update_visualization(server, traj, frame)
        server.flush()
        image = client.get_render(**view)
        archive.writestr(
            f'frame_{frame:06d}.jpg',
            iio.imwrite('<bytes>', image, extension='.jpg'),
        )

    client.send_file_download(
        filename=f'playback_{index}.zip', content=buffer.getvalue()
    )
    client.add_notification(
        title=f'Convert playback_{index}.zip to MP4',
        body=(
            f'ffmpeg -framerate {1 / task.model.opt.timestep:.12g} '
            '-start_number 0 -i frame_%06d.jpg '
            '-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
            '-c:v libx264 -pix_fmt yuv420p '
            f'../playback_{index}.mp4'
        ),
        auto_close_seconds=None,
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
    record = server.gui.add_button(
        'Download playback frames',
        disabled=True,
        hint='Record every frame of the selected playback as a ZIP of JPEGs.',
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

    @record.on_click
    def _request_recording(event):
      if event.client is None:
        return
      with self._condition:
        if not self._playbacks or self._recording is not None:
          return
        self._recording = (event.client, self._selected)
        playback.disabled = True
        record.disabled = True
        record.label = 'Recording...'
        self._condition.notify_all()

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
          record.disabled = False
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
        client, index = recording
        try:
          self._record_playback(server, client, index)
        except Exception as error:
          client.add_notification(
              title='Recording failed', body=str(error), color='red'
          )
        finally:
          with self._condition:
            self._recording = None
            playback.max = len(self._playbacks) - 1
            playback.disabled = False
            record.disabled = False
            record.label = 'Download playback frames'
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
