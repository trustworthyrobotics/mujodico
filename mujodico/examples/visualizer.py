import threading
import time


class Visualizer:

  def __init__(self):
    self._condition = threading.Condition()
    self._task = None
    self._traj = None
    self._version = 0
    self._thread = None

  def run(self):
    with self._condition:
      if self._thread is not None and self._thread.is_alive():
        return
      self._thread = threading.Thread(target=self._visualization_loop, daemon=True)
      self._thread.start()

  def update(self, task, traj):
    with self._condition:
      self._task = task
      self._traj = dict(traj)
      self._version += 1
      self._condition.notify_all()

  def _visualization_loop(self):
    import viser

    server = viser.ViserServer()
    version = -1
    frame = 0

    while True:
      with self._condition:
        while self._task is None:
          self._condition.wait()
        if self._version != version:
          task = self._task
          traj = self._traj
          version = self._version
          frame = 0
        elif frame > task.horizon:
          self._condition.wait_for(
              lambda: self._version != version,
              timeout=1.0,
          )
          if self._version != version:
            continue
          frame = 0

      tic = time.perf_counter()
      task.update_visualization(server, traj, frame)
      remaining = task.model.opt.timestep - (time.perf_counter() - tic)
      if remaining > 0:
        with self._condition:
          if self._version == version:
            self._condition.wait(timeout=remaining)
      frame += 1
