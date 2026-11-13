from __future__ import annotations

import contextlib
import json
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
  from plotly.graph_objects import Figure
  from mujodico.examples.tasks.util_uplot import UplotFigure
except ImportError as e:
  raise ImportError(
      'Optional visualization dependencies are not installed. Install them'
      ' with:\n\n    pip install mjviser plotly'
  ) from e


_INSTALL = (
    'Plot recording requires Playwright and Chromium. Install them with:\n\n'
    '    pip install playwright imageio\n'
    '    python -m playwright install chromium'
)


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
    self._plots: dict[str, Figure | UplotFigure] = {}
    self._plot_handles: dict[str, tuple] = {}
    self._plot_renderer = None

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

  def set_plot(self, name: str, figure: Figure | UplotFigure) -> None:
    with self._lock:
      previous = self._plots.get(name)
      if previous is not None and (
          isinstance(previous, UplotFigure) != isinstance(figure, UplotFigure)
          or (
              isinstance(figure, UplotFigure)
              and len(previous.panels) != len(figure.panels)
          )
      ):
        for handle in self._plot_handles.pop(name):
          handle.remove()
        previous = None

      if isinstance(figure, UplotFigure):
        if previous is None:
          with self._folder:
            self._plot_handles[name] = tuple(
                self._server.gui.add_uplot(
                    panel.data,
                    panel.series,
                    **figure.panel_options(i),
                )
                for i, panel in enumerate(figure.panels)
            )
        else:
          for i, (handle, panel) in enumerate(
              zip(self._plot_handles[name], figure.panels)
          ):
            if previous.panels[i].series != panel.series:
              handle.series = panel.series
            options = figure.panel_options(i)
            old_options = previous.panel_options(i)
            for key, value in options.items():
              if old_options.get(key) != value:
                setattr(handle, key, value)
            handle.data = panel.data
      else:
        if previous is None:
          with self._folder:
            self._plot_handles[name] = (
                self._server.gui.add_plotly(figure, aspect=3 / 4),
            )
        else:
          self._plot_handles[name][0].figure = figure
      self._plots[name] = figure

  @contextlib.contextmanager
  @override
  def plot_renderer(self):
    """Keep one browser alive for the duration of a recording."""
    with self._lock:
      self._close_plot_renderer()
      self._open_plot_renderer()
    try:
      yield
    finally:
      with self._lock:
        self._close_plot_renderer()

  def _open_plot_renderer(self):
    if self._plot_renderer is None:
      self._plot_renderer = PlotRenderer().open()

  def _close_plot_renderer(self):
    renderer, self._plot_renderer = self._plot_renderer, None
    if renderer is not None:
      renderer.close()

  @override
  def get_plot_frames(self) -> dict[str, bytes]:
    with self._lock:
      if not self._plots:
        return {}
      try:
        self._open_plot_renderer()
        return {
            name: self._plot_renderer.render(name, figure)
            for name, figure in self._plots.items()
        }
      except Exception:
        self._close_plot_renderer()
        raise

  @override
  def cleanup(self) -> None:
    with self._lock:
      self._close_plot_renderer()
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
      self._plot_handles.clear()


class PlotRenderer:
  """Own a browser on the calling thread, reusing chart instances per frame."""

  def __init__(self):
    self._playwright = None
    self._browser = None
    self._page = None
    self._kinds = {}
    self._dimensions = {}
    self._uplot_server = None
    self._uplot_page = None
    self._uplot_plots = {}
    self._uplot_frame = 0

  def open(self):
    if self._page is not None:
      return self
    try:
      from playwright.sync_api import sync_playwright
    except ImportError as error:
      raise ImportError(_INSTALL) from error
    try:
      self._playwright = sync_playwright().start()
      self._browser = self._playwright.chromium.launch(timeout=30000)
      self._page = self._browser.new_page(
          viewport={'width': 400, 'height': 1000},
          device_scale_factor=3,
      )
      self._page.set_default_timeout(30000)
      # All scripts and styles are local. Block incidental external requests.
      self._page.route('**/*', lambda route: route.abort())
      self._page.set_content("""<!doctype html><html><head><style>
        body { margin: 0; background: white; color: #343a40;
               font: 11px sans-serif; }
        .figure { width: 400px; box-sizing: border-box; background: white; }
      </style></head><body></body></html>""")
      from plotly.offline import get_plotlyjs

      self._page.add_script_tag(content=get_plotlyjs())
      self._page.evaluate('window.plots = new Map()')
    except Exception as error:
      self.close()
      raise RuntimeError(
          f'Could not start plot image export. {_INSTALL} Details: {error}'
      ) from error
    return self

  def render(self, name, figure) -> bytes:
    if self._page is None:
      raise RuntimeError('Open the plot renderer before exporting frames.')
    is_uplot = isinstance(figure, UplotFigure)
    kind = ('uplot', len(figure.panels)) if is_uplot else ('plotly', 1)
    if name in self._kinds and self._kinds[name] != kind:
      raise ValueError(
          'A plot cannot change backend or panel count during recording.'
      )
    self._kinds[name] = kind

    if is_uplot:
      return self._render_uplot(name, figure)
    payload = dict(name=name, figure=json.loads(figure.to_json()))
    self._page.evaluate(
        """async ({name, figure}) => {
      let plot = window.plots.get(name);
      if (!plot) {
        const root = document.createElement('div');
        root.className = 'figure';
        root.style.height = '300px';
        document.body.appendChild(root);
        plot = {root};
        window.plots.set(name, plot);
      }
      const layout = {
        ...figure.layout, width: 400, height: 300, autosize: false,
        paper_bgcolor: 'white', plot_bgcolor: 'white',
        font: {...figure.layout.font, color: '#343a40'},
        margin: {l: 48, r: 16, t: 12, b: 44},
        transition: {duration: 0},
      };
      const config = {staticPlot: true, displayModeBar: false};
      if (!plot.fixedLayout) {
        // Let Plotly reserve room for labels and legends once. Its margin
        // and scaleanchor solvers can otherwise oscillate across identical
        // layout inputs, moving the plot box and changing the world scale.
        await Plotly.react(plot.root, figure.data, layout, config);
        await document.fonts.ready;
        const full = plot.root._fullLayout;
        const {l, r, t, b, p} = full._size;
        const axes = {};
        for (const [key, axis] of Object.entries(full)) {
          if (!/^[xy]axis[0-9]*$/.test(key)) continue;
          axes[key] = {
            type: axis.type, domain: axis.domain.slice(),
            range: axis.range.slice(), autorange: false, automargin: false,
            // The resolved ranges already enforce the requested aspect
            // ratio. Fixed pixels/ranges keep it without another solve.
            scaleanchor: false, matches: null,
          };
        }
        plot.fixedLayout = {
          margin: {l, r, t, b, pad: p, autoexpand: false}, axes,
        };
      }
      layout.margin = plot.fixedLayout.margin;
      for (const [key, axis] of Object.entries(plot.fixedLayout.axes)) {
        layout[key] = {...layout[key], ...axis};
      }
      await Plotly.react(plot.root, figure.data, layout, config);
    }""",
        payload,
    )
    self._page.evaluate("""async () => {
      await document.fonts.ready;
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    }""")
    element = self._page.evaluate_handle(
        'name => window.plots.get(name).root', name
    )
    try:
      return self._capture_plot(name, element.as_element())
    finally:
      element.dispose()

  def _capture_plot(self, name, element) -> bytes:
    box = element.bounding_box()
    dimensions = (round(box['width']), round(box['height']))
    if name in self._dimensions and dimensions != self._dimensions[name]:
      raise ValueError('Plot image dimensions changed during recording.')
    self._dimensions[name] = dimensions
    return element.screenshot(type='png', animations='disabled', timeout=30000)

  def _open_uplot_page(self):
    if self._uplot_page is not None:
      return
    self._uplot_server = viser.ViserServer(
        host='127.0.0.1', port=0, verbose=False
    )
    self._uplot_server.gui.configure_theme(
        control_width='large',
        dark_mode=False,
        show_logo=False,
        show_share_button=False,
    )
    self._uplot_server.gui.main_panel.dock_right()
    url = f'http://127.0.0.1:{self._uplot_server.get_port()}/'
    self._uplot_page = self._browser.new_page(
        viewport={'width': 400, 'height': 1000},
        device_scale_factor=3,
    )
    self._uplot_page.set_default_timeout(30000)
    self._uplot_page.route(
        '**/*',
        lambda route: route.continue_()
        if route.request.url.startswith(url)
        else route.abort(),
    )
    self._uplot_page.goto(url, wait_until='domcontentloaded')
    # Keep the native Viser DOM and React components in place. Only the
    # selected plot group is laid out for capture; other GUI and scene UI is
    # hidden. Ancestors must not clip a tall stack of panels to the viewport.
    self._uplot_page.add_style_tag(content="""
      [data-export-path] {
        display: block !important; position: static !important;
        transform: none !important; overflow: visible !important;
        width: 400px !important; height: auto !important;
        min-height: 0 !important; max-height: none !important;
        margin: 0 !important; padding: 0 !important;
      }
      [data-export-path] > :not([data-export-path]):not([data-export-root]) {
        display: none !important;
      }
      [data-export-root] {
        display: flex !important; flex-direction: column; gap: 8px;
        width: 400px !important; box-sizing: border-box;
        padding: 8px !important; background: white;
      }
      [data-export-root] .uplot-container {
        margin: 0 !important; padding: 0 !important; border: 0 !important;
        background: white !important;
      }
      [data-export-marker], [data-export-root] button,
      [data-export-root] .u-cursor-x, [data-export-root] .u-cursor-y,
      [data-export-root] .u-cursor-pt { display: none !important; }
    """)

  def _render_uplot(self, name: str, figure: UplotFigure) -> bytes:
    self._open_uplot_page()
    server, page = self._uplot_server, self._uplot_page
    self._uplot_frame += 1
    with server.atomic():
      plot = self._uplot_plots.get(name)
      if plot is None:
        marker_id = f'mujodico-export-{len(self._uplot_plots)}'
        folder = server.gui.add_folder('Plot export', expand_by_default=True)
        with folder:
          marker = server.gui.add_html('')
          handles = tuple(
              server.gui.add_uplot(
                  panel.data, panel.series, **figure.panel_options(i)
              )
              for i, panel in enumerate(figure.panels)
          )
        plot = dict(
            marker_id=marker_id, marker=marker, handles=handles, figure=figure
        )
        self._uplot_plots[name] = plot
      else:
        previous = plot['figure']
        for i, (handle, panel) in enumerate(
            zip(plot['handles'], figure.panels)
        ):
          if previous.panels[i].series != panel.series:
            handle.series = panel.series
          old_options = previous.panel_options(i)
          for key, value in figure.panel_options(i).items():
            if old_options.get(key) != value:
              setattr(handle, key, value)
          handle.data = panel.data
        plot['figure'] = figure
      # This message follows all chart updates in the same Viser batch. Wait
      # for its DOM marker, then for React/uPlot to finish the browser paint.
      plot['marker'].content = (
          f'<span id="{plot["marker_id"]}"'
          f' data-frame="{self._uplot_frame}"></span>'
      )
    server.flush()
    marker = page.locator(
        f'#{plot["marker_id"]}[data-frame="{self._uplot_frame}"]'
    )
    marker.wait_for(state='attached')
    heights = []
    for i in range(len(figure.panels)):
      options = figure.panel_options(i)
      legend_height = 24 if options['legend'].get('show', True) else 0
      heights.append(options['height'] + legend_height)
    marker.evaluate(
        """(marker, heights) => {
      document.querySelectorAll('[data-export-path], [data-export-root]').forEach(el => {
        el.removeAttribute('data-export-path');
        el.removeAttribute('data-export-root');
      });
      // The HTML marker and charts share the folder's GUI container.
      const root = marker.parentElement.parentElement;
      marker.parentElement.setAttribute('data-export-marker', '');
      root.setAttribute('data-export-root', '');
      root.style.minHeight = `${16 + heights.reduce((a, b) => a + b, 0) + 8 * (heights.length - 1)}px`;
      root.querySelectorAll('.uplot-container').forEach((el, i) => {
        el.style.minHeight = `${heights[i]}px`;
      });
      for (let el = root.parentElement; el; el = el.parentElement) {
        el.setAttribute('data-export-path', '');
      }
    }""",
        heights,
    )
    root = page.locator('[data-export-root]')
    page.wait_for_function(
        """count => {
      const charts = document.querySelectorAll('[data-export-root] .uplot');
      return charts.length === count && [...charts].every(el => el.querySelector('canvas'));
    }""",
        arg=len(figure.panels),
    )
    page.evaluate("""async () => {
      await document.fonts.ready;
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    }""")
    return self._capture_plot(name, root)

  def close(self):
    try:
      if self._browser is not None:
        self._browser.close()
    finally:
      self._browser = None
      self._page = None
      self._uplot_page = None
      try:
        if self._uplot_server is not None:
          self._uplot_server.stop()
      finally:
        self._uplot_server = None
        self._uplot_plots.clear()
        self._uplot_frame = 0
        if self._playwright is not None:
          self._playwright.stop()
          self._playwright = None
        self._kinds.clear()
        self._dimensions.clear()
