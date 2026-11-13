from dataclasses import dataclass, field
from typing import Any

import numpy as np
from viser import uplot


@dataclass(frozen=True)
class UplotPanel:
  data: tuple[np.ndarray, ...]
  series: tuple[uplot.Series, ...]
  options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UplotFigure:
  panels: tuple[UplotPanel, ...]

  def __post_init__(self):
    if not self.panels:
      raise ValueError('A uPlot figure needs at least one panel.')

  def panel_options(self, index: int) -> dict[str, Any]:
    panel = self.panels[index]
    options = {'legend': {'show': False}, **panel.options}
    # uPlot's height includes axes, but excludes the HTML legend.
    height = 150 if len(self.panels) > 1 else 284
    if len(self.panels) == 1 and options['legend'].get('show', True):
      height -= 24
    return {**options, 'height': height}


@dataclass(frozen=True)
class ReferenceLine:
  value: float
  label: str = 'Goal'
  color: str = 'black'
  dash: tuple[float, ...] = (6, 4)


def time_series_panel(
    ts,
    values,
    count: int,
    *,
    ylabel: str,
    labels: tuple[str, ...] | None = None,
    colors: tuple[str, ...] | None = None,
    references: tuple[ReferenceLine, ...] = (),
    xlabel: str = 'Time (sec)',
    tick_step: float | None = None,
) -> UplotPanel:
  """Reveal a prefix while keeping full-horizon references and fixed ranges.

  Trailing NaNs suppress future samples in Viser's Float64Array transport.
  References have finite values over the entire time axis, including frame 0.
  """
  ts = np.asarray(ts, dtype=np.float64)
  values = np.asarray(values, dtype=np.float64)
  if values.ndim == 1:
    values = values[:, None]
  count = min(max(count, 0), len(values))
  references = tuple(ref for ref in references if np.isfinite(ref.value))
  data = [ts]
  series = [uplot.Series(label='Time')]
  for i in range(values.shape[1]):
    visible = np.full(len(ts), np.nan)
    visible[:count] = values[:count, i]
    data.append(visible)
    series.append(
        uplot.Series(
            label=labels[i] if labels else ylabel,
            stroke=colors[i] if colors else 'blue',
            width=1.5,
            points={'show': False},
        )
    )
  for ref in references:
    data.append(np.full(len(ts), ref.value, dtype=np.float64))
    series.append(
        uplot.Series(
            label=ref.label,
            stroke=ref.color,
            dash=ref.dash,
            width=1.5,
            points={'show': False},
        )
    )

  finite = values[np.isfinite(values)]
  extrema = [*finite, *(ref.value for ref in references)]
  lower, upper = (min(extrema), max(extrema)) if extrema else (0.0, 0.0)
  padding = max(upper - lower, 1e-5) * 0.05
  yaxis = dict(
      label=ylabel,
      labelSize=18,
      labelFont='11px sans-serif',
      font='11px sans-serif',
      size=60,
  )
  if tick_step is not None:
    # uPlot chooses an increment that meets the available pixel spacing.
    # Supplying only one increment can leave the axis without any ticks.
    yaxis['incrs'] = tuple(tick_step * 2**i for i in range(16))
    yaxis['space'] = 20
  return UplotPanel(
      data=tuple(data),
      series=tuple(series),
      options=dict(
          scales={
              'x': {
                  'time': False,
                  'range': (float(ts[0]), float(max(ts[-1], ts[0] + 1e-6))),
              },
              'y': {'range': (float(lower - padding), float(upper + padding))},
          },
          axes=(
              {
                  'label': xlabel,
                  'labelSize': 18,
                  'font': '11px sans-serif',
                  'size': 32,
              },
              yaxis,
          ),
          legend={'show': labels is not None, 'live': False},
          padding=(8, 16, 0, 0),
      ),
  )
