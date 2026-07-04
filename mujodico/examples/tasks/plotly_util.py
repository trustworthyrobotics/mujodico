import numpy as np

try:
  import plotly.graph_objects as go
except ImportError as e:
  raise ImportError(
      "Optional visualization dependencies are not installed. "
      "Install them with:\n\n"
      "    pip install plotly"
  ) from e


def add_circle(fig: go.Figure, x, y, r, color="black", name=None, **kwargs):
  t = np.linspace(0, 2 * np.pi)
  fig.add_trace(
      go.Scatter(
          x=x + r * np.cos(t),
          y=y + r * np.sin(t),
          fill="toself",
          mode="lines",
          line=dict(color=color),
          fillcolor=color,
          name=name,
          showlegend=False if name is None else True,
          **kwargs,
      )
  )


def add_rectangle(
    fig: go.Figure, x, y, theta, w, h, color="black", name=None, **kwargs
):
  corners = np.array([
      [-w / 2, w / 2, w / 2, -w / 2, -w / 2],
      [-h / 2, -h / 2, h / 2, h / 2, -h / 2],
  ]).T

  c, s = np.cos(theta), np.sin(theta)
  R = np.array([[c, -s], [s, c]])

  pts = corners @ R.T
  pts[:, 0] += x
  pts[:, 1] += y

  fig.add_trace(
      go.Scatter(
          x=pts[:, 0],
          y=pts[:, 1],
          fill="toself",
          mode="lines",
          line=dict(color=color),
          fillcolor=color,
          name=name,
          showlegend=False if name is None else True,
          **kwargs,
      )
  )
