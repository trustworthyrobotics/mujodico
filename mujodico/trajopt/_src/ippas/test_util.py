import jax.numpy as jp

from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel


def simple_model(**kwargs):
  args = dict(
      N=3,
      nx=1,
      nu=1,
      initial_state=lambda p: p[:1],
      dynamics=lambda x, u, k, p: x + u,
      stage_cost=lambda x, u, k, p: 0.5 * jp.sum(u**2),
      terminal_cost=lambda x, p: 5 * jp.sum((x - p[1]) ** 2),
  )
  args.update(kwargs)
  return StagewiseOcpModel(**args)


def point_mass_model():
  import numpy as np

  from mujodico.trajopt._src.ippas.ip_algorithm import InitialGuess

  N = 99

  def dynamics(x, u, k, p):
    return (
        x
        + jp.concatenate((0.05 * x[2:], 0.05 * u))
        + k * jp.array([0.01, 0.02, 0.03, 0.04])
    )

  def inequalities(x, u, k, p):
    return jp.array([(u[0] - 0.05) / (0.55 + 0.1 * k), u[1] + 0.1 * k])

  m = StagewiseOcpModel(
      N,
      4,
      2,
      initial_state=lambda p: jp.array([-0.04, -0.02, -0.01, 0.0]),
      dynamics=dynamics,
      stage_cost=lambda x, u, k, p: jp.sum(u * u),
      terminal_cost=lambda x, p: jp.array(0.0),
      stage_inequality_constraint=inequalities,
      stage_inequality_bounds=(
          np.array([-1.0, -0.75]),
          np.array([1.0, np.inf]),
      ),
      params_trace=None,
  )
  m.initialize(None)
  xs = jp.arange(N + 1)[:, None] * jp.array([0.01, 0.02, 0.03, 0.04])
  us = jp.arange(N)[:, None] * jp.array([0.05, 0.06])
  return m, InitialGuess(xs, us)
