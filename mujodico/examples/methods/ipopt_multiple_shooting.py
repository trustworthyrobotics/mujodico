import cyipopt
import jax
import jax.numpy as jp
from mujodico import mjx
from mujodico.examples.tasks import Task
import numpy as np


def solve(task: Task):
  model = task.model

  mjx_smooth_model = mjx.put_model(
      model,
      contact_approximation="log barrier",
      contact_smoothing=1e1,
  )

  mjx_model = mjx.put_model(
      model,
      contact_approximation="log barrier",
      contact_smoothing=0.0,
  )

  mjx_data = mjx.make_data(mjx_model)

  nq = model.nq
  nx = nq + model.nv
  nu = model.nu

  @jax.jit
  def step(x, u):
    d = mjx_data.replace(qpos=x[:nq], qvel=x[nq:], ctrl=u)
    d0 = mjx.step(mjx_model, d)
    xnext = jp.concatenate((d0.qpos, d0.qvel))
    d1 = mjx.step(mjx_smooth_model, d)
    xnext_smooth = jp.concatenate((d1.qpos, d1.qvel))
    return (
        jax.lax.stop_gradient(xnext)
        + xnext_smooth
        - jax.lax.stop_gradient(xnext_smooth)
    )

  N = task.horizon
  n_vars = (N + 1) * nx + N * nu

  def pack(xs: jax.Array | np.ndarray, us: jax.Array | np.ndarray):
    assert xs.shape == (N + 1, nx)
    assert us.shape == (N, nu)
    cp = jp if isinstance(xs, jax.Array) else np

    return cp.concatenate([
        cp.hstack((xs[:-1], us)).ravel(),
        xs[-1],
    ])

  def unpack(vars: jax.Array | jax.Array):
    assert vars.shape == (n_vars,)
    cp = jp if isinstance(vars, jax.Array) else np

    xus = vars[:-nx].reshape(N, nx + nu)
    xs = cp.vstack((xus[:, :nx], vars[-nx:]))
    us = xus[:, nx:]

    return xs, us

  x_goal = jp.array(task.x_goal)
  Q = jp.array(task.Q)
  Qf = jp.array(task.Qf)
  u_ref = jp.array(task.u_ref)
  R = jp.array(task.R)

  @jax.jit
  def objective(z):
    xs, us = unpack(z)

    dx = xs[:-1] - x_goal
    dxf = xs[-1] - x_goal
    du = us - u_ref

    cost = jp.sum((dx @ Q) * dx) + jp.sum((du @ R) * du) + dxf @ Qf @ dxf
    return cost

  @jax.jit
  def constraints(z):
    xs, us = unpack(z)
    xs_next = jax.vmap(step)(xs[:-1], us)
    return (xs[1:] - xs_next).ravel()

  objective_grad = jax.jit(jax.grad(objective))
  constraint_jac = jax.jit(jax.jacfwd(constraints))

  class NLP:
    def objective(self, z):
      return float(objective(jp.asarray(z)))

    def gradient(self, z):
      return np.asarray(objective_grad(jp.asarray(z)))

    def constraints(self, z):
      return np.asarray(constraints(jp.asarray(z)))

    def jacobian(self, z):
      return np.asarray(constraint_jac(jp.asarray(z)).reshape(-1))

  x_lb, x_ub = task.x_bounds
  u_lb, u_ub = task.u_bounds

  xs_lb = np.repeat(x_lb[None, :], N + 1, axis=0)
  xs_lb[0] = task.x_0
  xs_ub = np.repeat(x_ub[None, :], N + 1, axis=0)
  xs_ub[0] = task.x_0

  us_lb = np.repeat(u_lb[None, :], N, axis=0)
  us_ub = np.repeat(u_ub[None, :], N, axis=0)

  lb = pack(xs_lb, us_lb)
  ub = pack(xs_ub, us_ub)

  cl = np.zeros(N * nx)
  cu = np.zeros(N * nx)

  problem = cyipopt.Problem(
      n=n_vars,
      m=N * nx,
      problem_obj=NLP(),
      lb=lb,
      ub=ub,
      cl=cl,
      cu=cu,
  )

  problem.add_option("max_iter", 100)

  xs_init = np.linspace(task.x_0, task.x_goal, N + 1)
  us_init = np.zeros((N, nu))

  vars_init = np.concatenate([xs_init.ravel(), us_init.ravel()])

  vars_opt, info = problem.solve(vars_init)

  xs_opt, us_opt = unpack(vars_opt)

  return xs_opt, us_opt
