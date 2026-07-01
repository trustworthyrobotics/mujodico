import jax
import jax.numpy as jp
from mujodico import mjx
from mujodico.examples.tasks import Task
import numpy as np

try:
  import cyipopt
except ImportError as e:
  raise ImportError(
      "Optional solver dependency not installed. Install it with:\n\n"
      "    pip install cyipopt"
  ) from e


class IpoptSingleShooting:

  def __init__(self, task: Task, contact_approx: mjx.ContactApproxType):
    model = task.model

    mjx_smooth_model = mjx.put_model(
        model,
        contact_approximation=contact_approx,
        contact_smoothing=task.contact_smoothing,
    )

    mjx_model = mjx.put_model(
        model,
        contact_approximation=contact_approx,
        contact_smoothing=0.0,
    )

    mjx_data = mjx.make_data(mjx_model)

    N = task.horizon
    nu = task.nu

    x_0 = jp.array(task.x_0)
    x_goal = jp.array(task.x_goal)
    Q = jp.array(task.Q)
    Qf = jp.array(task.Qf)
    u_ref = jp.array(task.u_ref)
    R = jp.array(task.R)

    @jax.jit
    def step(x, u):
      nq, nv = mjx_model.nq, mjx_model.nv
      d = mjx_data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)
      d0 = mjx.step(mjx_model, d)
      xnext = jp.concatenate((d0.qpos, d0.qvel, d0.act))
      d1 = mjx.step(mjx_smooth_model, d)
      xnext_smooth = jp.concatenate((d1.qpos, d1.qvel, d1.act))
      return (
          jax.lax.stop_gradient(xnext)
          + xnext_smooth
          - jax.lax.stop_gradient(xnext_smooth)
      )

    @jax.jit
    def rollout(x_0, us):
      def scan_step(x, u):
        x_next = step(x, u)
        return x_next, x_next

      _, xs = jax.lax.scan(scan_step, x_0, us)
      return jp.concatenate([x_0[None], xs], axis=0)

    @jax.jit
    def objective(vars):
      us = vars.reshape(N, nu)
      xs = rollout(x_0, us)

      dx = xs[:-1] - x_goal
      dxf = xs[-1] - x_goal
      du = us - u_ref

      cost = jp.sum((dx @ Q) * dx) + jp.sum((du @ R) * du) + dxf @ Qf @ dxf
      return cost

    objective_grad = jax.jit(jax.grad(objective))

    class NLP:

      def objective(self, vars):
        return float(objective(jp.asarray(vars)))

      def gradient(self, vars):
        return np.asarray(objective_grad(jp.asarray(vars)))

    u_lb, u_ub = task.u_bounds
    us_lb = np.repeat(u_lb[None, :], N, axis=0)
    us_ub = np.repeat(u_ub[None, :], N, axis=0)

    vars_lb = us_lb.ravel()
    vars_ub = us_ub.ravel()

    problem = cyipopt.Problem(
        n=N * nu,
        m=0,
        problem_obj=NLP(),
        lb=vars_lb,
        ub=vars_ub,
    )

    self.task = task
    self.rollout = rollout
    self.problem = problem

  def solve(
      self,
      *,
      us_init: np.ndarray,
      iterations: int,
      **kwargs,
  ):
    N, nu = self.task.horizon, self.task.nu
    assert us_init.shape == (N, nu)

    vars_init = us_init.ravel()

    self.problem.add_option("max_iter", iterations)

    us_opt, info = self.problem.solve(vars_init)

    us_opt = jp.array(us_opt).reshape(N, nu)

    x_0 = jp.array(self.task.x_0)
    xs_opt = self.rollout(x_0, us_opt)

    return np.asarray(xs_opt), np.asarray(us_opt)
