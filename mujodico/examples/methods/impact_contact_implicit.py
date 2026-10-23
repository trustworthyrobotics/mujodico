import jax
import jax.numpy as jp
from mujodico import mjx
from mujodico.examples import methods
from mujodico.examples.tasks import Task
import numpy as np


def _cost_factor(weight):
  """Returns L such that L.T @ L equals a positive semidefinite weight."""
  eigenvalues, eigenvectors = np.linalg.eigh(weight)
  if np.any(eigenvalues < -1e-10):
    raise ValueError('IMPACT requires positive semidefinite cost weights.')
  return jp.asarray(
      np.sqrt(np.maximum(eigenvalues, 0))[:, None] * eigenvectors.T
  )


def _make_stage(trans, task):
  from impact.jax import StageProblem

  Q = _cost_factor(task.Q)
  Qf = _cost_factor(task.Qf)
  R = _cost_factor(task.R)
  quaternion_w_idxs = task.quaternion_w_idxs

  def state_residual(x, goal, factor, weight):
    error = x - goal
    rotations = []
    for i in quaternion_w_idxs:
      error = error.at[i : i + 4].set(0.0)
      q = mjx._src.math.normalize(x[i : i + 4])
      q_goal = mjx._src.math.normalize(goal[i : i + 4])
      scale = jp.sqrt(jp.mean(jp.diag(weight[i : i + 4, i : i + 4])))
      # Squared norm is weight * (1 - (q.T @ q_goal)**2), as in
      # util.state_cost, with finite derivatives even at the goal.
      rotations.append(scale * (q - jp.dot(q, q_goal) * q_goal))
    return jp.concatenate([factor @ error, *rotations])

  x_lb, x_ub = (jp.asarray(bound) for bound in trans.x_bounds)
  w_lb, w_ub = (jp.asarray(bound) for bound in trans.w_bounds)
  x_lb_idx, x_ub_idx = (
      np.flatnonzero(np.isfinite(bound)) for bound in trans.x_bounds
  )
  # Nonnegativity of the contact variables is already enforced by G.
  w_lb_idx, w_ub_idx = (
      np.flatnonzero(np.isfinite(bound[: task.nu])) for bound in trans.w_bounds
  )

  class ContactImplicitStage(StageProblem):
    state_dim = property(lambda self: trans.nx)
    control_dim = property(lambda self: trans.nw)
    comp_dim = property(lambda self: trans.local_complementarity.size)
    ineq_dim = property(
        lambda self: (
            x_lb_idx.size + x_ub_idx.size + w_lb_idx.size + w_ub_idx.size
        )
    )
    time_step = property(lambda self: float(task.model.opt.timestep))
    cost_is_linear = property(lambda self: not quaternion_w_idxs)

    def step(self, x, w, p):
      return trans.dynamics(x, w)

    def G(self, x, w, p):
      return w[task.nu :]

    def H(self, x, w, p):
      return trans.stage_inequality(x, w)

    def ineq(self, x, w, p):
      # IMPACT uses <= 0. Bound the integrated state to cover knots 1..N,
      # including the terminal knot; knot 0 is fixed to the task's valid x_0.
      x_next = self.step(x, w, p) if x_lb_idx.size + x_ub_idx.size else x
      return jp.concatenate((
          x_lb[x_lb_idx] - x_next[x_lb_idx],
          x_next[x_ub_idx] - x_ub[x_ub_idx],
          w_lb[w_lb_idx] - w[w_lb_idx],
          w[w_ub_idx] - w_ub[w_ub_idx],
      ))

    def cost_residual(self, X, W, p):
      params = trans.params
      states = jax.vmap(
          lambda x: state_residual(x, params.x_goal, Q, params.Q)
      )(X[:-1])
      controls = (W[:, : task.nu] - params.u_ref) @ R.T
      terminal = state_residual(X[-1], params.x_goal, Qf, params.Qf)
      return jp.concatenate(
          (states.reshape(-1), controls.reshape(-1), terminal)
      )

  return ContactImplicitStage()


class ImpactContactImplicit:
  """Solves stage contact laws at x_k, then integrates their forces to x_{k+1}.

  ``config`` exposes IMPACT's AuLa settings. ``solve`` sets its horizon,
  initial state, iteration limit and print level from the example task/arguments.
  """

  def __init__(self, task: Task, *, jit: bool = True):
    try:
      from impact import AulaConfig
      from impact.jax import MultipleShootingSolver
    except ImportError as e:
      raise ImportError(
          'Optional solver depedency not installed. Install IMPACT with\n\n'
          '   pip install git+https://github.com/wei-chen-li/IMPACT.git'
      ) from e

    self.transcription = methods.ContactImplicitTranscription(
        task, contact_at_next_state=False
    )
    self.config = AulaConfig()
    self._solver_type = MultipleShootingSolver
    self._jit = jit
    self.update(task)

  def update(self, task: Task):
    self.transcription.update(task)
    self._nu = task.nu
    self.stage = _make_stage(self.transcription, task)
    self.solver = self._solver_type(
        self.stage, stage_factory=lambda stage, config: stage, jit=self._jit
    )

  def solve(
      self,
      traj_init: dict,
      *,
      iterations: int = 500,
      verbose: bool = True,
  ):
    trans = self.transcription
    nu = self._nu
    xs_init = np.asarray(traj_init['xs'])
    us_init = np.asarray(traj_init['us'])
    cfs_init = traj_init.get('cfs', np.zeros((trans.N, trans.nw - nu)))
    ws_init = np.hstack((us_init, cfs_init))

    config = self.config
    config.horizon = trans.N
    config.x_0 = np.asarray(trans.params.x_0)
    config.max_outer_iters = iterations
    config.print_level = 1 if verbose else 0
    # IMPACT takes component-first guesses and returns component-first results.
    solution = self.solver.solve_with_initial_guess(
        config, xs_init.T, ws_init.T
    )
    xs_opt = np.asarray(solution.state_trajectory).T
    ws_opt = np.asarray(solution.control_trajectory).T

    if verbose:
      print(
          f'Method: {self.__class__.__name__}\n'
          f'  status: {solution.status_message}\n'
          f'  solve time: {solution.solve_time:.6f} sec\n'
          f'  outer iterations: {solution.outer_iterations}'
      )

    return {
        'xs': xs_opt,
        'us': ws_opt[:, :nu],
        'cfs': ws_opt[:, nu:],
        'info': solution,
    }
