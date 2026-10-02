"""JAX/PAS FATROP method exposed through the examples runner."""

import time

import jax
import jax.numpy as jp
import numpy as np

from mujodico.examples.methods import support
from mujodico.examples.tasks import Task
from mujodico import mjx
from mujodico.trajopt._src.fatrop_ref import FatropRefOptions
from mujodico.trajopt._src.fatrop_ref import FatropReferenceJAX


class IppasLiftedMultipleShooting:

  def __init__(
      self,
      task: Task,
      *,
      gauss_newton: bool = True,
  ):
    if gauss_newton:
      hessian_mode = 'cost'
    else:
      hessian_mode = 'exact_lagrangian'

    self.task = task
    self.problem = ContactProblem(task)
    self.hessian_mode = hessian_mode

  def solve(
      self,
      traj_init: dict,
      *,
      iterations: int,
      stagewise: bool = False,
      gpu_line_search: bool = False,
      max_soc: int = 4,
  ):
    xs_init = traj_init['xs']
    us_init = traj_init['us']

    options = FatropRefOptions(
        max_iter=iterations,
        linear_backend='pas',
        hessian_mode=self.hessian_mode,
        # Match benchmark_fatrop_v64_one.py's ordinary-filter default.  The
        # watchdog remains available through the lower-level API.
        enable_watchdog=False,
        max_soc=max_soc,
        use_gpu_line_search=gpu_line_search,
    )

    tic = time.perf_counter()
    result = self.problem.ref.solve(
        xs_init, np.concatenate([
            us_init,
            np.zeros((self.task.horizon, self.problem.ncf), dtype=us_init.dtype),
        ], axis=1), options,
        stagewise=stagewise,
    )
    jax.block_until_ready(result.x)
    total_time = time.perf_counter() - tic

    print(f'''
Method: {self.__class__.__name__}
  total time: {total_time:.6f} sec
  solver status: {result.status}
  accepted steps: {result.accepted_steps}
''')

    return {
        'xs': np.asarray(result.x),
        'us': np.asarray(result.u[:, :self.task.nu]),
        'cfs': np.asarray(result.u[:, self.task.nu:]),
    }


class ContactProblem:
    """The exact lifted formulation used by CasADiFatropLiftedMultipleShooting.

    CasADi's variable bounds are represented by identity path rows here because
    FATROP's IPM internally introduces slacks and bound multipliers for them.
    """
    def __init__(self, task, dtype=jp.float64):
        self.task, self.nu, self.dtype = task, task.nu, dtype
        model = mjx.put_model(
            task.model, impl='jax', contact_approximation=task.contact_approx,
            contact_smoothing=task.contact_smoothing * 0.01)
        if dtype == jp.float32:
            def cast_leaf(value):
                if hasattr(value, 'dtype') and jp.issubdtype(value.dtype, jp.inexact):
                    return value.astype(jp.float32)
                return value
            model = jax.tree.map(cast_leaf, model)
        data = mjx.make_data(model, impl=model.impl)
        self.mm, self.md = model, data

        addresses, socs = support.extract_contact_indices(model, data)
        self.ncf = len(addresses)
        self.socs = socs

        x_goal = jp.asarray(task.x_goal, dtype=dtype)
        Q, Qf = jp.asarray(task.Q, dtype=dtype), jp.asarray(task.Qf, dtype=dtype)
        u_ref, R = jp.asarray(task.u_ref, dtype=dtype), jp.asarray(task.R, dtype=dtype)

        def forward(x, u):
            d = data.replace(qpos=x[:model.nq],
                             qvel=x[model.nq:model.nq + model.nv],
                             act=x[model.nq + model.nv:], ctrl=u)
            return mjx.forward(model, d)._impl.efc_force[addresses]

        def advance(x, u, contact_force):
            force = jp.zeros(data._impl.nefc, dtype=x.dtype).at[addresses].set(contact_force)
            return support.step_given_efc_force(model, x, u, force)

        self.forward, self.advance = jax.jit(forward), jax.jit(advance)
        self.physical_step = jax.jit(lambda x, u: (advance(x, u, forward(x, u)), forward(x, u)))

        def stage_cost(x, augmented_u):
            dx, du = x - x_goal, augmented_u[:task.nu] - u_ref
            return dx @ (Q @ dx) + du @ (R @ du)

        def terminal_cost(x):
            dx = x - x_goal
            return dx @ (Qf @ dx)

        def transition(x, augmented_u):
            return advance(x, augmented_u[:task.nu], augmented_u[task.nu:])

        def stage_equality(x, augmented_u):
            return augmented_u[task.nu:] - forward(x, augmented_u[:task.nu])

        self.stage_cost, self.terminal_cost = stage_cost, terminal_cost
        self.transition, self.stage_eq = transition, stage_equality

        control_lb, control_ub = task.u_bounds
        np_dtype = np.float32 if dtype == jp.float32 else np.float64
        state_lb, state_ub = task.x_bounds
        state_bounded = np.nonzero(
            np.isfinite(state_lb) | np.isfinite(state_ub))[0].astype(np.int32)
        force_lb = np.zeros(self.ncf, dtype=np_dtype)
        force_ub = np.full(self.ncf, np.inf, dtype=np_dtype)
        for soc in socs:
            force_lb[soc.start_index + 1:soc.start_index + soc.dim] = -np.inf
        augmented_lb = np.concatenate([control_lb, force_lb])
        augmented_ub = np.concatenate([control_ub, force_ub])
        bounded = np.nonzero(np.isfinite(augmented_lb) | np.isfinite(augmented_ub))[0].astype(np.int32)
        self.bounded = bounded
        # CasADi FATROP receives x/u bounds as variable bounds.  The JAX
        # reference represents those same bounds as identity path rows so the
        # structured solver can handle them through its existing slack/bound
        # machinery.  State rows must precede control rows only by convention;
        # their ordering is otherwise immaterial.
        inequality_lb = [state_lb[i] for i in state_bounded]
        inequality_ub = [state_ub[i] for i in state_bounded]
        inequality_lb += [augmented_lb[i] for i in bounded]
        inequality_ub += [augmented_ub[i] for i in bounded]
        for _ in socs:
            inequality_lb.append(-np.inf)
            inequality_ub.append(0.0)
        inequality_lb = np.asarray(inequality_lb, dtype=np_dtype)
        inequality_ub = np.asarray(inequality_ub, dtype=np_dtype)
        bounded_jax = jp.asarray(bounded)

        state_bounded_jax = jp.asarray(state_bounded)

        def stage_inequality(x, augmented_u):
            values = []
            if state_bounded.size:
                values.append(x[state_bounded_jax])
            if bounded.size:
                values.append(augmented_u[bounded_jax])
            contact_force = augmented_u[task.nu:]
            for soc in socs:
                normal = contact_force[soc.start_index]
                tangent = (contact_force[soc.start_index + 1:soc.start_index + soc.dim]
                           / soc.friction)
                values.append(jp.asarray([tangent @ tangent - normal * normal]))
            return jp.concatenate(values) if values else jp.zeros((0,), dtype=augmented_u.dtype)

        self.stage_ineq = stage_inequality
        self.ineq_lb, self.ineq_ub = inequality_lb, inequality_ub
        self.ref = FatropReferenceJAX(
            horizon=task.horizon, num_states=task.nx,
            num_controls=task.nu + self.ncf,
            x0=jp.asarray(task.x_0, dtype=dtype), stage_cost=stage_cost,
            terminal_cost=terminal_cost, transition=transition,
            stage_equality=stage_equality, stage_inequality=stage_inequality,
            ineq_lb=inequality_lb, ineq_ub=inequality_ub)
