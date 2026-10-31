# IPPAS

IPPAS is a JAX primal-dual interior-point solver for nonlinear, stagewise
optimal-control problems. It retains the FATROP algorithm components while
replacing its trajectory linear algebra with the serial and parallel
associative-scan boundary formulation from `IP-PAS.tex`.

The implementation supports infeasible equality starts, native variable bounds,
two-sided nonlinear constraints, terminal constraints, exact retraction-coordinate
Lagrangian derivatives, filter backtracking, second-order correction, watchdog
recovery, monotone barrier updates, curvature/regularity correction, iterative
refinement, and feasibility restoration. Dense and graph NLP backends and the
separate Mehrotra QP solver are outside this package.

## Use

Construct the solver once with static structure. Pass changing numerical data
as a JAX parameter PyTree to every solve:

```python
import jax
import jax.numpy as jnp
from mujodico.trajopt.ippas import InitialGuess, IppasSolver, Options

jax.config.update('jax_enable_x64', True)

solver = IppasSolver(
    horizon=20, num_states=1, num_controls=1,
    initial_state=lambda p: p['x0'],
    dynamics=lambda x, u, p: x + u,
    stage_cost=lambda x, u, p: 0.5 * jnp.sum(u * u),
    terminal_cost=lambda x, p: 5.0 * jnp.sum((x - p['goal'])**2),
    u_bounds=(-1.0, 1.0),
    options=Options(linear_backend='parallel'),
)
compiled_solve = jax.jit(solver.solve)
guess = InitialGuess(jnp.zeros((21, 1)), jnp.zeros((20, 1)))
params = {'x0': jnp.array([0.0]), 'goal': jnp.array([2.0])}
result = compiled_solve(params, guess)
jax.block_until_ready(result.xs)
```

Callbacks use `stage_cost(x, u, params)`, `dynamics(x, u, params)`,
`stage_equality(x, u, params)`, and `stage_inequality(x, u, params)`.
Terminal callbacks receive `(x, params)`. Inequality callbacks return constraint
values; pass their lower and upper bounds using `stage_inequality_bounds` and
`terminal_inequality_bounds`. A missing constraint callback represents an empty
block. Initial-state values come from `initial_state(params)`.

Changing parameter values with the same PyTree structure, leaf shapes, and dtypes
reuses the compiled solve. Horizon, dimensions, callbacks, finite-bound topology,
and `Options` are static. Results and bounded trace storage are PyTrees; numeric
termination codes are defined by `convergence.Status`. Rendering and host-side
reporting happen after the compiled solve.

Arithmetic follows the floating dtype returned by `initial_state`; callback
outputs are converted to that precision. Use float64 for tight KKT tolerances.
JIT compatibility and dynamic parameters do not imply an implicit derivative of
the optimized solution: the solver does not provide an argmin sensitivity API.

## Numerical conventions

- Running stages have uniform state/control dimensions and constraint structure;
  the terminal stage has separate callbacks and dimensions. Bounds may be finite
  on either side, and fixed components are handled as equalities rather than
  zero-width barrier intervals.
- `linear_backend='serial'` uses sequential scans. `'parallel'` (also `'pas'`)
  uses associative scans. Both represent the same regularized Newton system.
- The document assumes invertible local control/equality blocks. The solver
  extends those blocks with explicit dual regularization when needed, including
  state-only equality constraints. Regularization is included consistently in
  residual checks; no local pseudoinverse silently substitutes a different solve.
  A precision-scaled floor keeps dual damping distinguishable from zero in
  float32. The actual primal and dual shifts are recorded in the trace.
- Curvature and factor-regularity checks can be more conservative than FATROP's
  generalized Riccati pivot checks. Numerical equality with native FATROP's
  iteration sequence is therefore not promised. A regularized linear step is
  still tested against the original nonlinear constraints before acceptance.
- SOC and refinement reuse cached matrix factors and solve new right-hand sides.
  SOC addresses nonlinear constraint curvature; it is separate from Mehrotra
  predictor-corrector centering.
- `hessian_mode='exact_lagrangian'` includes constraint curvature.
  `'cost'` explicitly requests cost-only curvature. Arbitrary scalar costs do not
  automatically supply least-squares Gauss-Newton residuals.

`manifold.py` provides `Euclidean`, `Quaternion`, `ProductManifold`, and
`CustomManifold`. Quaternions use Hamilton `[w, x, y, z]` storage and three
right/body-frame tangent coordinates. All trial, SOC, and restoration updates
use retractions. Native bounds apply to genuinely additive coordinates; finite
bounds on nonadditive storage coordinates are routed through nonlinear
constraints. Quaternion local coordinates use the shortest rotation and exclude
the usual rotation-by-pi chart singularity.

The manifold Newton assembly uses an auxiliary next-state tangent copy to retain
physical defect Jacobians and adjacent-state curvature in a stagewise KKT
representation. This avoids assuming that every manifold defect already has
next-state Jacobian `-I`.

## Components and tests

Files follow FATROP's component boundaries. Features belonging to one native
component stay together in the JAX implementation:

FATROP paths below are relative to the FATROP checkout root. They identify
corresponding native components; the associative-scan algebra in
`linear_solver.py` comes from `IP-PAS.tex` and replaces the native Riccati solver.

| JAX module | FATROP source files | Responsibilities |
| --- | --- | --- |
| `solver.py` | `include/fatrop/ip_algorithm/ip_alg_builder.hxx`<br>`include/fatrop/ip_algorithm/ip_algorithm.hxx`<br>`include/fatrop/ip_algorithm/ip_iteration_output.hxx` | Construction, main interior-point loop, results and traces |
| `problem.py` | `include/fatrop/ocp/ocp_abstract.hpp`<br>`include/fatrop/ocp/nlp_ocp.hxx`<br>`include/fatrop/ip_algorithm/ip_nlp_orig.hxx` | OCP interface, evaluations and tangent derivatives |
| `iterate.py` | `include/fatrop/ip_algorithm/ip_iterate.hxx`<br>`include/fatrop/ip_algorithm/ip_data.hxx` | Primal-dual iterates, bounds, complementarity and residual metrics |
| `initialization.py` | `include/fatrop/ip_algorithm/ip_initializer.hxx`<br>`include/fatrop/ip_algorithm/ip_eq_mult_initializer.hxx` | Interior initialization and equality-multiplier seeds |
| `barrier.py` | `include/fatrop/ip_algorithm/ip_mu_update.hxx` | Monotone barrier-parameter updates |
| `convergence.py` | `include/fatrop/ip_algorithm/ip_convergence_check.hxx` | KKT convergence and acceptable-iterate checks |
| `filter.py` | `include/fatrop/ip_algorithm/ip_filter.hpp`<br>`src/ip_algorithm/ip_filter.cpp` | Filter storage, acceptance and updates |
| `search_direction.py` | `include/fatrop/ip_algorithm/ip_search_dir.hxx`<br>`include/fatrop/linear_algebra/linear_solver.hxx` | Newton direction, primal/dual inertia correction and refinement |
| `line_search.py` | `include/fatrop/ip_algorithm/ip_linesearch.hxx` | Filter/Armijo backtracking, SOC, watchdog, tiny steps and soft restoration |
| `restoration.py` | `include/fatrop/ip_algorithm/ip_nlp_resto.hxx`<br>`include/fatrop/ip_algorithm/ip_resto_phase_min_cl1.hxx`<br>`include/fatrop/ip_algorithm/ip_initializer_resto.hxx`<br>`include/fatrop/ip_algorithm/ip_convergence_check_resto.hxx` | Elastic restoration NLP, initialization, proximity terms and phase controller |
| `linear_solver.py` | `src/ocp/aug_system_solver.cpp`<br>`src/ocp/pd_solver_orig.cpp`<br>`src/ocp/pd_solver_resto.cpp` | Condensation, replacement serial/parallel scans, reusable factors and direction recovery |
| `manifold.py` | `include/fatrop/ocp/ocp_abstract.hpp`<br>`examples/so3_helpers.hpp`<br>`examples/ocp/quaternion_lie_multistep_example.cpp` | Retraction callbacks and quaternion geometry; generic product/custom helpers are JAX additions |
| `options.py` | `include/fatrop/common/options.hpp` | Solver configuration; immutable options replace the native registry |

Trace entries have a `valid` mask and include residuals, barrier parameters,
primal/dual step lengths, regularization, factorization/refinement counts, SOC
attempts, and restoration iterations. `step_type` is 0 for initialization or a
failed step, 1 for ordinary line search, 2 for SOC, 3 for soft restoration,
4 for hard restoration, 5 for a watchdog trial, and 6 for watchdog rollback.

Tests cover policy
transitions and failure paths, dense KKT residuals, backend agreement, factor
reuse, terminal constraints, and manifold derivatives.

From the repository root, run:

```bash
JAX_PLATFORMS=cpu /home/wei-chen/miniforge3/envs/mujodico/bin/python \
  -m unittest discover -s mujodico/trajopt/ippas/tests -p 'test_*.py' -v
```

For an installed package, use the same command with its test directory. To test
GPU execution, select a GPU-capable JAX environment and set `JAX_PLATFORMS=cuda`.
CPU test success does not establish GPU performance; measure compilation and
steady-state solve times separately.

The example method `IppasLiftedMultipleShooting` keeps the existing `xs/us/cfs`
interface, passes native bounds directly, constructs a quaternion product from
the task metadata, and caches a compiled solver for each static option set.
The legacy `gpu_line_search` argument remains accepted; all globalization paths
are compiled for either value. Its `gauss_newton=True` flag preserves the
historical cost-only Hessian selection.

See `NOTICE.md` and `licenses/` for retained upstream notices and license texts.
