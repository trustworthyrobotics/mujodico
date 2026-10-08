import importlib

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mujodico.trajopt._src import abstract_nlp_model


class NonnegativeQuadraticNLP(abstract_nlp_model.AbstractNLPModel):
  """A mathematical program:
    minimize    (x₁ − 2)² + (x₂ − 1)²
    subject to  0 ≤ x₁ ≤ 1
                0 ≤ x₂ ≤ 1
                x₁ ⟂ x₂
  Without complementarity, the solution is (1, 1), with objective 1.
  With x₁ ⟂ x₂, the global solution is (1, 0), with objective 2.
  """

  @property
  def num_vars(self):
    return 2

  @property
  def var_bounds(self):
    return np.zeros(2), np.ones(2)

  @property
  def num_constraints(self):
    return 0

  @property
  def constraint_bounds(self):
    return np.empty(0), np.empty(0)

  def objective(self, x):
    return float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2)

  def gradient(self, x):
    return 2.0 * (np.asarray(x) - np.array([2.0, 1.0]))

  def constraints(self, x):
    del x
    return np.empty(0)

  def jacobian_structure(self):
    return []

  def jacobian_entries(self, x):
    del x
    return np.empty(0)

  def hessian_structure(self):
    return [(0, 0), (1, 1)]

  def hessian_entries(self, x, lagrange, objective_factor):
    del x, lagrange
    return np.full(2, 2.0 * objective_factor)


class AbstractNLPModelTest(parameterized.TestCase):

  @parameterized.parameters('NLPModelsIpopt', 'MadNLP')
  def test_solve_nlp(self, solver_package):
    try:
      juliacall = importlib.import_module('juliacall')
      jl = juliacall.Main
    except Exception as e:
      self.skipTest(f'juliacall/Julia is unavailable: {e}')

    try:
      jl.seval(f'using {solver_package}')
    except Exception as e:
      self.skipTest(f'{solver_package} is not installed in Julia: {e}')

    julia_nlp = NonnegativeQuadraticNLP().to_nlpmodels_jl()

    if solver_package == 'NLPModelsIpopt':
      stats = jl.NLPModelsIpopt.ipopt(julia_nlp, print_level=0)
    elif solver_package == 'MadNLP':
      stats = jl.MadNLP.madnlp(julia_nlp, print_level=5)

    np.testing.assert_allclose(stats.solution, [1.0, 1.0], atol=1e-4)
    np.testing.assert_allclose(stats.objective, 1.0, atol=1e-6)

  def test_solve_nlp_ipopt(self):
    try:
      importlib.import_module('cyipopt')
    except Exception as e:
      self.skipTest(f'cyipopt is unavailable: {e}')

    cyipopt_nlp_problem = NonnegativeQuadraticNLP().to_cyipopt()
    cyipopt_nlp_problem.add_option('print_level', 0)

    x0 = [0.0, 0.0]
    _, stats = cyipopt_nlp_problem.solve(x0)

    np.testing.assert_allclose(stats['x'], [1.0, 1.0], atol=1e-4)
    np.testing.assert_allclose(stats['obj_val'], 1.0, atol=1e-6)

  @parameterized.parameters('RelaxationSolver', 'PenaltySolver', 'Crossover')
  def test_solve_mpcc(self, solver_method):
    try:
      juliacall = importlib.import_module('juliacall')
      jl = juliacall.Main
    except Exception as e:
      self.skipTest(f'juliacall/Julia is unavailable: {e}')

    try:
      jl.seval('using CCOpt; using MPCCModels')
    except Exception as e:
      self.skipTest(f'CCOpt is not installed in Julia: {e}')

    julia_nlp = NonnegativeQuadraticNLP().to_nlpmodels_jl()

    ind_x1 = jl.collect(jl.Int64, [1])
    ind_x2 = jl.collect(jl.Int64, [2])
    mpcc = jl.MPCCModels.MPCCModel(julia_nlp, ind_x1, ind_x2)

    if solver_method == 'Crossover':
      print_options = jl.seval('(print_level=5,)')
      options = jl.CCOpt.CrossoverOptions(
          lpcc_solver_kwargs=print_options,
          bnlp_solver_kwargs=print_options,
      )
      stats = jl.CCOpt.crossover(mpcc, julia_nlp.meta.x0, options)
    else:
      Solver = getattr(jl.CCOpt, solver_method)
      solver = Solver(mpcc, print_level=5)
      stats = getattr(jl.CCOpt, 'solve_homotopy!')(solver)

    np.testing.assert_allclose(stats.solution, [1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(stats.objective, 2.0, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
