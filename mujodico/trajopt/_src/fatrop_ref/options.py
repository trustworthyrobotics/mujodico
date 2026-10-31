"""Static, hashable configuration for the compiled primal-dual solver."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Options:
  max_iter: int = 100
  tol: float = 1e-7
  acceptable_tol: float = 1e-6
  acceptable_iter: int = 15
  constr_viol_tol: float = 1e-6
  linear_backend: str = 'parallel'
  hessian_mode: str = 'exact_lagrangian'
  mu_init: float = 0.1
  mu_min: float = 1e-11
  barrier_tol_factor: float = 10.0
  mu_linear_decrease_factor: float = 0.2
  mu_superlinear_decrease_power: float = 1.5
  compl_inf_tol: float = 1e-4
  tau_min: float = 0.99
  bound_push: float = 1e-2
  bound_frac: float = 1e-2
  bound_relax_factor: float = 1e-8
  warm_start_mult_bound_push: float = 1e-3
  lam_max: float = 1e3
  initialize_dual: bool = True
  kappa_d: float = 1e-5
  kappa_sigma: float = 1e10
  smax: float = 100.0
  max_ls: int = 40
  max_soc: int = 4
  kappa_soc: float = 0.99
  alpha_red_factor: float = 0.5
  alpha_min_frac: float = 0.05
  theta_max_fact: float = 1e4
  theta_min_fact: float = 1e-4
  s_phi: float = 2.3
  s_theta: float = 1.1
  delta: float = 1.0
  gamma_theta: float = 1e-5
  gamma_phi: float = 1e-8
  eta_phi: float = 1e-8
  obj_max_incr: float = 5.0
  filter_reset_trigger: int = 5
  max_filter_resets: int = -1
  delta_w0: float = 1e-4
  delta_wmin: float = 1e-20
  delta_w_max: float = 1e40
  kappa_wmin: float = 1.0 / 3.0
  kappa_wplus: float = 8.0
  kappa_wplusem: float = 100.0
  delta_c: float = 1e-6
  kappa_c: float = 0.25
  max_regularization: int = 32
  max_refinement: int = 3
  linear_tol: float = 1e-10
  pivot_tol: float = 1e-10
  enable_watchdog: bool = True
  watchdog_trigger: int = 10
  watchdog_max: int = 3
  max_soft_resto_iters: int = 10
  soft_resto_reduction: float = 1.0 - 1e-4
  max_resto_iter: int = 100
  resto_reduction: float = 0.9
  rho: float = 1000.0
  zeta: float = 1.0
  enable_restoration: bool = True
  stagewise: bool = False
  use_gpu_line_search: bool = True
  print_level: int = 0

  def __post_init__(self):
    if self.linear_backend not in ('serial', 'parallel', 'pas'):
      raise ValueError('linear_backend must be serial, parallel, or pas')
    if self.hessian_mode not in ('exact_lagrangian', 'cost', 'gauss_newton'):
      raise ValueError('unsupported hessian_mode')
    for name in ('max_iter', 'max_ls', 'max_regularization', 'max_resto_iter'):
      if getattr(self, name) < 1:
        raise ValueError(f'{name} must be positive')
    if self.max_soc < 0 or self.max_soft_resto_iters < 0:
      raise ValueError(
          'correction/restoration iteration counts must be nonnegative'
      )
    if not (0.0 < self.alpha_red_factor < 1.0):
      raise ValueError(
          'alpha_red_factor must lie strictly between zero and one'
      )
    if not (0.0 < self.tau_min < 1.0):
      raise ValueError('tau_min must lie strictly between zero and one')
    if self.tol <= 0 or self.mu_min <= 0 or self.mu_init <= 0:
      raise ValueError('tolerance and barrier parameters must be positive')


SolverOptions = Options
