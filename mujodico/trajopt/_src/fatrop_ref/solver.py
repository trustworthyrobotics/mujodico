"""JAX port of FATROP's primal-dual filter IPM policy.

Scope of this reference port:
* multiple-shooting dynamics
* stage-local equality constraints
* stage-local two-sided inequality functions h(x,u) in [l,u]
* explicit slack variables and lower/upper bound duals
* monotone IPOPT/FATROP barrier update
* FATROP-style filter line search and inertia correction
* interchangeable serial and PAS generalized-Riccati reduced KKT backends

The nonlinear policy is intentionally kept identical between the serial and PAS
versions.  Only ``linear_backend`` changes.

PORTABLE SNAPSHOT: folds the validated v69/v72 restoration path together
with the v73/v74/v76 filter fixes and v82 accepted-step multiplier safeguard.
The historical runtime patch files are shipped inside this package so the
solver has no dependency on repository-root experiment scripts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import math, time
import numpy as np
import jax
import jax.numpy as jnp
from .linear import solve_serial, solve_pas, LinearSolution

Array=jax.Array

@dataclass(frozen=True)
class FatropRefOptions:
    max_iter:int=1000
    tol:float=1e-8
    acceptable_tol:float=1e-6
    acceptable_iter:int=15
    mu_init:float=1e-1
    mu_min:float=1e-11
    kappa_eta:float=10.0
    kappa_mu:float=0.2
    theta_mu:float=1.5
    compl_inf_tol:float=1e-4
    tau_min:float=0.99
    # IPOPT/FATROP filter defaults
    theta_max_fact:float=1e4
    theta_min_fact:float=1e-4
    s_phi:float=2.3
    delta:float=1.0
    s_theta:float=1.1
    gamma_theta:float=1e-5
    gamma_phi:float=1e-8
    eta_phi:float=1e-8
    obj_max_incr:float=5.0
    alpha_min_frac:float=0.05
    tiny_step_tol:float=0.0
    # IPOPT/FATROP filter-reset heuristic defaults.  After an accepted trial,
    # if the last rejected trial in filter_reset_trigger consecutive NLP
    # iterations was rejected by the filter, clear only the filter memory.
    max_filter_resets:int=-1
    filter_reset_trigger:int=5
    gamma_alpha:float=0.5
    max_soc:int=4
    kappa_soc:float=0.99
    # inertia correction, IPOPT defaults used by FATROP naming
    delta_w0:float=1e-4
    delta_wmin:float=1e-20
    kappa_wmin:float=1/3
    kappa_wplus:float=8.0
    kappa_wplusem:float=100.0
    max_inertia_tries:int=20
    # slack/bound initialization
    bound_push:float=1e-2
    bound_frac:float=1e-2
    warm_start_mult_bound_push:float=1e-3
    # FATROP initial equality/inequality multiplier least-squares estimate.
    # The official implementation discards the estimate if any multiplier
    # magnitude exceeds lammax.
    lammax:float=1e3
    initialize_dual:bool=True
    # Source-fidelity knobs that were previously controlled by environment
    # variables in the diagnostic branches.  Defaults preserve the latest
    # stable v76/v82 benchmark behavior; full restoration is available without
    # any external patch files.
    kappa_d:float=0.0
    # Upstream FATROP always has restoration enabled after ordinary
    # globalization fails.  Keep the switch for experiments, but make the
    # source-faithful behavior the default.
    enable_ordinary_restoration:bool=True
    kappa_sigma:float=1e10
    # FATROP scales the dual and complementarity parts of e_mu when the
    # average multiplier is large.  This prevents a few large multipliers
    # from making the barrier update overly conservative.
    smax:float=100.0
    # watchdog
    max_watchdog_steps:int=3
    watchdog_shortened_iter_trigger:int=10
    enable_watchdog:bool=True
    # implementation
    linear_backend:str='serial' # serial | pas
    hessian_mode:str='cost'     # zero | cost | exact_lagrangian
    # Experimental GPU path.  The default keeps the source-faithful Python
    # globalization loop; enabling this moves ordinary filter backtracking
    # into a device-side lax.while_loop.  SOC/watchdog cases fall back to the
    # source-faithful path.
    use_gpu_line_search:bool=False
    print_level:int=0
    max_ls:int=40
    iteration_callback:Any=None

@dataclass
class FatropRefResult:
    x:np.ndarray; u:np.ndarray; s:np.ndarray
    lambda_dyn:np.ndarray; beta_eq:np.ndarray; gamma_ineq:np.ndarray
    zL:np.ndarray; zU:np.ndarray
    objective:float; mu:float; iterations:int; accepted_steps:int
    status:str; solve_time:float; trace:list=field(default_factory=list)

class _Filter:
    # V76_SOURCE_FILTER_RESET_AND_CORNERS
    def __init__(self,theta0,opt):
        # FATROP/Ipopt Filter stores the already-margin-shifted corner, not the
        # raw (theta,phi) reference pair.  This matters for dominance pruning.
        self.entries=[]
        self.theta_max=opt.theta_max_fact*max(1.0,float(theta0))
        self.theta_min=opt.theta_min_fact*max(1.0,float(theta0))
        self.o=opt
        self.last_rejection_due_to_filter=False
        self.count_successive_filter_rejections=0
        self.n_filter_resets=0
        self.current_iteration=-1
    def reset(self):
        # Mirrors FilterLSAcceptor::Reset(): reset filter and rejection history,
        # but NOT n_filter_resets, which is a run-level cap.
        self.last_rejection_due_to_filter=False
        self.count_successive_filter_rejections=0
        self.entries=[]
    def acceptable(self,theta,phi):
        if not np.isfinite(theta+phi) or theta>self.theta_max:return False
        # Entries already include gamma_theta/gamma_phi margins.  Upstream's
        # FilterEntry::Acceptable uses <= in either coordinate.
        return all(theta <= th or phi <= ph for th,ph in self.entries)
    def _switching_condition(self,theta_ref,alpha_test,dphi_ref):
        desc=-dphi_ref
        return (np.isfinite(desc) and desc>0.0 and alpha_test>0.0 and
                alpha_test*desc**self.o.s_phi >
                self.o.delta*max(theta_ref,0.0)**self.o.s_theta)
    def _armijo_holds(self,phi_trial,phi_ref,alpha_test,dphi_ref):
        return phi_trial <= phi_ref+self.o.eta_phi*alpha_test*dphi_ref
    def _current_acceptable(self,theta,phi,th0,ph0,alpha_test,dphi):
        switching=self._switching_condition(th0,alpha_test,dphi)
        if switching and th0<=self.theta_min:
            return self._armijo_holds(phi,ph0,alpha_test,dphi)
        return (theta <= (1-self.o.gamma_theta)*th0 or
                phi <= ph0-self.o.gamma_phi*th0)
    def _accepted_filter_reset_heuristic(self):
        # FilterLSAcceptor::CheckAcceptabilityOfTrialPoint runs this only after
        # the point passed BOTH the current-iterate test and the current filter.
        if self.o.max_filter_resets > 0 or self.o.max_filter_resets == -1:
            if (self.n_filter_resets < self.o.max_filter_resets or
                    self.o.max_filter_resets == -1):
                if self.last_rejection_due_to_filter:
                    self.count_successive_filter_rejections += 1
                    if (self.count_successive_filter_rejections >=
                            self.o.filter_reset_trigger):
                        old_n=self.n_filter_resets
                        self.reset()
                        self.n_filter_resets=old_n+1
                        if self.o.print_level:
                            print(
                                f"V76_FILTER_RESET it={self.current_iteration} "
                                f"count={self.n_filter_resets}",flush=True)
                else:
                    self.count_successive_filter_rejections=0
        self.last_rejection_due_to_filter=False
    def trial(self,theta,phi,th0,ph0,alpha_test,dphi,alpha_update=None):
        # Source order is important for the reset heuristic: first test against
        # the current/reference iterate, and only then test the historical filter.
        if not np.isfinite(theta+phi) or theta>self.theta_max:
            return False,False
        # IpLinesearch::is_acceptable_to_current_iterate rejects an objective
        # barrier increase that is many orders of magnitude larger than the
        # current value.  The filter alone permits such a point whenever its
        # constraint violation improves, which is especially harmful for
        # contact restoration steps.
        if phi > ph0:
            base=1.0 if abs(ph0)<=10.0 else math.log10(abs(ph0))
            if math.log10(max(phi-ph0,np.finfo(float).tiny)) > self.o.obj_max_incr+base:
                self.last_rejection_due_to_filter=False
                return False,False
        if not self._current_acceptable(theta,phi,th0,ph0,alpha_test,dphi):
            self.last_rejection_due_to_filter=False
            return False,False
        if not all(theta <= th or phi <= ph for th,ph in self.entries):
            self.last_rejection_due_to_filter=True
            return False,False
        # We have an accepted point.  Upstream applies the filter reset heuristic
        # BEFORE clearing last_rejection_due_to_filter_.
        self._accepted_filter_reset_heuristic()
        if alpha_update is None:
            alpha_update=alpha_test
        switching_update=self._switching_condition(th0,alpha_update,dphi)
        ftype=switching_update and self._armijo_holds(
            phi,ph0,alpha_update,dphi)
        return True,ftype
    def add(self,theta,phi):
        # FilterLSAcceptor::AugmentFilter stores the corner INCLUDING margin.
        th_add=(1-self.o.gamma_theta)*float(theta)
        ph_add=float(phi)-self.o.gamma_phi*float(theta)
        # Upstream prunes an old entry only when the new corner strictly
        # dominates it in BOTH coordinates.
        self.entries=[(t,p) for t,p in self.entries
                      if not (t>th_add and p>ph_add)]
        self.entries.append((th_add,ph_add))

# PORTABLE_FATROP_PAS_SNAPSHOT_20260915
class FatropReferenceJAX:
    def __init__(self,*,horizon:int,num_states:int,num_controls:int,x0,
                 stage_cost:Callable,terminal_cost:Callable,transition:Callable,
                 stage_equality:Callable,stage_inequality:Callable,
                 ineq_lb,ineq_ub):
        self.N=horizon;self.nx=num_states;self.nu=num_controls
        self.x0=jnp.asarray(x0)
        self.stage_cost=stage_cost;self.terminal_cost=terminal_cost
        self.transition=transition;self.stage_equality=stage_equality;self.stage_inequality=stage_inequality
        self.ineq_lb=jnp.asarray(ineq_lb,dtype=self.x0.dtype);self.ineq_ub=jnp.asarray(ineq_ub,dtype=self.x0.dtype)
        # infer dimensions
        utest=jnp.zeros((num_controls,),dtype=self.x0.dtype)
        self.ne=int(stage_equality(self.x0,utest).shape[0]);self.ni=int(stage_inequality(self.x0,utest).shape[0])
        if self.ineq_lb.shape!=(self.ni,) or self.ineq_ub.shape!=(self.ni,):raise ValueError('inequality bound shape')
        if self.ni and np.any(~(np.isfinite(np.asarray(self.ineq_lb))|np.isfinite(np.asarray(self.ineq_ub)))):
            raise ValueError('each inequality row needs at least one finite bound')
        # CPU/debug Jacobian path: exact forward AD in small tangent batches.
        # This avoids jacfwd's single very-wide XLA graph on MJX contact models.
        self._jvp_batch_size=8
        # CPU/debug path evaluates MJX derivatives in fixed stage chunks.
        # All current contact benchmarks have horizons divisible by 10; this
        # cuts Python dispatch by ~10x without compiling one huge horizon-wide
        # XLA graph.
        self._stage_chunk_size=10
        nv=self.nx+self.nu; nb=(nv+self._jvp_batch_size-1)//self._jvp_batch_size
        bx=np.zeros((nb,self._jvp_batch_size,self.nx),dtype=float)
        bu=np.zeros((nb,self._jvp_batch_size,self.nu),dtype=float)
        for j in range(nv):
            b=j//self._jvp_batch_size; q=j%self._jvp_batch_size
            if j<self.nx: bx[b,q,j]=1.0
            else: bu[b,q,j-self.nx]=1.0
        self._jvp_basis_x=jnp.asarray(bx,dtype=self.x0.dtype)
        self._jvp_basis_u=jnp.asarray(bu,dtype=self.x0.dtype)
        self._jvp_num_dirs=nv
        self._build_derivative_kernels()
        self._build_direction_kernels()

    def _build_derivative_kernels(self):
        # Capture these static problem dimensions/arrays in the fused metric
        # kernels below.  Keeping them in the closure makes the jitted path
        # independent of Python instance attribute lookups and, more
        # importantly, avoids an undefined-name failure when the kernel is
        # first compiled.
        ni=self.ni
        lb=self.ineq_lb
        ub=self.ineq_ub

        def one(x,u,xn):
            gx,gu=jax.grad(self.stage_cost,argnums=(0,1))(x,u)
            (Hxx,Hxu),(_,Huu)=jax.hessian(self.stage_cost,argnums=(0,1))(x,u)
            f=self.transition(x,u);A,B=jax.jacfwd(self.transition,argnums=(0,1))(x,u)
            ce=self.stage_equality(x,u);E,F=jax.jacfwd(self.stage_equality,argnums=(0,1))(x,u)
            hi=self.stage_inequality(x,u);Gx,Gu=jax.jacfwd(self.stage_inequality,argnums=(0,1))(x,u)
            return gx,gu,Hxx,Hxu,Huu,f-xn,A,B,ce,E,F,hi,Gx,Gu
        def cost_derivs(x,u):
            gx,gu=jax.grad(self.stage_cost,argnums=(0,1))(x,u)
            (Hxx,Hxu),(_,Huu)=jax.hessian(self.stage_cost,argnums=(0,1))(x,u)
            return gx,gu,Hxx,Hxu,Huu
        def cons_value(x,u,xn):
            return self.transition(x,u)-xn,self.stage_equality(x,u),self.stage_inequality(x,u)
        def cons_only(x,u):
            return self.transition(x,u),self.stage_equality(x,u),self.stage_inequality(x,u)
        def cons_jvp_one(x,u,tx,tu):
            _,dy=jax.jvp(cons_only,(x,u),(tx,tu))
            return dy
        def cons_jvp_batch(x,u,tx,tu):
            return jax.vmap(cons_jvp_one,in_axes=(None,None,0,0))(x,u,tx,tu)
        def terminal(x):
            return jax.grad(self.terminal_cost)(x),jax.hessian(self.terminal_cost)(x)
        self._one=jax.jit(one);self._terminal=jax.jit(terminal)
        self._cost_derivs=jax.jit(cost_derivs)
        self._cons_value=jax.jit(cons_value)
        self._cons_jvp_batch=jax.jit(cons_jvp_batch)
        # Fixed-size stage batches for the CPU/debug path.  These reuse one
        # executable across chunks and avoid N separate Python dispatches.
        self._cost_derivs_stagebatch=jax.jit(jax.vmap(cost_derivs,in_axes=(0,0)))
        self._cons_value_stagebatch=jax.jit(jax.vmap(cons_value,in_axes=(0,0,0)))
        self._cons_jvp_stagebatch=jax.jit(
            jax.vmap(cons_jvp_batch,in_axes=(0,0,None,None)))
        self._one_vmap=jax.jit(jax.vmap(one,in_axes=(0,0,0)))
        def value_one(x,u,xn):
            return self.stage_cost(x,u), self.transition(x,u)-xn, self.stage_equality(x,u), self.stage_inequality(x,u)
        self._value_one=jax.jit(value_one)
        self._value_vmap=jax.jit(jax.vmap(value_one,in_axes=(0,0,0)))
        self._value_stagebatch=jax.jit(jax.vmap(value_one,in_axes=(0,0,0)))
        self._value_laxmap=jax.jit(lambda xs,us,xns: jax.lax.map(lambda z: value_one(z[0],z[1],z[2]), (xs,us,xns)))
        self._objective_jit=jax.jit(lambda x,u: jnp.sum(jax.vmap(self.stage_cost)(x[:-1],u))+self.terminal_cost(x[-1]))
        def trial_eval(x,u,s,mu,kappa_d):
            # Fuse nonlinear values, violation norm, interior check, and
            # barrier objective into one device program per line-search trial.
            costs,rd,re,h=jax.vmap(value_one,in_axes=(0,0,0))(
                x[:-1],u,x[1:])
            rh=h-s
            objective=jnp.sum(costs)+self.terminal_cost(x[-1])
            if self.ni:
                lb=self.ineq_lb[None,:];ub=self.ineq_ub[None,:]
                flo=jnp.isfinite(lb);fhi=jnp.isfinite(ub)
                Sl=jnp.where(flo,s-lb,1.0);Su=jnp.where(fhi,ub-s,1.0)
                feasible=jnp.all((~flo)|(Sl>0)) & jnp.all((~fhi)|(Su>0))
                logs=jnp.sum(jnp.where(flo,jnp.log(Sl),0.0))+jnp.sum(
                    jnp.where(fhi,jnp.log(Su),0.0))
                lower_only=flo&(~fhi);upper_only=fhi&(~flo)
                damping=(jnp.sum(jnp.where(lower_only,Sl,0.0))+
                         jnp.sum(jnp.where(upper_only,Su,0.0)))
                phi=jnp.where(feasible,
                    objective-mu*logs+kappa_d*mu*damping,jnp.inf)
            else:
                feasible=jnp.array(True);phi=objective
            theta=jnp.sum(jnp.abs(rd))+jnp.sum(jnp.abs(re))+jnp.sum(jnp.abs(rh))
            return rd,re,rh,phi,theta,feasible
        self._trial_eval_jit=jax.jit(trial_eval)
        # Batched inequality initialization avoids one Python/JAX dispatch
        # per shooting stage at the start of every solve.
        self._ineq_vmap_jit=jax.jit(
            jax.vmap(self.stage_inequality,in_axes=(0,0)))

        def gpu_line_search(x,u,s,dx,du,ds,alpha0,theta0,phi0,dphi,mu,kappa_d,
                            filter_theta,filter_phi,filter_count,theta_max,
                            theta_min,s_phi,delta,s_theta,gamma_theta,
                            gamma_phi,eta_phi,gamma_alpha,max_ls):
            """Run ordinary filter backtracking entirely on the device.

            The filter is represented by fixed-size padded arrays because JAX
            requires static shapes.  ``filter_count`` masks the unused tail.
            This path deliberately covers ordinary backtracking only; the
            caller retains the existing Python path for SOC and watchdog
            semantics.
            """
            zeros_rd=jnp.zeros((self.N,self.nx),dtype=x.dtype)
            zeros_re=jnp.zeros((self.N,self.ne),dtype=x.dtype)
            zeros_rh=jnp.zeros((self.N,self.ni),dtype=x.dtype)
            inf=jnp.asarray(jnp.inf,dtype=x.dtype)
            false=jnp.asarray(False)
            init=(jnp.asarray(0,jnp.int32),jnp.asarray(alpha0,dtype=x.dtype),
                  false,jnp.zeros_like(x),jnp.zeros_like(u),jnp.zeros_like(s),
                  zeros_rd,zeros_re,zeros_rh,inf,inf,false,
                  zeros_rd,zeros_re,zeros_rh,inf,inf,false)

            def cond(c):
                i,_,accepted,*_=c
                return (i < max_ls) & (~accepted)

            def body(c):
                (i,alpha,accepted,x_out,u_out,s_out,rd_out,re_out,rh_out,
                 phi_out,theta_out,feasible_out,first_rd,first_re,first_rh,
                 first_phi,first_theta,first_feasible)=c
                xt=x.at[0].set(self.x0)
                xt=xt.at[1:].set(x[1:]+alpha*dx)
                ut=u+alpha*du
                st=s+alpha*ds
                rd,re,rh,phi,theta,feasible=self._trial_eval_jit(
                    xt,ut,st,jnp.asarray(mu,dtype=x.dtype),
                    jnp.asarray(kappa_d,dtype=x.dtype))
                # Ordinary line search tests every trial against the same
                # current iterate reference and the frozen alpha test.
                desc=-dphi
                switching=((jnp.isfinite(desc)) & (desc>0.0) &
                           (alpha0>0.0) &
                           (alpha0*desc**s_phi >
                            delta*jnp.maximum(theta0,0.0)**s_theta))
                current=jnp.where(
                    switching & (theta0<=theta_min),
                    phi <= phi0 + eta_phi*alpha0*dphi,
                    (theta <= (1.0-gamma_theta)*theta0) |
                    (phi <= phi0-gamma_phi*theta0))
                idx=jnp.arange(filter_theta.shape[0])
                active=idx < filter_count
                historical=jnp.all(
                    (~active) | (theta <= filter_theta) |
                    (phi <= filter_phi))
                accepted_trial=feasible & (theta <= theta_max) & current & historical
                take_first=(i==0)
                rd_first=jnp.where(take_first,rd,first_rd)
                re_first=jnp.where(take_first,re,first_re)
                rh_first=jnp.where(take_first,rh,first_rh)
                phi_first=jnp.where(take_first,phi,first_phi)
                theta_first=jnp.where(take_first,theta,first_theta)
                feasible_first=jnp.where(take_first,feasible,first_feasible)
                return (
                    i+1,jnp.where(accepted_trial,alpha,alpha*gamma_alpha),
                    accepted_trial,
                    jnp.where(accepted_trial,xt,x_out),
                    jnp.where(accepted_trial,ut,u_out),
                    jnp.where(accepted_trial,st,s_out),
                    jnp.where(accepted_trial,rd,rd_out),
                    jnp.where(accepted_trial,re,re_out),
                    jnp.where(accepted_trial,rh,rh_out),
                    jnp.where(accepted_trial,phi,phi_out),
                    jnp.where(accepted_trial,theta,theta_out),
                    jnp.where(accepted_trial,feasible,feasible_out),
                    rd_first,re_first,rh_first,phi_first,theta_first,
                    feasible_first)

            out=jax.lax.while_loop(cond,body,init)
            return out
        self._gpu_line_search_jit=jax.jit(gpu_line_search)
        def stationarity_fast(gx,gu,A,B,E,F,Gx,Gu,gf,lam,beta,gamma):
            prev=jnp.concatenate([jnp.zeros((1,self.nx),dtype=gx.dtype),lam[:-1]],0)
            rx=gx+jnp.einsum('nij,ni->nj',A,lam)-prev
            ru=gu+jnp.einsum('nij,ni->nj',B,lam)
            if self.ne:
                rx=rx+jnp.einsum('nij,ni->nj',E,beta);ru=ru+jnp.einsum('nij,ni->nj',F,beta)
            if self.ni:
                rx=rx+jnp.einsum('nij,ni->nj',Gx,gamma);ru=ru+jnp.einsum('nij,ni->nj',Gu,gamma)
            return rx,ru,gf-lam[-1]
        self._stationarity_jit=jax.jit(stationarity_fast)

        # The outer FATROP policy still needs host scalars for its Python
        # filter/watchdog decisions, but computing each scalar separately
        # caused several implicit device synchronizations per NLP iteration.
        # Keep the policy unchanged and return all convergence metrics from one
        # device program instead.  The tuple is transferred once by
        # ``_residual_measure`` below.
        def residual_metrics(gx,gu,A,B,E,F,Gx,Gu,gf,rd,re,rh,lam,beta,gamma,
                             s,zL,zU,mu,smax):
            rx,ru,rxf=stationarity_fast(
                gx,gu,A,B,E,F,Gx,Gu,gf,lam,beta,gamma)
            dual=jnp.maximum(
                jnp.max(jnp.abs(rx[1:]),initial=0),
                jnp.maximum(jnp.max(jnp.abs(ru),initial=0),
                            jnp.max(jnp.abs(rxf),initial=0)))
            primal=jnp.maximum(
                jnp.max(jnp.abs(rd),initial=0),
                jnp.maximum(jnp.max(jnp.abs(re),initial=0),
                            jnp.max(jnp.abs(rh),initial=0)))
            if ni:
                flo=jnp.isfinite(lb)[None,:]
                fhi=jnp.isfinite(ub)[None,:]
                Sl=jnp.where(flo,s-lb[None,:],1.0)
                Su=jnp.where(fhi,ub[None,:]-s,1.0)
                rs=-gamma-zL+zU
                dual=jnp.maximum(dual,jnp.max(jnp.abs(rs),initial=0))
                rL=jnp.where(flo,Sl*zL-mu,0.0)
                rU=jnp.where(fhi,Su*zU-mu,0.0)
                comp=jnp.maximum(jnp.max(jnp.abs(rL),initial=0),
                                 jnp.max(jnp.abs(rU),initial=0))
            else:
                comp=jnp.asarray(0.0,dtype=gx.dtype)
            # Match IpIterate::e_mu(): primal feasibility is unscaled, while
            # dual infeasibility and complementarity are divided by the
            # corresponding mean-multiplier scale once that mean exceeds
            # ``smax``.  The upstream solver uses L1 means for this test and
            # an infinity norm for each residual family.
            bound_count=self.N*(jnp.sum(jnp.isfinite(lb))+jnp.sum(jnp.isfinite(ub)))
            eq_count=lam.size + beta.size + gamma.size
            eq_count=jnp.asarray(eq_count,dtype=gx.dtype)
            bound_count=jnp.maximum(jnp.asarray(bound_count,dtype=gx.dtype),1.0)
            eq_count=jnp.maximum(eq_count,1.0)
            lam_l1=(jnp.sum(jnp.abs(lam))+jnp.sum(jnp.abs(beta))+
                    jnp.sum(jnp.abs(gamma)))
            z_l1=jnp.sum(jnp.abs(zL))+jnp.sum(jnp.abs(zU))
            lam_mean=(lam_l1+z_l1)/(eq_count+bound_count)
            z_mean=z_l1/bound_count
            dual=dual/jnp.maximum(1.0,lam_mean/smax)
            comp=comp/jnp.maximum(1.0,z_mean/smax)
            return jnp.stack((jnp.maximum(dual,jnp.maximum(primal,comp)),
                              dual,primal,comp))
        self._residual_metrics_jit=jax.jit(residual_metrics)
        def exact_hess(x,u,lam,beta,gamma):
            def lag(xx,uu):
                val=self.stage_cost(xx,uu)+jnp.dot(lam,self.transition(xx,uu))
                if self.ne: val=val+jnp.dot(beta,self.stage_equality(xx,uu))
                if self.ni: val=val+jnp.dot(gamma,self.stage_inequality(xx,uu))
                return val
            (Hxx,Hxu),(_,Huu)=jax.hessian(lag,argnums=(0,1))(x,u)
            return Hxx,Hxu,Huu
        self._exact_hess=jax.jit(exact_hess)
        self._exact_hess_vmap=jax.jit(jax.vmap(exact_hess,in_axes=(0,0,0,0,0)))
        self._exact_hess_stagebatch=jax.jit(jax.vmap(exact_hess,in_axes=(0,0,0,0,0)))
        self._exact_hess_laxmap=jax.jit(lambda xs,us,ls,bs,gs: jax.lax.map(lambda z: exact_hess(z[0],z[1],z[2],z[3],z[4]), (xs,us,ls,bs,gs)))

    def _build_direction_kernels(self):
        lb=self.ineq_lb;ub=self.ineq_ub;ne=self.ne;ni=self.ni;nx=self.nx
        def core(lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,kappa_d,backend_pas):
            gx,gu,Q,S,R,rd,A,B,re,E,F,h,Gx,Gu,gf,Qf=lin
            prev=jnp.concatenate([jnp.zeros((1,nx),dtype=x.dtype),lam[:-1]],axis=0)
            rx=gx+jnp.einsum('nij,ni->nj',A,lam)-prev
            ru=gu+jnp.einsum('nij,ni->nj',B,lam)
            if ne:
                rx=rx+jnp.einsum('nij,ni->nj',E,beta)
                ru=ru+jnp.einsum('nij,ni->nj',F,beta)
            if ni:
                rx=rx+jnp.einsum('nij,ni->nj',Gx,gamma)
                ru=ru+jnp.einsum('nij,ni->nj',Gu,gamma)
            rxf=gf-lam[-1]
            if ni:
                flo=jnp.isfinite(lb)[None,:];fhi=jnp.isfinite(ub)[None,:]
                Sl=jnp.where(flo,s-lb[None,:],1.0);Su=jnp.where(fhi,ub[None,:]-s,1.0)
                rh=h-s
                lower_only=flo & (~fhi); upper_only=fhi & (~flo)
                damp_grad=kappa_d*mu*(lower_only.astype(s.dtype)-upper_only.astype(s.dtype))
                rs=damp_grad-gamma-zL+zU
                rL=jnp.where(flo,Sl*zL-mu,0.0);rU=jnp.where(fhi,Su*zU-mu,0.0)
                sig=jnp.where(flo,zL/Sl,0.0)+jnp.where(fhi,zU/Su,0.0)
                # FATROP v0.0.4 regularizes both the primal Hessian and the
                # inequality Schur coefficient.  Its Riccati inequality branch
                # forms (sig + delta_w) * [G; h] and then adds gradb to the
                # affine row before the Schur update.  In the equivalent
                # augmented KKT representation this means
                #     W = diag(1 / (sig + delta_w))
                #     g = rh + W * gradb,
                # rather than using the unregularized sig here.
                sig_reg=sig+reg
                # Keep the diagonal Schur block compact.  Expanding this to
                # (N,ni,ni) was a large allocation for contact-rich GPU runs.
                wd=1.0/sig_reg;W=wd
                ss=rs+jnp.where(flo,rL/Sl,0.0)-jnp.where(fhi,rU/Su,0.0)
                gt=rh+wd*ss
            else:
                rh=jnp.zeros((self.N,0),dtype=x.dtype);Sl=rh;Su=rh;flo=jnp.zeros_like(rh,dtype=bool);fhi=flo
                rL=rh;rU=rh;W=jnp.zeros((self.N,0),dtype=x.dtype);gt=rh
            def do_serial(_):
                return solve_serial(Q=Q,S=S,R=R,q=rx,r=ru,A=A,B=B,a=rd,E=E,F=F,e=re,Gx=Gx,Gu=Gu,g=gt,W=W,Qf=Qf,qf=rxf,dx0=jnp.zeros((nx,),dtype=x.dtype),regularization=reg)
            def do_pas(_):
                return solve_pas(Q=Q,S=S,R=R,q=rx,r=ru,A=A,B=B,a=rd,E=E,F=F,e=re,Gx=Gx,Gu=Gu,g=gt,W=W,Qf=Qf,qf=rxf,dx0=jnp.zeros((nx,),dtype=x.dtype),regularization=reg)
            sol=jax.lax.cond(backend_pas,do_pas,do_serial,operand=None)
            if ni:
                ds=jnp.einsum('nij,nj->ni',Gx,sol.dx[:-1])+jnp.einsum('nij,nj->ni',Gu,sol.du)+rh
                dzL=jnp.where(flo,(-rL-zL*ds)/Sl,0.0)
                dzU=jnp.where(fhi,(-rU+zU*ds)/Su,0.0)
            else:
                ds=jnp.zeros_like(s);dzL=ds;dzU=ds
            return sol,ds,dzL,dzU
        # Separate compiled executables avoid tracing both O(N) serial and scan paths at every call.
        self._direction_serial_jit=jax.jit(lambda lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,kappa_d:core(lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,kappa_d,False))
        self._direction_pas_jit=jax.jit(lambda lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,kappa_d:core(lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,kappa_d,True))

        def soc_core(lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,kappa_d,rd_soc,re_soc,rh_soc,backend_pas):
            # FATROP SOC reuses the *same* KKT matrix and the current dual /
            # complementarity residuals.  Only the primal-constraint RHS is
            # replaced by the accumulated SOC residual.  The solve therefore
            # returns a fresh total search direction; it is not an additive
            # correction to the rejected Newton direction.
            gx,gu,Q,S,R,_,A,B,_,E,F,h,Gx,Gu,gf,Qf=lin
            dtype=Q.dtype
            prev=jnp.concatenate([jnp.zeros((1,nx),dtype=x.dtype),lam[:-1]],axis=0)
            rx=gx+jnp.einsum('nij,ni->nj',A,lam)-prev
            ru=gu+jnp.einsum('nij,ni->nj',B,lam)
            if ne:
                rx=rx+jnp.einsum('nij,ni->nj',E,beta)
                ru=ru+jnp.einsum('nij,ni->nj',F,beta)
            if ni:
                rx=rx+jnp.einsum('nij,ni->nj',Gx,gamma)
                ru=ru+jnp.einsum('nij,ni->nj',Gu,gamma)
            rxf=gf-lam[-1]
            if ni:
                flo=jnp.isfinite(lb)[None,:]; fhi=jnp.isfinite(ub)[None,:]
                Sl=jnp.where(flo,s-lb[None,:],1.0); Su=jnp.where(fhi,ub[None,:]-s,1.0)
                lower_only=flo & (~fhi); upper_only=fhi & (~flo)
                damp_grad=kappa_d*mu*(lower_only.astype(s.dtype)-upper_only.astype(s.dtype))
                rs=damp_grad-gamma-zL+zU
                rL=jnp.where(flo,Sl*zL-mu,0.0); rU=jnp.where(fhi,Su*zU-mu,0.0)
                sig=jnp.where(flo,zL/Sl,0.0)+jnp.where(fhi,zU/Su,0.0)
                sig_reg=sig+reg
                wd=1.0/sig_reg; W=wd
                ss=rs+jnp.where(flo,rL/Sl,0.0)-jnp.where(fhi,rU/Su,0.0)
                gt=rh_soc+wd*ss
            else:
                flo=jnp.zeros((self.N,0),dtype=bool);fhi=flo
                Sl=jnp.zeros((self.N,0),dtype=dtype);Su=Sl
                rL=Sl;rU=Sl;W=jnp.zeros((self.N,0),dtype=dtype);gt=Sl
            def do_serial(_):
                return solve_serial(Q=Q,S=S,R=R,q=rx,r=ru,A=A,B=B,a=rd_soc,E=E,F=F,e=re_soc,
                                    Gx=Gx,Gu=Gu,g=gt,W=W,Qf=Qf,qf=rxf,
                                    dx0=jnp.zeros((nx,),dtype=dtype),regularization=reg)
            def do_pas(_):
                return solve_pas(Q=Q,S=S,R=R,q=rx,r=ru,A=A,B=B,a=rd_soc,E=E,F=F,e=re_soc,
                                 Gx=Gx,Gu=Gu,g=gt,W=W,Qf=Qf,qf=rxf,
                                 dx0=jnp.zeros((nx,),dtype=dtype),regularization=reg)
            sol=jax.lax.cond(backend_pas,do_pas,do_serial,operand=None)
            if ni:
                ds=jnp.einsum('nij,nj->ni',Gx,sol.dx[:-1])+jnp.einsum('nij,nj->ni',Gu,sol.du)+rh_soc
                dzL=jnp.where(flo,(-rL-zL*ds)/Sl,0.0)
                dzU=jnp.where(fhi,(-rU+zU*ds)/Su,0.0)
            else:
                ds=jnp.zeros((self.N,0),dtype=dtype);dzL=ds;dzU=ds
            return sol,ds,dzL,dzU
        self._soc_serial_jit=jax.jit(lambda lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,kappa_d,rd,re,rh: soc_core(lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,kappa_d,rd,re,rh,False))
        self._soc_pas_jit=jax.jit(lambda lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,kappa_d,rd,re,rh: soc_core(lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,kappa_d,rd,re,rh,True))

        # Never materialize search directions on the host just to check them.
        # The previous ``np.asarray`` checks copied O(N*(nx+nu+ne+ni)) values
        # from GPU to CPU on every inertia attempt and every SOC attempt.
        def direction_ok(sol, ds, dzL, dzU):
            return (sol.inertia_ok & jnp.all(jnp.isfinite(sol.dx)) &
                    jnp.all(jnp.isfinite(sol.du)) &
                    jnp.all(jnp.isfinite(sol.dlambda_dyn)) &
                    jnp.all(jnp.isfinite(sol.dbeta_eq)) &
                    jnp.all(jnp.isfinite(sol.dgamma_ineq)) &
                    jnp.all(jnp.isfinite(ds)) &
                    jnp.all(jnp.isfinite(dzL)) &
                    jnp.all(jnp.isfinite(dzU)))
        self._direction_ok_jit = jax.jit(direction_ok)

        def max_step_metrics(s,zL,zU,ds,dzL,dzU,mu):
            tau=jnp.maximum(jnp.asarray(0.99,dtype=s.dtype),1.0-mu)
            if ni:
                flo=jnp.isfinite(lb)[None,:];fhi=jnp.isfinite(ub)[None,:]
                Sl=jnp.where(flo,s-lb[None,:],1.0)
                Su=jnp.where(fhi,ub[None,:]-s,1.0)
                pv=jnp.where(flo&(ds<0),tau*Sl/(-ds),jnp.inf)
                pv=jnp.minimum(pv,jnp.where(fhi&(ds>0),tau*Su/ds,jnp.inf))
                dv=jnp.minimum(
                    jnp.where(flo&(dzL<0),tau*zL/(-dzL),jnp.inf),
                    jnp.where(fhi&(dzU<0),tau*zU/(-dzU),jnp.inf))
                return (jnp.minimum(1.0,jnp.min(pv,initial=1.0)),
                        jnp.minimum(1.0,jnp.min(dv,initial=1.0)))
            return jnp.asarray(1.0,s.dtype),jnp.asarray(1.0,s.dtype)
        self._max_step_jit=jax.jit(max_step_metrics)

        def current_step_metrics(lin,x,u,s,zL,zU,sol,ds,dzL,dzU,mu,kd):
            valid=(sol.inertia_ok & jnp.all(jnp.isfinite(sol.dx)) &
                   jnp.all(jnp.isfinite(sol.du)) &
                   jnp.all(jnp.isfinite(sol.dlambda_dyn)) &
                   jnp.all(jnp.isfinite(sol.dbeta_eq)) &
                   jnp.all(jnp.isfinite(sol.dgamma_ineq)) &
                   jnp.all(jnp.isfinite(ds)) & jnp.all(jnp.isfinite(dzL)) &
                   jnp.all(jnp.isfinite(dzU)))
            ap,ad=max_step_metrics(s,zL,zU,ds,dzL,dzU,mu)
            rd,re,h=lin[5],lin[8],lin[11]
            rh=h-s
            theta=jnp.sum(jnp.abs(rd))+jnp.sum(jnp.abs(re))+jnp.sum(jnp.abs(rh))
            objective=self._objective_jit(x,u)
            dphi=(jnp.sum(lin[0]*sol.dx[:-1])+jnp.sum(lin[1]*sol.du)+
                   lin[-2]@sol.dx[-1])
            if ni:
                flo=jnp.isfinite(lb)[None,:];fhi=jnp.isfinite(ub)[None,:]
                Sl=jnp.where(flo,s-lb[None,:],1.0)
                Su=jnp.where(fhi,ub[None,:]-s,1.0)
                logs=jnp.sum(jnp.where(flo,jnp.log(Sl),0.0))+jnp.sum(
                    jnp.where(fhi,jnp.log(Su),0.0))
                lower_only=flo&(~fhi);upper_only=fhi&(~flo)
                damping=jnp.sum(jnp.where(lower_only,Sl,0.0))+jnp.sum(
                    jnp.where(upper_only,Su,0.0))
                phi=objective-mu*logs+kd*mu*damping
                dphi=dphi+jnp.sum(jnp.where(flo,-mu/Sl*ds,0.0)+
                                  jnp.where(fhi,mu/Su*ds,0.0))
                dphi=dphi+kd*mu*jnp.sum(
                    jnp.where(lower_only,ds,0.0)-jnp.where(upper_only,ds,0.0))
            else:
                phi=objective
            return valid,ap,ad,phi,theta,dphi
        self._current_step_metrics_jit=jax.jit(current_step_metrics)

    def _linearize(self,x,u,stagewise=False):
        if stagewise:
            # Chunk stages into a fixed shape so the same compiled executable
            # is reused.  Fall back to the historical scalar path only for a
            # final remainder chunk (none of the five current benchmarks has
            # one with the default chunk size).
            cs=min(self._stage_chunk_size,self.N)
            vals=[]
            for k0 in range(0,self.N,cs):
                k1=min(self.N,k0+cs)
                if k1-k0==cs:
                    xs=x[k0:k1]; us=u[k0:k1]; xns=x[k0+1:k1+1]
                    cvals=self._cost_derivs_stagebatch(xs,us)
                    rd,re,h=self._cons_value_stagebatch(xs,us,xns)
                    dfs=[];des=[];dhs=[]
                    for b in range(self._jvp_basis_x.shape[0]):
                        dfb,deb,dhb=self._cons_jvp_stagebatch(
                            xs,us,self._jvp_basis_x[b],self._jvp_basis_u[b])
                        dfs.append(dfb);des.append(deb);dhs.append(dhb)
                    # stage x direction x output -> stage x output x direction
                    df=jnp.concatenate(dfs,axis=1)[:,:self._jvp_num_dirs]
                    de=jnp.concatenate(des,axis=1)[:,:self._jvp_num_dirs]
                    dh=jnp.concatenate(dhs,axis=1)[:,:self._jvp_num_dirs]
                    A=df[:,:self.nx].transpose(0,2,1); B=df[:,self.nx:].transpose(0,2,1)
                    E=de[:,:self.nx].transpose(0,2,1); F=de[:,self.nx:].transpose(0,2,1)
                    Gx=dh[:,:self.nx].transpose(0,2,1); Gu=dh[:,self.nx:].transpose(0,2,1)
                    gx,gu,Hxx,Hxu,Huu=cvals
                    vals.append((gx,gu,Hxx,Hxu,Huu,rd,A,B,re,E,F,h,Gx,Gu))
                else:
                    sv=[]
                    for k in range(k0,k1):
                        gx,gu,Hxx,Hxu,Huu=self._cost_derivs(x[k],u[k])
                        rd,re,h=self._cons_value(x[k],u[k],x[k+1])
                        dfs=[];des=[];dhs=[]
                        for b in range(self._jvp_basis_x.shape[0]):
                            dfb,deb,dhb=self._cons_jvp_batch(
                                x[k],u[k],self._jvp_basis_x[b],self._jvp_basis_u[b])
                            dfs.append(dfb);des.append(deb);dhs.append(dhb)
                        df=jnp.concatenate(dfs,axis=0)[:self._jvp_num_dirs]
                        de=jnp.concatenate(des,axis=0)[:self._jvp_num_dirs]
                        dh=jnp.concatenate(dhs,axis=0)[:self._jvp_num_dirs]
                        A=df[:self.nx].T; B=df[self.nx:].T
                        E=de[:self.nx].T; F=de[self.nx:].T
                        Gx=dh[:self.nx].T; Gu=dh[self.nx:].T
                        sv.append((gx,gu,Hxx,Hxu,Huu,rd,A,B,re,E,F,h,Gx,Gu))
                    vals.append(tuple(jnp.stack([v[i] for v in sv]) for i in range(14)))
            out=tuple(jnp.concatenate([v[i] for v in vals],axis=0) for i in range(14))
        else:
            out=self._one_vmap(x[:-1],u,x[1:])
        gf,Hf=self._terminal(x[-1]);return (*out,gf,Hf)

    def _push_slacks(self,h,opt):
        lb=self.ineq_lb[None,:];ub=self.ineq_ub[None,:]
        s=h
        flo=jnp.isfinite(lb);fhi=jnp.isfinite(ub)
        # IPOPT-style absolute/relative interior push.
        pl=opt.bound_push*jnp.maximum(1.0,jnp.abs(lb))
        pu=opt.bound_push*jnp.maximum(1.0,jnp.abs(ub))
        both=flo&fhi; width=jnp.where(both,ub-lb,1.0)
        pl=jnp.where(both,jnp.minimum(pl,opt.bound_frac*width),pl)
        pu=jnp.where(both,jnp.minimum(pu,opt.bound_frac*width),pu)
        s=jnp.where(flo,jnp.maximum(s,lb+pl),s)
        s=jnp.where(fhi,jnp.minimum(s,ub-pu),s)
        return s

    def _margins(self,s):
        lb=self.ineq_lb[None,:];ub=self.ineq_ub[None,:]
        flo=jnp.isfinite(lb);fhi=jnp.isfinite(ub)
        Sl=jnp.where(flo,s-lb,1.0);Su=jnp.where(fhi,ub-s,1.0)
        return flo,fhi,Sl,Su

    def _objective(self,x,u):
        return float(self._objective_jit(x,u))

    def _barrier_phi(self,x,u,s,mu):
        f=self._objective(x,u)
        if not self.ni:return f
        flo,fhi,Sl,Su=self._margins(s)
        if bool(jnp.any((flo&(Sl<=0))|(fhi&(Su<=0)))):return math.inf
        val=jnp.where(flo,jnp.log(Sl),0).sum()+jnp.where(fhi,jnp.log(Su),0).sum()
        kd=float(getattr(self,'_active_kappa_d',0.0))
        if kd:
            lower_only=flo & (~fhi); upper_only=fhi & (~flo)
            damp=jnp.where(lower_only,Sl,0.0).sum()+jnp.where(upper_only,Su,0.0).sum()
            return f-mu*float(val)+kd*mu*float(damp)
        return f-mu*float(val)

    def _constraints(self,x,u,s):
        if getattr(self,'_stagewise_active',False):
            cs=min(self._stage_chunk_size,self.N); chunks=[]
            for k0 in range(0,self.N,cs):
                k1=min(self.N,k0+cs)
                if k1-k0==cs:
                    chunks.append(self._value_stagebatch(x[k0:k1],u[k0:k1],x[k0+1:k1+1]))
                else:
                    vals=[self._value_one(x[k],u[k],x[k+1]) for k in range(k0,k1)]
                    chunks.append(tuple(jnp.stack([v[i] for v in vals]) for i in range(4)))
            rd=jnp.concatenate([v[1] for v in chunks],axis=0)
            re=(jnp.concatenate([v[2] for v in chunks],axis=0) if self.ne
                else jnp.zeros((self.N,0),dtype=x.dtype))
            hh=(jnp.concatenate([v[3] for v in chunks],axis=0) if self.ni
                else jnp.zeros((self.N,0),dtype=x.dtype))
        else:
            _,rd,re,hh=self._value_vmap(x[:-1],u,x[1:])
        return rd,re,hh-s

    def _trial_eval(self,x,u,s,mu):
        kd=jnp.asarray(float(getattr(self,'_active_kappa_d',0.0)),dtype=x.dtype)
        rd,re,rh,phi,theta,feasible=self._trial_eval_jit(
            x,u,s,jnp.asarray(mu,dtype=x.dtype),kd)
        # Transfer only three scalars.  Residual arrays remain on device and
        # are consumed only if SOC is attempted.
        phi_h,theta_h,feasible_h=jax.device_get((phi,theta,feasible))
        return (rd,re,rh),float(phi_h),float(theta_h),bool(feasible_h)

    @staticmethod
    def _constraints_from_linearization(lin,s):
        """Reuse residual values already produced by ``_linearize``."""
        return lin[5],lin[8],lin[11]-s

    @staticmethod
    def _theta(rd,re,rh,norm='l1'):
        if norm=='linf':
            return float(max(float(jnp.max(jnp.abs(rd),initial=0)),float(jnp.max(jnp.abs(re),initial=0)),float(jnp.max(jnp.abs(rh),initial=0))))
        return float(jnp.sum(jnp.abs(rd))+jnp.sum(jnp.abs(re))+jnp.sum(jnp.abs(rh)))

    def _stationarity(self,lin,lam,beta,gamma):
        gx,gu,_,_,_,_,A,B,_,E,F,_,Gx,Gu,gf,_=lin
        return self._stationarity_jit(gx,gu,A,B,E,F,Gx,Gu,gf,lam,beta,gamma)

    def _residual_measure(self,lin,x,u,s,lam,beta,gamma,zL,zU,mu):
        # ``lin`` belongs to this exact accepted iterate and already contains
        # all nonlinear constraint values.  Re-evaluating MJX here used to add
        # a complete batched contact-model launch to every convergence check.
        rd,re,rh=self._constraints_from_linearization(lin,s)
        metrics=self._residual_metrics_jit(
            lin[0],lin[1],lin[6],lin[7],lin[9],lin[10],lin[12],lin[13],
            lin[14],rd,re,rh,lam,beta,gamma,s,zL,zU,
            jnp.asarray(mu,dtype=s.dtype),
            jnp.asarray(float(getattr(self, '_active_smax', 100.0)),dtype=s.dtype))
        em,dual,primal,comp=jax.device_get(metrics)
        return float(em),float(dual),float(primal),float(comp)

    def _initial_dual_estimate(self,lin,zL,zU,backend,lammax):
        """FATROP's least-squares initial multiplier estimate.

        The official OCP initializer replaces the primal Hessian by identity,
        zeros all constraint residual right-hand sides, and solves the same
        structured KKT system.  Inequality slacks are primal variables with an
        identity Hessian. Eliminating their direction gives an inequality-dual
        block ``-I`` and affine term ``grad_s - zL + zU``.  For these OCPs the
        objective has no direct slack term, so ``grad_s=0``.

        The returned dynamics/local/inequality multipliers are accepted only
        when their maximum magnitude is below ``lammax``, matching
        FatropAlg::perform_initializiation_dual().
        """
        gx,gu,_,_,_,_,A,B,_,E,F,_,Gx,Gu,gf,_=lin
        dtype=gx.dtype
        Q=jnp.broadcast_to(jnp.eye(self.nx,dtype=dtype),(self.N,self.nx,self.nx))
        R=jnp.broadcast_to(jnp.eye(self.nu,dtype=dtype),(self.N,self.nu,self.nu))
        S=jnp.zeros((self.N,self.nx,self.nu),dtype=dtype)
        Qf=jnp.eye(self.nx,dtype=dtype)
        a=jnp.zeros((self.N,self.nx),dtype=dtype)
        e=jnp.zeros((self.N,self.ne),dtype=dtype)
        if self.ni:
            # slack stationarity: grad_s - gamma - zL + zU = 0.
            # Eliminating identity-Hessian ds gives
            # G dx - gamma + (grad_s-zL+zU) = 0.
            g=-zL+zU
            W=jnp.ones((self.N,self.ni),dtype=dtype)
        else:
            g=jnp.zeros((self.N,0),dtype=dtype)
            W=jnp.zeros((self.N,0),dtype=dtype)
        fn=solve_serial if backend=='serial' else solve_pas
        sol=fn(Q=Q,S=S,R=R,q=gx,r=gu,A=A,B=B,a=a,E=E,F=F,e=e,
               Gx=Gx,Gu=Gu,g=g,W=W,Qf=Qf,qf=gf,
               dx0=jnp.zeros((self.nx,),dtype=dtype),regularization=0.0)
        if not bool(sol.inertia_ok):
            return (jnp.zeros_like(sol.dlambda_dyn),jnp.zeros_like(sol.dbeta_eq),
                    jnp.zeros_like(sol.dgamma_ineq),False)
        vmax=max(float(jnp.max(jnp.abs(sol.dlambda_dyn),initial=0.0)),
                 float(jnp.max(jnp.abs(sol.dbeta_eq),initial=0.0)),
                 float(jnp.max(jnp.abs(sol.dgamma_ineq),initial=0.0)))
        if (not np.isfinite(vmax)) or vmax>=float(lammax):
            return (jnp.zeros_like(sol.dlambda_dyn),jnp.zeros_like(sol.dbeta_eq),
                    jnp.zeros_like(sol.dgamma_ineq),False)
        return sol.dlambda_dyn,sol.dbeta_eq,sol.dgamma_ineq,True

    def _active_hessian_lin(self,lin,x,u,lam,beta,gamma):
        """Return the linearization with the Hessian used by the current FATROP KKT solve."""
        gx,gu,Q,S,R,rd,A,B,re,E,F,h,Gx,Gu,gf,Qf=lin
        if getattr(self,'_active_hessian_mode','cost')=='zero':
            zq=jnp.zeros_like(Q); zs=jnp.zeros_like(S); zr=jnp.zeros_like(R)
            zf=jnp.zeros_like(Qf)
            return (gx,gu,zq,zs,zr,rd,A,B,re,E,F,h,Gx,Gu,gf,zf)
        if getattr(self,'_active_hessian_mode','cost')=='exact_lagrangian':
            if getattr(self,'_stagewise_active',False):
                cs=min(self._stage_chunk_size,self.N); qv=[];sv=[];rv=[]
                for k0 in range(0,self.N,cs):
                    k1=min(self.N,k0+cs)
                    if k1-k0==cs:
                        qq,ss,rr=self._exact_hess_stagebatch(
                            x[k0:k1],u[k0:k1],lam[k0:k1],beta[k0:k1],gamma[k0:k1])
                    else:
                        hv=[self._exact_hess(x[k],u[k],lam[k],beta[k],gamma[k]) for k in range(k0,k1)]
                        qq=jnp.stack([v[0] for v in hv]);ss=jnp.stack([v[1] for v in hv]);rr=jnp.stack([v[2] for v in hv])
                    qv.append(qq);sv.append(ss);rv.append(rr)
                Q=jnp.concatenate(qv);S=jnp.concatenate(sv);R=jnp.concatenate(rv)
            else:
                Q,S,R=self._exact_hess_vmap(x[:-1],u,lam,beta,gamma)
            lin=(gx,gu,Q,S,R,rd,A,B,re,E,F,h,Gx,Gu,gf,Qf)
        return lin

    def _direction(self,lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,backend):
        lin_used=self._active_hessian_lin(lin,x,u,lam,beta,gamma)
        fn=self._direction_serial_jit if backend=='serial' else self._direction_pas_jit
        kd=jnp.asarray(float(getattr(self,'_active_kappa_d',0.0)),dtype=x.dtype)
        sol=fn(lin_used,x,u,s,lam,beta,gamma,zL,zU,jnp.asarray(mu,dtype=x.dtype),jnp.asarray(reg,dtype=x.dtype),kd)
        return (*sol,lin_used)

    def _soc_direction(self,lin_used,x,u,s,lam,beta,gamma,zL,zU,mu,reg,soc_residuals,backend):
        """Solve FATROP's SOC RHS using the already-factorized KKT semantics.

        ``soc_residuals`` is the accumulated primal constraint RHS.  The dual
        stationarity and complementarity RHSs remain those of the current
        iterate, matching FATROP's ``solve_soc_rhs`` path.
        """
        rd_t,re_t,rh_t=soc_residuals
        fn=self._soc_serial_jit if backend=='serial' else self._soc_pas_jit
        kd=jnp.asarray(float(getattr(self,'_active_kappa_d',0.0)),dtype=s.dtype)
        return fn(lin_used,x,u,s,lam,beta,gamma,zL,zU,
                  jnp.asarray(mu,dtype=s.dtype),jnp.asarray(reg,dtype=s.dtype),kd,
                  rd_t,re_t,rh_t)

    def _v72_full_restoration(self,*,x,u,s,lam,beta,gamma,zL,zU,mu,opt,stagewise,it):
        """Run the validated v37+v43 full restoration from the current iterate."""
        from . import restoration_v32 as _rv
        if not getattr(_rv,'_v44_hard_dynamics_installed',False):
            from .fatrop_v43_runtime_patch import install_v43_hard_dynamics
            install_v43_hard_dynamics(_rv)
            _rv._v44_hard_dynamics_installed=True
        return _rv.full_restoration(
            self,x=x,u=u,s=s,lam=lam,beta=beta,gamma=gamma,zL=zL,zU=zU,
            mu_orig=mu,opt=opt,stagewise=stagewise,outer_iteration=max(0,it-1),
            backend=opt.linear_backend,print_rows=bool(opt.print_level))

    def _max_step(self,s,zL,zU,ds,dzL,dzU,mu):
        ap,ad=jax.device_get(self._max_step_jit(
            s,zL,zU,ds,dzL,dzU,jnp.asarray(mu,dtype=s.dtype)))
        return float(ap),float(ad)

    def solve(self,x_init,u_init,opt:FatropRefOptions=FatropRefOptions(),*,stagewise=False):
        self._active_kappa_d=float(opt.kappa_d)
        self._active_smax=float(opt.smax)
        self._v72_ordinary_restoration=bool(opt.enable_ordinary_restoration)
        t0=time.perf_counter();self._active_hessian_mode=opt.hessian_mode;self._stagewise_active=stagewise;x=jnp.asarray(x_init);u=jnp.asarray(u_init)
        if x.shape!=(self.N+1,self.nx) or u.shape!=(self.N,self.nu):raise ValueError('shape')
        # fixed initial state
        x=x.at[0].set(self.x0)
        hs=self._ineq_vmap_jit(x[:-1],u) if self.ni else jnp.zeros((self.N,0),dtype=x.dtype)
        s=self._push_slacks(hs,opt) if self.ni else hs
        lam=jnp.zeros((self.N,self.nx),dtype=x.dtype);beta=jnp.zeros((self.N,self.ne),dtype=x.dtype);gamma=jnp.zeros((self.N,self.ni),dtype=x.dtype)
        flo,fhi,Sl,Su=self._margins(s) if self.ni else (jnp.zeros_like(s,dtype=bool),)*2+(s,s)
        if self.ni:
            flo=jnp.broadcast_to(flo,s.shape); fhi=jnp.broadcast_to(fhi,s.shape)
        zL=jnp.where(flo,1.0,0.0);zU=jnp.where(fhi,1.0,0.0)
        mu=float(opt.mu_init);trace=[];accepted=0;acceptable_count=0;last_reg=0.0;last_nonzero_reg=0.0
        lin=self._linearize(x,u,stagewise)
        if opt.initialize_dual:
            lam0,beta0,gamma0,_dual_init_ok=self._initial_dual_estimate(
                lin,zL,zU,opt.linear_backend,opt.lammax)
            lam,beta,gamma=lam0,beta0,gamma0
        rd,re,rh=self._constraints_from_linearization(lin,s);theta_l1=self._theta(rd,re,rh,'l1');filt=_Filter(theta_l1,opt)
        # Cache the metrics produced while recording an accepted iterate.  The
        # next outer iteration starts from that exact same point and barrier
        # value, so recomputing them immediately would launch and synchronize
        # the same device program twice.
        last_metrics=None

        # Official prints iter 0 before a step.
        def record(it,alpha_pr=0,alpha_du=0,ls=0,typ='x',reg=0):
            nonlocal lin,last_metrics
            em,dui,pri,comp=self._residual_measure(lin,x,u,s,lam,beta,gamma,zL,zU,mu)
            # ``pri`` is exactly the L-infinity constraint violation used for
            # the printed ``cv`` field.
            obj=self._objective(x,u);cv=pri
            last_metrics=(em,dui,pri,comp)
            row=dict(iter=it,obj=obj,cv=cv,du=dui,mu=mu,reg=reg,alpha_pr=alpha_pr,alpha_du=alpha_du,ls=ls,type=typ,e_mu=em,comp=comp)
            trace.append(row)
            if opt.print_level:
                rg='-.-' if reg==0 else f'{math.log10(reg):4.1f}'
                print(f"{it:3d} {obj:14.7e} {cv:9.2e} {dui:9.2e} {math.log10(mu):5.1f} {rg:>5} {alpha_du:9.2e} {alpha_pr:9.2e} {ls:2d}{typ}",flush=True)
            if opt.iteration_callback is not None:opt.iteration_callback(row,np.asarray(x),np.asarray(u))
            return em
        record(0)
        em,dui,pri,comp=last_metrics
        status='max_iter'
        shortened_count=0
        watchdog_active=False
        watchdog_steps=0
        watchdog_checkpoint=None
        for it in range(1,opt.max_iter+1):
            filt.current_iteration=it
            # Monotone barrier update; repeat without consuming an NLP iteration.
            mu_changed=False
            while em < opt.kappa_eta*mu and mu>opt.mu_min:
                # Match IpMonotoneMuUpdate: the complementarity tolerance
                # provides the lower floor for the monotone barrier sequence.
                mu_floor=min(opt.tol,opt.compl_inf_tol)/(opt.kappa_eta+1.0)
                mun=max(opt.mu_min,mu_floor,
                        min(opt.kappa_mu*mu,mu**opt.theta_mu))
                if mun>=mu*(1-1e-15):break
                mu=mun;mu_changed=True
                em,dui,pri,comp=self._residual_measure(lin,x,u,s,lam,beta,gamma,zL,zU,mu)
            if mu_changed:
                # FATROP resets filter entries and successive-rejection history
                # when entering a new barrier subproblem.  The run-level number
                # of filter resets remains capped by max_filter_resets.
                filt.reset()
                shortened_count=0
                watchdog_active=False;watchdog_steps=0;watchdog_checkpoint=None
            if em <= opt.tol:
                status='success';break
            if em <= opt.acceptable_tol: acceptable_count+=1
            else: acceptable_count=0
            if acceptable_count>=opt.acceptable_iter:
                status='acceptable';break
            # Official FATROP enters watchdog mode after ten consecutive
            # shortened iterations.  Save the complete primal-dual point and
            # filter state; trial points during watchdog are compared against
            # this checkpoint rather than only the immediately previous point.
            if (opt.enable_watchdog and opt.max_watchdog_steps>0 and
                (not watchdog_active) and
                shortened_count>=opt.watchdog_shortened_iter_trigger):
                watchdog_active=True;watchdog_steps=0
                # FATROP's watchdog stores the *current iterate* and keeps the
                # corresponding line-search reference quantities fixed while
                # it temporarily allows otherwise rejected first trial points.
                # The directional derivative / alpha test depend on the search
                # direction computed below, so fill those two fields after the
                # inertia-corrected direction is available.
                watchdog_checkpoint=dict(
                    x=x,u=u,s=s,lam=lam,beta=beta,gamma=gamma,zL=zL,zU=zU,
                    mu=mu,last_reg=last_reg,last_nonzero_reg=last_nonzero_reg,
                    filter_entries=list(filt.entries),
                    theta=self._theta(
                        *self._constraints_from_linearization(lin,s),'l1'),
                    phi=self._barrier_phi(x,u,s,mu),
                    dphi=None,alpha_test=None)
            # inertia correction loop
            # FATROP inertia-correction schedule.  There are two distinct
            # growth regimes in the official implementation:
            #   * if the previous accepted iteration was unregularized, first
            #     try delta_w=0, then introduce delta_w0 and grow it by the
            #     aggressive kappa_wplusem factor;
            #   * if the previous iteration already used regularization, start
            #     from kappa_wmin*last_reg and, if needed, grow by kappa_wplus.
            # This distinction is observable in the official trace: with the
            # defaults delta_w0=1e-4 and kappa_wplusem=100 the first nonzero
            # search is 1e-4 -> 1e-2 -> 1.0, not the 8x sequence used by the
            # earlier prototype.
            had_previous_reg=(last_nonzero_reg>0.0)
            # FATROP always probes the *unregularized* system first on a new
            # NLP iteration.  The previous successful delta_w is only used to
            # choose the first nonzero correction if that zero-shift probe is
            # still indefinite.  This matters after a regularized iteration:
            # the very next iteration can return to delta_w=0 immediately.
            reg=0.0
            sol=None
            introduced_nonzero=False
            step_metrics=None
            kd=jnp.asarray(float(getattr(self,'_active_kappa_d',0.0)),dtype=x.dtype)
            for ir in range(opt.max_inertia_tries):
                sol,ds,dzL,dzU,lin_used=self._direction(lin,x,u,s,lam,beta,gamma,zL,zU,mu,reg,opt.linear_backend)
                step_metrics=self._current_step_metrics_jit(
                    lin,x,u,s,zL,zU,sol,ds,dzL,dzU,
                    jnp.asarray(mu,dtype=x.dtype),kd)
                step_metrics_host=jax.device_get(step_metrics)
                direction_ok=step_metrics_host[0]
                if bool(direction_ok):
                    break
                if reg==0.0:
                    reg=(max(opt.delta_wmin,opt.kappa_wmin*last_nonzero_reg)
                         if had_previous_reg else opt.delta_w0)
                    introduced_nonzero=True
                else:
                    grow=(opt.kappa_wplus if had_previous_reg else opt.kappa_wplusem)
                    reg=max(opt.delta_wmin,grow*reg)
            else:
                status='inertia_failure';break
            last_reg=reg
            if reg>0.0:
                last_nonzero_reg=reg
            _,apmax,admax,phi0,th0,dphi=step_metrics_host
            apmax=float(apmax);admax=float(admax);phi0=float(phi0)
            th0=float(th0);dphi=float(dphi)
            if watchdog_active and watchdog_checkpoint is not None and watchdog_checkpoint['dphi'] is None:
                watchdog_checkpoint['dphi']=dphi
                watchdog_checkpoint['alpha_test']=apmax
                # V29_EXACT_WATCHDOG_RETRY
                # Ipopt/FATROP saves the checkpoint search direction.  If the
                # watchdog excursion later fails, ordinary backtracking resumes
                # from this SAME direction rather than solving a new KKT system.
                watchdog_checkpoint['saved_sol']=sol
                watchdog_checkpoint['saved_ds']=ds
                watchdog_checkpoint['saved_dzL']=dzL
                watchdog_checkpoint['saved_dzU']=dzU
                watchdog_checkpoint['saved_apmax']=apmax
                watchdog_checkpoint['saved_admax']=admax
                watchdog_checkpoint['saved_reg']=reg
                # Some installed v9/v16 branches structurally retain dormant
                # direct-bound dual directions.  Preserve them if present, but
                # do not require them.
                if 'dzuL' in locals():
                    watchdog_checkpoint['saved_dzuL']=dzuL
                if 'dzuU' in locals():
                    watchdog_checkpoint['saved_dzuU']=dzuU
                if 'zuL' in locals():
                    watchdog_checkpoint['saved_zuL']=zuL
                if 'zuU' in locals():
                    watchdog_checkpoint['saved_zuU']=zuU
            # Filter backtracking.  In watchdog mode FATROP compares the
            # *first* trial only to the saved checkpoint.  If that point fails
            # the filter, watchdog mode still takes it provisionally rather
            # than backtracking.  This is the mechanism that lets FATROP escape
            # a long sequence of tiny filter steps.
            accepted_trial=False;alpha=apmax;lsn=0;ftype=False;soc_used=False
            watchdog_rejected_trial=False
            old=(x,u,s,lam,beta,gamma,zL,zU)
            if watchdog_active and watchdog_checkpoint is not None:
                wb=watchdog_checkpoint
                th_ref=float(wb['theta']);phi_ref=float(wb['phi'])
                dphi_ref=float(wb['dphi']);alpha_test=float(wb['alpha_test'])
            else:
                th_ref=th0;phi_ref=phi0;dphi_ref=dphi;alpha_test=None
            first_trial_res=None; first_trial_theta=None
            # FATROP's minimum backtracking threshold is tied to the
            # directional barrier decrease and the current violation.  It is
            # only a termination floor; the first trial is always evaluated.
            alpha_min=opt.gamma_theta
            if dphi < 0.0:
                if th0 > 0.0:
                    alpha_min=min(opt.gamma_theta,
                                  opt.gamma_phi*th0/(-dphi))
                    if th0 <= filt.theta_min:
                        alpha_min=min(alpha_min,
                                      opt.delta*th0**opt.s_theta/
                                      ((-dphi)**opt.s_phi))
            alpha_min*=opt.alpha_min_frac
            # Experimental device-side ordinary line search.  SOC and
            # watchdog cases use the source-faithful Python loop below.
            if (opt.use_gpu_line_search and not watchdog_active and
                    opt.max_soc==0):
                nfilter=max(opt.max_iter+1,len(filt.entries),1)
                ft=jnp.full((nfilter,),jnp.inf,dtype=x.dtype)
                fp=jnp.full((nfilter,),jnp.inf,dtype=x.dtype)
                if filt.entries:
                    ef=jnp.asarray(filt.entries,dtype=x.dtype)
                    ft=ft.at[:len(filt.entries)].set(ef[:,0])
                    fp=fp.at[:len(filt.entries)].set(ef[:,1])
                kd_dev=jnp.asarray(float(getattr(self,'_active_kappa_d',0.0)),dtype=x.dtype)
                gpu_ls=self._gpu_line_search_jit(
                    x,u,s,sol.dx[1:],sol.du,ds,
                    jnp.asarray(apmax,dtype=x.dtype),jnp.asarray(th_ref,dtype=x.dtype),
                    jnp.asarray(phi_ref,dtype=x.dtype),jnp.asarray(dphi_ref,dtype=x.dtype),
                    jnp.asarray(mu,dtype=x.dtype),kd_dev,ft,fp,
                    jnp.asarray(len(filt.entries),dtype=jnp.int32),
                    jnp.asarray(filt.theta_max,dtype=x.dtype),
                    jnp.asarray(filt.theta_min,dtype=x.dtype),
                    jnp.asarray(opt.s_phi,dtype=x.dtype),jnp.asarray(opt.delta,dtype=x.dtype),
                    jnp.asarray(opt.s_theta,dtype=x.dtype),jnp.asarray(opt.gamma_theta,dtype=x.dtype),
                    jnp.asarray(opt.gamma_phi,dtype=x.dtype),jnp.asarray(opt.eta_phi,dtype=x.dtype),
                    jnp.asarray(opt.gamma_alpha,dtype=x.dtype),
                    jnp.asarray(opt.max_ls,dtype=jnp.int32))
                # Fetch only scalar decisions.  Accepted trajectories and
                # residual arrays stay as JAX device buffers; fetching the
                # whole tuple here would copy O(N) data to the host every NLP
                # iteration and then copy it back on the next direction solve.
                (ls_count,alpha_out,gpu_accepted,phit,tht,valid,first_phi,
                 first_theta,first_valid)=jax.device_get((
                     gpu_ls[0],gpu_ls[1],gpu_ls[2],gpu_ls[9],gpu_ls[10],
                     gpu_ls[11],gpu_ls[15],gpu_ls[16],gpu_ls[17]))
                xt,ut,st=gpu_ls[3],gpu_ls[4],gpu_ls[5]
                first_rd,first_re,first_rh=gpu_ls[12],gpu_ls[13],gpu_ls[14]
                lsn=int(ls_count)
                alpha=float(alpha_out)
                ad=admax
                first_trial_res=(first_rd,first_re,first_rh)
                first_trial_theta=float(first_theta)
                if bool(gpu_accepted):
                    # This host call updates filter bookkeeping only; all
                    # candidate evaluations and acceptance tests were on GPU.
                    ok,ftype=filt.trial(
                        float(tht),float(phit),th_ref,phi_ref,apmax,dphi_ref,
                        alpha_update=apmax)
                    if not ok:
                        raise RuntimeError('device and host filter decisions differ')
                    x,u,s=xt,ut,st
                    lam=lam+alpha*sol.dlambda_dyn
                    beta=beta+alpha*sol.dbeta_eq
                    gamma=gamma+alpha*sol.dgamma_ineq
                    zL=zL+ad*dzL
                    zU=zU+ad*dzU
                    accepted_trial=True

            for lsn in range(1,opt.max_ls+1 if not accepted_trial else 1):
                # Upstream stops backtracking below alpha_min, while always
                # evaluating the first maximum feasible trial.
                if lsn > 1 and alpha <= alpha_min:
                    break
                ad=admax
                xt=(x+alpha*sol.dx).at[0].set(self.x0);ut=u+alpha*sol.du;st=s+alpha*ds
                # Official FatropData::update_trial_step(alpha_primal, alpha_dual)
                # advances the equality/path multipliers with alpha_primal;
                # alpha_dual is used only for the simple-bound multipliers z.
                lamt=lam+alpha*sol.dlambda_dyn;betat=beta+alpha*sol.dbeta_eq;gammat=gamma+alpha*sol.dgamma_ineq
                zLt=zL+ad*dzL;zUt=zU+ad*dzU
                trial_res,phit,tht,_trial_feasible=self._trial_eval(
                    xt,ut,st,mu)
                if lsn==1:
                    first_trial_res=trial_res;first_trial_theta=tht
                # FATROP freezes alpha_primal_test across all backtracks.
                # Ordinary: current alpha_primal_max. Watchdog: saved test alpha.
                test_alpha=(alpha_test if watchdog_active and alpha_test is not None else apmax)
                ok,ftype=filt.trial(
                    tht,phit,th_ref,phi_ref,test_alpha,dphi_ref,
                    alpha_update=apmax)
                if ok:
                    x,u,s,lam,beta,gamma,zL,zU=xt,ut,st,lamt,betat,gammat,zLt,zUt
                    accepted_trial=True
                    # An acceptable watchdog excursion commits immediately.
                    if watchdog_active:
                        watchdog_active=False;watchdog_steps=0;watchdog_checkpoint=None
                    break

                if watchdog_active:
                    # FATROP does not backtrack a watchdog trial.  It accepts
                    # the first maximum-feasible point provisionally even when
                    # the ordinary filter rejects it, retaining the original
                    # checkpoint/reference for the next watchdog iteration.
                    x,u,s,lam,beta,gamma,zL,zU=xt,ut,st,lamt,betat,gammat,zLt,zUt
                    accepted_trial=True;watchdog_rejected_trial=True
                    break

                # FATROP starts SOC after a rejected trial whose nonlinear
                # constraint violation increased.  The official algorithm does
                # *not* add a zero-stationarity correction to the Newton step.
                # Instead it repeatedly resolves the same KKT matrix for a
                # fresh total direction.  The dual/complementarity RHS stays at
                # the current iterate while only the primal constraint RHS is
                # accumulated as
                #     g_soc <- g_soc + alpha_soc * g_trial,
                # initialized with g_soc = g_current.
                if lsn==1 and opt.max_soc>0 and tht>th0:
                    base_alpha=alpha
                    soc_acc=self._constraints_from_linearization(lin,s)
                    trial_soc_res=trial_res
                    theta_soc_old=tht
                    alpha_soc=base_alpha
                    for _soc_it in range(opt.max_soc):
                        soc_acc=tuple(a+alpha_soc*b for a,b in zip(soc_acc,trial_soc_res))
                        ssol,sds,sdzL,sdzU=self._soc_direction(
                            lin_used,x,u,s,lam,beta,gamma,zL,zU,mu,reg,
                            soc_acc,opt.linear_backend)
                        if not bool(self._direction_ok_jit(
                                ssol,sds,sdzL,sdzU)):
                            break
                        # FATROP recomputes the primal fraction-to-boundary for
                        # the fresh SOC direction and forms the trial directly
                        # from the *current* iterate.
                        alpha_soc_new,ad_soc=self._max_step(
                            s,zL,zU,sds,sdzL,sdzU,mu)
                        xts=(x+alpha_soc_new*ssol.dx).at[0].set(self.x0)
                        uts=u+alpha_soc_new*ssol.du
                        sts=s+alpha_soc_new*sds
                        new_soc_res,phis,ths,valid=self._trial_eval(
                            xts,uts,sts,mu)
                        if not valid:break
                        # FATROP tests an SOC point with the original rejected
                        # alpha_primal_test, even though alpha_soc_new may differ.
                        oks,fts=filt.trial(
                            ths,phis,th_ref,phi_ref,base_alpha,dphi,
                            alpha_update=apmax)
                        if oks:
                            x,u,s=xts,uts,sts
                            lam=lam+alpha_soc_new*ssol.dlambda_dyn
                            beta=beta+alpha_soc_new*ssol.dbeta_eq
                            gamma=gamma+alpha_soc_new*ssol.dgamma_ineq
                            zL=zL+ad_soc*sdzL;zU=zU+ad_soc*sdzU
                            accepted_trial=True;soc_used=True;ftype=fts
                            alpha=alpha_soc_new;ad=ad_soc
                            break
                        # Continue only while the corrected trial improves
                        # constraint violation by FATROP's kappa_soc factor.
                        if ths>=opt.kappa_soc*theta_soc_old:break
                        theta_soc_old=ths
                        trial_soc_res=new_soc_res
                        alpha_soc=alpha_soc_new
                    if accepted_trial:break
                alpha*=opt.gamma_alpha
            if not accepted_trial:
                if watchdog_active and watchdog_checkpoint is not None:
                    wb=watchdog_checkpoint
                    x,u,s=wb['x'],wb['u'],wb['s'];lam,beta,gamma=wb['lam'],wb['beta'],wb['gamma']
                    zL,zU=wb['zL'],wb['zU'];mu=wb['mu'];last_reg=wb['last_reg'];last_nonzero_reg=wb['last_nonzero_reg']
                    filt.entries=list(wb['filter_entries']);watchdog_active=False;watchdog_steps=0;watchdog_checkpoint=None;shortened_count=0
                    lin=self._linearize(x,u,stagewise)
                    record(it,0.0,0.0,lsn,'x',reg)
                    continue
                if self._v72_ordinary_restoration:
                    if opt.print_level:
                        print(f"V72_ORDINARY_RESTORATION_ENTER it={it} J={self._objective(x,u):.9e} cv={self._theta(*self._constraints(x,u,s),'linf'):.9e}",flush=True)
                    _rr=self._v72_full_restoration(x=x,u=u,s=s,lam=lam,beta=beta,gamma=gamma,zL=zL,zU=zU,mu=mu,opt=opt,stagewise=stagewise,it=it)
                    if bool(_rr.success):
                        x,u,s=_rr.x,_rr.u,_rr.s
                        lam,beta,gamma=_rr.lambda_dyn,_rr.beta_eq,_rr.gamma_ineq
                        zL,zU=_rr.zL,_rr.zU
                        lin=self._linearize(x,u,stagewise)
                        _th_rest=self._theta(
                            *self._constraints_from_linearization(lin,s),'l1')
                        filt=_Filter(_th_rest,opt)
                        shortened_count=0;acceptable_count=0
                        last_reg=0.0;last_nonzero_reg=0.0
                        watchdog_active=False;watchdog_steps=0;watchdog_rejected_trial=False;watchdog_checkpoint=None
                        accepted+=int(_rr.iterations)
                        _r_alpha=(float(_rr.rows[-1].get('alpha_pr',0.0)) if getattr(_rr,'rows',None) else 0.0)
                        record(it,_r_alpha,0.0,max(1,int(_rr.iterations)),'r',0.0)
                        if opt.print_level:
                            print(f"V72_ORDINARY_RESTORATION_HANDOFF it={it} J={self._objective(x,u):.9e} cv={self._theta(*self._constraints(x,u,s),'linf'):.9e} steps={int(_rr.iterations)}",flush=True)
                        continue
                    status='restoration_failure'
                    if opt.print_level:
                        print(f"V72_ORDINARY_RESTORATION_FAILED it={it} reason={_rr.reason}",flush=True)
                    break
                status='line_search_failure';break

            # Watchdog bookkeeping: source-faithful exhaustion path.
            # V69_SOURCE_FAITHFUL_WATCHDOG_RESTORATION
            # FATROP permits max_watchdog_steps-1 visible provisional `w` rows.
            # On the next rejected watchdog trial it restores the saved iterate,
            # retries ordinary backtracking in the SAME line search with the
            # first (maximum-step) point skipped, and only then enters
            # restoration if the retry remains unacceptable.
            if watchdog_active and watchdog_rejected_trial:
                watchdog_steps+=1
                if watchdog_steps>=opt.max_watchdog_steps:
                    if watchdog_checkpoint is not None:
                        wb=watchdog_checkpoint
                        # Restore the checkpoint primal/dual/filter state.
                        x,u,s=wb['x'],wb['u'],wb['s']
                        lam,beta,gamma=wb['lam'],wb['beta'],wb['gamma']
                        zL,zU=wb['zL'],wb['zU']
                        mu=float(wb['mu'])
                        last_reg=float(wb['last_reg'])
                        last_nonzero_reg=float(wb['last_nonzero_reg'])
                        filt.entries=list(wb['filter_entries'])
                        if 'zuL' in locals() and 'saved_zuL' in wb:
                            zuL=wb['saved_zuL']
                        if 'zuU' in locals() and 'saved_zuU' in wb:
                            zuU=wb['saved_zuU']

                        # Restore the exact search direction saved when watchdog
                        # was armed.  This is FATROP stop_watchdog()+
                        # skip_first_trial_point semantics, not a fresh KKT solve.
                        sol=wb['saved_sol']; ds=wb['saved_ds']
                        dzL=wb['saved_dzL']; dzU=wb['saved_dzU']
                        apmax=float(wb['saved_apmax']); admax=float(wb['saved_admax'])
                        reg=float(wb['saved_reg'])
                        if 'dzuL' in locals() and 'saved_dzuL' in wb:
                            dzuL=wb['saved_dzuL']
                        if 'dzuU' in locals() and 'saved_dzuU' in wb:
                            dzuU=wb['saved_dzuU']

                        watchdog_active=False; watchdog_steps=0
                        watchdog_rejected_trial=False; shortened_count=0
                        accepted_trial=False; soc_used=False
                        th_ref=float(wb['theta']); phi_ref=float(wb['phi'])
                        dphi_ref=float(wb['dphi'])
                        alpha=apmax*opt.gamma_alpha
                        ad=admax
                        for retry_lsn in range(2,opt.max_ls+1):
                            xt=(x+alpha*sol.dx).at[0].set(self.x0)
                            ut=u+alpha*sol.du
                            st=s+alpha*ds
                            lamt=lam+alpha*sol.dlambda_dyn
                            betat=beta+alpha*sol.dbeta_eq
                            gammat=gamma+alpha*sol.dgamma_ineq
                            zLt=zL+ad*dzL; zUt=zU+ad*dzU
                            trial_res,phit,tht,_trial_feasible=self._trial_eval(
                                xt,ut,st,mu)
                            # Same frozen alpha_primal_test semantics as
                            # the ordinary line search: retry skips the first actual
                            # point but still tests with the original apmax.
                            ok,ftype=filt.trial(
                                tht,phit,th_ref,phi_ref,apmax,dphi_ref,
                                alpha_update=apmax)
                            if ok:
                                x,u,s=xt,ut,st
                                lam,beta,gamma=lamt,betat,gammat
                                zL,zU=zLt,zUt
                                accepted_trial=True; lsn=retry_lsn
                                break
                            alpha*=opt.gamma_alpha

                        if opt.print_level:
                            print(
                                f"V69_WATCHDOG_ROLLBACK_RETRY it={it} "
                                f"accepted={accepted_trial} alpha={alpha:.9e} "
                                f"ls={lsn}",flush=True)

                        watchdog_checkpoint=None
                        if not accepted_trial:
                            # The ordinary rollback retry failed.  Reuse the
                            # already validated v37+v43 full restoration KKT.
                            from . import restoration_v32 as _v69_rv
                            if not getattr(_v69_rv,'_v44_hard_dynamics_installed',False):
                                from .fatrop_v43_runtime_patch import install_v43_hard_dynamics
                                install_v43_hard_dynamics(_v69_rv)
                                _v69_rv._v44_hard_dynamics_installed=True

                            if opt.print_level:
                                print(
                                    f"V69_RESTORATION_ENTER it={it} "
                                    f"J={self._objective(x,u):.9e} "
                                    f"cv={self._theta(*self._constraints(x,u,s),'linf'):.9e}",
                                    flush=True)

                            _rr=_v69_rv.full_restoration(
                                self,x=x,u=u,s=s,
                                lam=lam,beta=beta,gamma=gamma,
                                zL=zL,zU=zU,mu_orig=mu,opt=opt,
                                stagewise=stagewise,
                                outer_iteration=max(0,it-1),
                                backend=opt.linear_backend,
                                print_rows=bool(opt.print_level))
                            if not bool(_rr.success):
                                watchdog_active=False; watchdog_steps=0
                                watchdog_checkpoint=None; shortened_count=0
                                status='restoration_failure'
                                if opt.print_level:
                                    print(f"V69_RESTORATION_FAILED it={it} reason={_rr.reason}",flush=True)
                                break

                            x=_rr.x; u=_rr.u; s=_rr.s
                            lam=_rr.lambda_dyn; beta=_rr.beta_eq; gamma=_rr.gamma_ineq
                            zL=_rr.zL; zU=_rr.zU
                            mu=float(wb['mu'])

                            # FATROP clears restoration/watchdog memory here.
                            # Do NOT arm a watchdog at the restored point.
                            lin=self._linearize(x,u,stagewise)
                            _th_rest=self._theta(
                                *self._constraints_from_linearization(lin,s),'l1')
                            _phi_rest=self._barrier_phi(x,u,s,mu)
                            filt=_Filter(_th_rest,opt)
                            shortened_count=0; acceptable_count=0
                            last_reg=0.0; last_nonzero_reg=0.0
                            watchdog_active=False; watchdog_steps=0
                            watchdog_rejected_trial=False; watchdog_checkpoint=None
                            accepted+=int(_rr.iterations)
                            _r_alpha=(float(_rr.rows[-1].get('alpha_pr',0.0))
                                      if getattr(_rr,'rows',None) else 0.0)
                            record(it,_r_alpha,0.0,max(1,int(_rr.iterations)),'r',0.0)
                            if opt.print_level:
                                print(
                                    f"V69_RESTORATION_HANDOFF it={it} "
                                    f"J={self._objective(x,u):.9e} "
                                    f"cv={self._theta(*self._constraints(x,u,s),'linf'):.9e} "
                                    f"steps={int(_rr.iterations)}",flush=True)
                            continue

            # FATROP's watchdog counter is based on whether ordinary
            # backtracking occurred (n_steps>0), not on alpha<1.  In our loop
            # lsn==1 means the first trial was accepted, so lsn>1 is the exact
            # analogue.  SOC accepted from the first trial likewise leaves the
            # count at zero.
            if lsn>1 and not soc_used and not watchdog_rejected_trial:
                shortened_count+=1
            else:
                shortened_count=0
            # A provisionally accepted rejected watchdog point must not alter
            # the filter.  A genuinely accepted watchdog point uses the saved
            # reference iterate when augmenting it.
            if not watchdog_rejected_trial and not ftype:
                filt.add(th_ref,phi_ref)

            # V82_KAPPA_SIGMA_ACCEPTED_STEP_CLAMP
            # FATROP/Ipopt AcceptTrialPoint safeguards every inequality-slack
            # bound multiplier around the primal estimate mu / slack:
            #   mu/(kappa_sigma*d) <= z <= kappa_sigma*mu/d.
            # Without this, z can drift by many orders of magnitude, driving
            # Sigma=zL/Sl+zU/Su huge and W=Sigma^{-1} nearly singular.
            if self.ni:
                _ks=float(opt.kappa_sigma)
                _flo,_fhi,_Sl,_Su=self._margins(s)
                _tiny=jnp.finfo(s.dtype).tiny
                _Sl_safe=jnp.maximum(_Sl,_tiny); _Su_safe=jnp.maximum(_Su,_tiny)
                _zL_lo=mu/(_ks*_Sl_safe); _zL_hi=_ks*mu/_Sl_safe
                _zU_lo=mu/(_ks*_Su_safe); _zU_hi=_ks*mu/_Su_safe
                _zL_new=jnp.where(_flo,jnp.minimum(jnp.maximum(zL,_zL_lo),_zL_hi),0.0)
                _zU_new=jnp.where(_fhi,jnp.minimum(jnp.maximum(zU,_zU_lo),_zU_hi),0.0)
                _corrL=jnp.where(_flo,jnp.abs(_zL_new-zL),0.0)
                _corrU=jnp.where(_fhi,jnp.abs(_zU_new-zU),0.0)
                _maxcorr=float(jnp.maximum(jnp.max(_corrL,initial=0.0),jnp.max(_corrU,initial=0.0)))
                _nclip=int(jnp.sum((_corrL>0)|(_corrU>0)))
                zL,zU=_zL_new,_zU_new
                if _maxcorr>0.0 and opt.print_level:
                    print(f'V82_KAPPA_SIGMA it={it} nclip={_nclip} maxcorr={_maxcorr:.9e}',flush=True)
            accepted+=1
            lin=self._linearize(x,u,stagewise)
            itype=(('F' if ftype else 'H') if soc_used else ('w' if watchdog_rejected_trial else ('f' if ftype else 'h')))
            record(it,alpha,ad,lsn,itype,reg)
        dt=time.perf_counter()-t0
        return FatropRefResult(np.asarray(x),np.asarray(u),np.asarray(s),np.asarray(lam),np.asarray(beta),np.asarray(gamma),np.asarray(zL),np.asarray(zU),self._objective(x,u),mu,len(trace)-1,accepted,status,dt,trace)
