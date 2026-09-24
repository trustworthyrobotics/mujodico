"""Serial and PAS generalized Riccati solvers for FATROP's reduced PD system.

The reduced stage KKT after eliminating inequality slacks and bound duals is

 [ R      F.T    Gu.T ] [du   ]   [ S.T ]
 [ F       0       0  ] [dbeta] + [ E   ] dx
 [ Gu      0      -W  ] [dgamma]   [ Gx  ]

       + [B.T;0;0] lambda_{k+1} + [r;e;gtilde] = 0,

where W = Sigma^{-1} is positive diagonal.  This matches FATROP's reduced
inequality equation G d - Sigma^{-1} dgamma + gtilde = 0.

The PAS backend forms a fixed-size primal-dual boundary relation and composes
adjacent segments by Schur elimination.  The serial backend is a conventional
backward generalized Riccati recursion.  They are algebraically identical.
"""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp

Array = jax.Array

class LinearSolution(NamedTuple):
    dx: Array
    du: Array
    dlambda_dyn: Array
    dbeta_eq: Array
    dgamma_ineq: Array
    inertia_ok: Array


def _solve(a,b):
    return jnp.linalg.solve(a,b)


def _diagonal_values(matrix_or_diagonal):
    """Accept the legacy dense-diagonal representation or a compact vector."""
    return (matrix_or_diagonal if matrix_or_diagonal.ndim == 1
            else jnp.diag(matrix_or_diagonal))


def _local_factor(Q,S,R,q,r,A,B,a,E,F,e,Gx,Gu,g,W,reg):
    """Eliminate [du, hard-eq dual, inequality dual] conditional on x/lambda+ ."""
    nx=Q.shape[0]; nu=R.shape[0]; ne=F.shape[0]; ni=Gu.shape[0]
    Qr=Q+reg*jnp.eye(nx,dtype=Q.dtype)
    Rr=R+reg*jnp.eye(nu,dtype=R.dtype)
    if ni:
        # Eliminate the diagonal inequality-dual block before the dense solve.
        # D = W^-1 = Sigma.  This is the augmented-system reduction used by
        # FATROP and shrinks the local solve from nu+ne+ni to nu+ne.
        D=1.0/_diagonal_values(W)
        DGx=D[:,None]*Gx; DGu=D[:,None]*Gu; Dg=D*g
        Qr=Qr+Gx.T@DGx
        Sr=S+Gx.T@DGu
        Rr=Rr+Gu.T@DGu
        qr=q+Gx.T@Dg
        rr=r+Gu.T@Dg
    else:
        D=jnp.zeros((0,),dtype=Q.dtype);Sr=S;qr=q;rr=r
    z_ee=jnp.zeros((ne,ne),dtype=R.dtype)
    M=jnp.block([[Rr,F.T],[F,z_ee]])
    Kx=jnp.concatenate([Sr.T,E],axis=0)
    k0=jnp.concatenate([rr,e],axis=0)
    Kl=jnp.concatenate([B.T,jnp.zeros((ne,nx),dtype=B.dtype)],axis=0)
    rhs=jnp.concatenate([Kx,k0[:,None],Kl],axis=1)
    Zeq=_solve(M,rhs)
    if ni:
        Zu=Zeq[:nu]
        Zgx=D[:,None]*(Gu@Zu[:,:nx]-Gx)
        zg0=D*(Gu@Zu[:,nx]-g)
        Zgl=D[:,None]*(Gu@Zu[:,nx+1:])
        Zx=jnp.concatenate([Zeq[:,:nx],Zgx],axis=0)
        z0=jnp.concatenate([Zeq[:,nx],zg0],axis=0)
        Zl=jnp.concatenate([Zeq[:,nx+1:],Zgl],axis=0)
    else:
        Zx=Zeq[:,:nx];z0=Zeq[:,nx];Zl=Zeq[:,nx+1:]
    Kx_full=jnp.concatenate([S.T,E,Gx],axis=0)
    Bbar=jnp.concatenate([B,jnp.zeros((nx,ne+ni),dtype=B.dtype)],axis=1)
    P=(Q+reg*jnp.eye(nx,dtype=Q.dtype))-Kx_full.T@Zx
    p=q-Kx_full.T@z0
    C=Bbar@Zl
    Ac=A-Bbar@Zx
    bc=a-Bbar@z0
    return (0.5*(P+P.T),p,0.5*(C+C.T),Ac,bc,Zx,z0,Zl)


def _compose(e2,e1):
    P1,p1,C1,A1,b1=e1
    P2,p2,C2,A2,b2=e2
    n=P1.shape[0]; I=jnp.eye(n,dtype=P1.dtype)
    L=I+P2@C1
    # Factor each interface matrix once and solve all right-hand sides in a
    # single call.  Associative scan invokes this at every tree node, so the
    # previous five separate solves paid repeated LU costs throughout the tree.
    XL=_solve(L,jnp.concatenate(
        [P2@A1,(p2+P2@b1)[:,None],A2.T],axis=1))
    Xpa=XL[:,:n];Xp=XL[:,n];Xat=XL[:,n+1:]
    # R = I + C1 P2 is L.T for symmetric P2/C1.  Avoid factoring R:
    #   R^-1 C1 = C1 L^-1
    #   R^-1 A1 = A1 - C1 L^-1 P2 A1
    #   R^-1(b1-C1 p2) = b1-C1 L^-1(p2+P2 b1).
    Xca=C1@Xat
    Xa=A1-C1@Xpa
    Xb=b1-C1@Xp
    P=P1+A1.T@Xpa
    p=p1+A1.T@Xp
    C=C2+A2@Xca
    Ao=A2@Xa
    bo=b2+A2@Xb
    return 0.5*(P+P.T),p,0.5*(C+C.T),Ao,bo


def _inertia_from_M(M,nu,nd,tol=1e-10):
    ev=jnp.linalg.eigvalsh(0.5*(M+M.T))
    scale=jnp.maximum(1.0,jnp.max(jnp.abs(ev)))
    tt=tol*scale
    return (jnp.sum(ev>tt)==nu) & (jnp.sum(ev<-tt)==nd)


def _fatrop_reduced_pd_ok(Huu, F, Gu, W, pivot_tol=1e-8):
    """Mirror FATROP's reduced-Cholesky regularization test.

    FATROP does not classify the eigenvalues of the complete saddle matrix.
    It eliminates the ``-W`` inequality block, projects stage equalities out
    of the control block, and checks Cholesky pivots of the remaining primal
    Hessian.  Full-saddle sign counting is particularly unreliable once the
    contact equality and bound-multiplier channels become ill-conditioned.
    """
    nu = Huu.shape[0]
    ne = F.shape[0]
    H = 0.5 * (Huu + Huu.T)
    if Gu.shape[0]:
        # W is diagonal by construction.  A dense batched solve here dominated
        # the GPU inertia check for the 32-contact Fingers task and needlessly
        # materialized an O(ni^3) factorization at every stage.
        inverse_diagonal = 1.0 / _diagonal_values(W)
        H = H + Gu.T @ (inverse_diagonal[:, None] * Gu)
        H = 0.5 * (H + H.T)
    rank_ok = jnp.array(True)
    if ne:
        q, r = jnp.linalg.qr(F.T, mode='complete')
        equality_pivots = jnp.abs(jnp.diag(r[:ne, :ne]))
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(F)))
        rank_ok = jnp.all(equality_pivots > 1e-12 * scale)
        null_basis = q[:, ne:]
    else:
        null_basis = jnp.eye(nu, dtype=H.dtype)
    if null_basis.shape[1] == 0:
        return rank_ok
    reduced = null_basis.T @ H @ null_basis
    reduced = 0.5 * (reduced + reduced.T)
    chol = jnp.linalg.cholesky(reduced)
    pivots = jnp.diag(chol)
    finite = jnp.all(jnp.isfinite(chol)) & jnp.all(pivots > 0.0)
    reference = jnp.maximum(jnp.abs(jnp.diag(reduced)),
                            jnp.finfo(reduced.dtype).tiny)
    stable = jnp.all(pivots * pivots > pivot_tol * reference)
    return rank_ok & finite & stable


def solve_pas(*,Q,S,R,q,r,A,B,a,E,F,e,Gx,Gu,g,W,Qf,qf,dx0,regularization=0.0):
    N,nx,_=Q.shape; nu=R.shape[-1]; ne=F.shape[1]; ni=Gu.shape[1]
    fac=jax.vmap(_local_factor,in_axes=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,None))(
        Q,S,R,q,r,A,B,a,E,F,e,Gx,Gu,g,W,regularization)
    P,p,C,Ac,bc,Zx,z0,Zl=fac
    Qfr=Qf+regularization*jnp.eye(nx,dtype=Q.dtype)
    Pall=jnp.concatenate([P,Qfr[None]],0)
    pall=jnp.concatenate([p,qf[None]],0)
    Call=jnp.concatenate([C,jnp.zeros((1,nx,nx),dtype=C.dtype)],0)
    Aall=jnp.concatenate([Ac,jnp.zeros((1,nx,nx),dtype=Ac.dtype)],0)
    ball=jnp.concatenate([bc,jnp.zeros((1,nx),dtype=bc.dtype)],0)
    VP,Vp,_,_,_=jax.lax.associative_scan(jax.vmap(_compose),(Pall,pall,Call,Aall,ball),reverse=True)
    def gain(Ck,Ak,bk,Pn,pn):
        L=jnp.eye(nx,dtype=Ck.dtype)+Ck@Pn
        X=_solve(L,jnp.concatenate([(bk-Ck@pn)[:,None],Ak],axis=1))
        return X[:,0],X[:,1:]
    kk,KK=jax.vmap(gain)(C,Ac,bc,VP[1:],Vp[1:])
    ks=jnp.concatenate([dx0[None],kk],0)
    Ks=jnp.concatenate([jnp.eye(nx,dtype=Q.dtype)[None],KK],0)
    def acomp(x,y):
        k1,K1=x; k2,K2=y
        return k2+K2@k1,K2@K1
    dx,_=jax.lax.associative_scan(jax.vmap(acomp),(ks,Ks))
    lnext=jax.vmap(lambda Pn,pn,xn:Pn@xn+pn)(VP[1:],Vp[1:],dx[1:])
    zz=-(jax.vmap(lambda zx,zv,zl,xk,lk:zx@xk+zv+zl@lk)(Zx,z0,Zl,dx[:-1],lnext))
    du=zz[:,:nu]; be=zz[:,nu:nu+ne]; ga=zz[:,nu+ne:]

    # FATROP inertia correction is defined on the *reduced* local saddle
    # matrix after incorporating the downstream value Hessian V_{k+1}.
    # The serial Riccati recursion tests exactly this matrix during its
    # backward sweep.  The reverse associative scan has already produced all
    # suffix value Hessians VP[k+1], so the identical tests can be evaluated
    # independently and in parallel without introducing a sequential pass.
    Iu=jnp.eye(nu,dtype=R.dtype)
    def reduced_inertia(Rk,Bk,Fk,Guk,Wk,Vn):
        Huu=Rk+regularization*Iu+Bk.T@Vn@Bk
        return _fatrop_reduced_pd_ok(Huu,Fk,Guk,Wk)
    oks=jax.vmap(reduced_inertia)(R,B,F,Gu,W,VP[1:])
    ok=jnp.all(oks)
    return LinearSolution(dx,du,lnext,be,ga,ok)


def solve_serial(*,Q,S,R,q,r,A,B,a,E,F,e,Gx,Gu,g,W,Qf,qf,dx0,regularization=0.0):
    """Serial generalized Riccati recursion using two ``lax.scan`` sweeps."""
    N,nx,_=Q.shape; nu=R.shape[-1]; ne=F.shape[1]; ni=Gu.shape[1]
    Ix=jnp.eye(nx,dtype=Q.dtype); Iu=jnp.eye(nu,dtype=Q.dtype)
    Vterm=Qf+regularization*Ix

    def backward(carry,inp):
        Vn,vn=carry
        Qk,Sk,Rk,qk,rk,Ak,Bk,ak,Ek,Fk,ek,Gxk,Guk,gk,Wk=inp
        Hxx=Qk+regularization*Ix+Ak.T@Vn@Ak
        Hxu=Sk+Ak.T@Vn@Bk
        Huu=Rk+regularization*Iu+Bk.T@Vn@Bk
        vv=Vn@ak+vn
        hx=qk+Ak.T@vv
        hu=rk+Bk.T@vv
        if ni:
            D=1.0/_diagonal_values(Wk)
            DGx=D[:,None]*Gxk;DGu=D[:,None]*Guk;Dg=D*gk
            Hxx=Hxx+Gxk.T@DGx
            Hxu=Hxu+Gxk.T@DGu
            Huu=Huu+Guk.T@DGu
            hx=hx+Gxk.T@Dg
            hu=hu+Guk.T@Dg
        M=jnp.block([[Huu,Fk.T],
                     [Fk,jnp.zeros((ne,ne),dtype=Q.dtype)]])
        Kx=jnp.concatenate([Hxu.T,Ek],axis=0)
        k0=jnp.concatenate([hu,ek],axis=0)
        Z=_solve(M,jnp.concatenate([Kx,k0[:,None]],axis=1))
        Zx=Z[:,:nx];z0=Z[:,nx]
        Vk=Hxx-Kx.T@Zx;vk=hx-Kx.T@z0
        Vk=.5*(Vk+Vk.T)
        ok=_fatrop_reduced_pd_ok(Huu,Fk,Guk,Wk)
        return (Vk,vk),(Vk,vk,ok)

    inputs=(Q,S,R,q,r,A,B,a,E,F,e,Gx,Gu,g,W)
    rev=tuple(jnp.flip(z,axis=0) for z in inputs)
    (_, _),(Vrev,vrev,okrev)=jax.lax.scan(backward,(Vterm,qf),rev)
    Vs=jnp.flip(Vrev,axis=0);vs=jnp.flip(vrev,axis=0)
    Vall=jnp.concatenate([Vs,Vterm[None]],axis=0)
    vall=jnp.concatenate([vs,qf[None]],axis=0)
    oks=jnp.flip(okrev,axis=0)

    def forward(xk,inp):
        Qk,Sk,Rk,qk,rk,Ak,Bk,ak,Ek,Fk,ek,Gxk,Guk,gk,Wk,Vn,vn=inp
        Hxu=Sk+Ak.T@Vn@Bk
        Huu=Rk+regularization*Iu+Bk.T@Vn@Bk
        vv=Vn@ak+vn
        hu=rk+Bk.T@vv
        if ni:
            D=1.0/_diagonal_values(Wk)
            DGu=D[:,None]*Guk
            Hxu=Hxu+Gxk.T@DGu
            Huu=Huu+Guk.T@DGu
            hu=hu+Guk.T@(D*gk)
        M=jnp.block([[Huu,Fk.T],
                     [Fk,jnp.zeros((ne,ne),dtype=Q.dtype)]])
        Kx=jnp.concatenate([Hxu.T,Ek],axis=0)
        k0=jnp.concatenate([hu,ek],axis=0)
        zz=-_solve(M,Kx@xk+k0)
        duk=zz[:nu];bek=zz[nu:nu+ne]
        gak=((1.0/_diagonal_values(Wk))*(Guk@duk+Gxk@xk+gk)
             if ni else jnp.zeros((0,),dtype=Q.dtype))
        xn=Ak@xk+Bk@duk+ak
        lk=Vn@xn+vn
        return xn,(xn,duk,lk,bek,gak)
    finputs=(Q,S,R,q,r,A,B,a,E,F,e,Gx,Gu,g,W,Vall[1:],vall[1:])
    _,(xtail,du,lam,be,ga)=jax.lax.scan(forward,dx0,finputs)
    dx=jnp.concatenate([dx0[None],xtail],axis=0)
    return LinearSolution(dx,du,lam,be,ga,jnp.all(oks))
