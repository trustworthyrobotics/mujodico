"""Regression tests for serial/PAS reduced KKT algebra."""
import jax
import jax.numpy as jnp
import numpy as np

from mujodico.trajopt._src.fatrop_ref.linear import solve_pas, solve_serial

jax.config.update('jax_enable_x64', True)


def _problem():
    rng=np.random.default_rng(7)
    N,nx,nu,ne,ni=8,3,4,1,2
    A=np.broadcast_to(np.eye(nx)*0.85,(N,nx,nx)).copy()
    A+=rng.normal(scale=0.02,size=A.shape)
    B=rng.normal(scale=0.1,size=(N,nx,nu))
    E=rng.normal(scale=0.1,size=(N,ne,nx))
    # Keep equality rows independent through a fixed control pivot.
    F=rng.normal(scale=0.1,size=(N,ne,nu));F[:,:,0]+=1.0
    Gx=rng.normal(scale=0.2,size=(N,ni,nx))
    Gu=rng.normal(scale=0.2,size=(N,ni,nu))
    Q=np.broadcast_to(np.eye(nx),(N,nx,nx)).copy()
    R=np.broadcast_to(np.eye(nu)*1.5,(N,nu,nu)).copy()
    S=rng.normal(scale=0.02,size=(N,nx,nu))
    args=dict(
        Q=Q,S=S,R=R,q=rng.normal(size=(N,nx)),
        r=rng.normal(size=(N,nu)),A=A,B=B,
        a=rng.normal(scale=0.1,size=(N,nx)),E=E,F=F,
        e=rng.normal(scale=0.1,size=(N,ne)),Gx=Gx,Gu=Gu,
        g=rng.normal(scale=0.1,size=(N,ni)),
        W=rng.uniform(0.2,2.0,size=(N,ni)),Qf=np.eye(nx)*2.0,
        qf=rng.normal(size=nx),dx0=np.zeros(nx),regularization=1e-5)
    return {k:jnp.asarray(v) if isinstance(v,np.ndarray) else v
            for k,v in args.items()}


def test_pas_matches_serial_with_compact_inequality_diagonal():
    args=_problem()
    serial=solve_serial(**args)
    pas=solve_pas(**args)
    for lhs,rhs in zip(serial[:-1],pas[:-1]):
        np.testing.assert_allclose(lhs,rhs,rtol=2e-9,atol=2e-9)
    assert bool(serial.inertia_ok)==bool(pas.inertia_ok)


def test_compact_diagonal_matches_legacy_dense_diagonal():
    args=_problem()
    compact=solve_pas(**args)
    dense_args={**args,'W':jax.vmap(jnp.diag)(args['W'])}
    dense=solve_pas(**dense_args)
    for lhs,rhs in zip(compact[:-1],dense[:-1]):
        np.testing.assert_allclose(lhs,rhs,rtol=2e-10,atol=2e-10)
