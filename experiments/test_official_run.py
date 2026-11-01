import pytest
from experiments.official_reference import *
from experiments.official_adapter import OfficialCodeGMRES

@pytest.mark.parametrize('kind',['normal','maxcap','tiny','zero','one','warm_ignored'])
def test_solver_matches_pinned(kind):
    n=7;rng=np.random.default_rng(52);A=np.eye(n)+rng.normal(size=(n,n))*.2;b=rng.normal(size=n);cap=n
    if kind=='maxcap':cap=2
    if kind=='tiny':b*=1e-11
    if kind=='zero':b*=0
    if kind=='one':A=2*np.eye(n)
    At,bt=torch.tensor(A),torch.tensor(b);count=[0]
    def mv(v):count[0]+=1;return At@v
    xt,(kt,h)=torch_gmres(mv,bt,x0=None,tol=1e-10,atol=1e-20,max_iter=cap,track=True)
    jj=OfficialCodeGMRES(tol=1e-10,atol=1e-20,capacity=cap,official_maxiter=cap)
    x,st=jax.jit(lambda bb:jj.solve_with_stats(lambda v:jnp.asarray(A)@v,bb,jnp.ones_like(bb)))(jnp.asarray(b))
    np.testing.assert_allclose(x,xt.numpy(),atol=1e-12,rtol=1e-10)
    assert int(st.iterations)==count[0]
    assert int(jj.reference_index(st))==int(kt)


def test_player_permutation():
    rng=np.random.default_rng(7);n=9;nx=4;A=np.eye(n)+rng.normal(size=(n,n))*.15;b=rng.normal(size=n)
    jj=OfficialCodeGMRES(tol=1e-12,atol=1e-20,capacity=n,official_maxiter=n,n_primal=nx)
    x,st=jax.jit(lambda bb:jj.solve_with_stats(lambda v:jnp.asarray(A)@v,bb,None))(jnp.asarray(b))
    np.testing.assert_allclose(x,np.linalg.solve(A,b),rtol=1e-11,atol=1e-12)
    assert int(st.iterations)<=n


def test_adapter_integrates_with_public_acgd():
    A=jnp.array([[1.,.3],[-.2,2.]]);f=lambda x,y:jnp.sin(x)@A@jnp.tanh(y)
    o=ACGD(jnp.array([.2,.3]),jnp.array([-.1,.5]),f,lr=.001,beta=.99,eps=1e-8,
        solver=OfficialCodeGMRES(capacity=4,official_maxiter=4,n_primal=2),formulation='block',on_failure='accept_inexact')
    o.step_many(3)
    assert int(o.timestep)==3


def test_filebacked_H_preserves_reference():
    from experiments.official_run import observe_reference
    mod=importlib.import_module('CGDs.gmres_acgd');b=torch.tensor([1.,2.,3.])
    args=dict(Avp=lambda v:2*v,b=b,tol=1e-10,atol=1e-20,max_iter=3000,track=True)
    x0,info0=mod.GMRES(**args);rows=[]
    with observe_reference(rows):x1,info1=mod.GMRES(**args)
    assert torch.equal(x0,x1) and info0==info1
    assert rows[0]['filebacked_H'] and rows[0]['arnoldi']==1
