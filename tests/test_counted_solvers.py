import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_cgd import solvers


@pytest.mark.parametrize('cls',[solvers.CG,solvers.GMRES])
def test_one_iteration_and_true_residual(cls):
    b=jnp.array([1.,2.,3.]); s=cls(tol=1e-10,atol=0,max_iter=10)
    x,st=jax.jit(lambda b:s.solve_with_stats(lambda v:2*v,b))(b)
    np.testing.assert_allclose(x,b/2,atol=1e-12)
    assert int(st.iterations)==1 and bool(st.converged)
    assert int(st.matvecs)==2  # one Krylov product + true residual, zero x0 omitted
    np.testing.assert_allclose(st.residual_norm,jnp.linalg.norm(b-2*x),atol=1e-14)


@pytest.mark.parametrize('cls',[solvers.CG,solvers.GMRES])
def test_zero_rhs_and_exact_warm_start(cls):
    s=cls(tol=1e-10,atol=0,max_iter=10)
    for b,x0 in [(jnp.zeros(3),None),(jnp.ones(3),jnp.ones(3)/2)]:
        x,st=s.solve_with_stats(lambda v:2*v,b,x0)
        assert int(st.iterations)==0 and bool(st.converged)
        assert not bool(st.breakdown)
        np.testing.assert_allclose(x,b/2,atol=1e-14)


@pytest.mark.parametrize('cls',[solvers.CG,solvers.GMRES])
def test_dense_accuracy_and_implicit_grad(cls):
    A=jnp.array([[4.,1.,0.],[1.,3.,1.],[0.,1.,2.]])
    b=jnp.array([1.,-2.,3.]);s=cls(tol=1e-10,max_iter=10)
    fn=lambda b:s.solve_with_stats(lambda v:A@v,b)[0]
    np.testing.assert_allclose(jax.jit(fn)(b),np.linalg.solve(A,b),atol=1e-9)
    np.testing.assert_allclose(jax.grad(lambda b:jnp.sum(fn(b)))(b),
                               np.linalg.solve(np.asarray(A).T,np.ones(3)),atol=1e-9)
    np.testing.assert_allclose(jax.jit(jax.vmap(fn))(jnp.stack([b,2*b])),
                               jnp.stack([fn(b),2*fn(b)]),atol=1e-9)


def test_cg_iteration_limit_and_breakdown():
    b=jnp.ones(3); A=jnp.diag(jnp.array([1.,3.,9.]))
    _,st=solvers.CG(tol=1e-12,max_iter=1).solve_with_stats(A,b)
    assert int(st.iterations)==1 and not bool(st.converged)
    _,st=solvers.CG().solve_with_stats(lambda v:-v,b)
    assert bool(st.breakdown) and not bool(st.converged)


def test_gmres_nonsymmetric_restarts_and_inner_count():
    A=jnp.array([[3.,2.,0.],[-1.,4.,1.],[0.,-1.,2.]])
    b=jnp.array([1.,2.,-1.])
    _,st=solvers.GMRES(tol=1e-14,max_iter=2,restart=1).solve_with_stats(A,b)
    assert int(st.iterations)==2 and int(st.restart_cycles)==2
    assert int(st.matvecs)==4 and not bool(st.converged)
    x,st=solvers.GMRES(tol=1e-10,max_iter=30,restart=2).solve_with_stats(A,b)
    assert bool(st.converged)
    np.testing.assert_allclose(x,np.linalg.solve(A,b),atol=1e-9)


@pytest.mark.parametrize('cls',[solvers.CG,solvers.GMRES])
def test_counts_match_executed_matvecs(cls):
    calls=[]
    def A(v):
        jax.debug.callback(lambda _:calls.append(1),v[0])
        return jnp.array([2.,3.,5.])*v
    x,st=jax.jit(lambda b:cls(tol=1e-10,max_iter=6).solve_with_stats(A,b))(jnp.ones(3))
    jax.block_until_ready((x,st));jax.effects_barrier()
    assert len(calls)==int(st.matvecs)


@pytest.mark.parametrize('cls',[solvers.CG,solvers.GMRES])
def test_nonfinite_and_zero_budget(cls):
    for b in [jnp.array([jnp.nan,1.]),jnp.array([jnp.inf,1.])]:
        _,st=cls(max_iter=2).solve_with_stats(lambda v:v,b)
        assert not bool(st.converged)
    _,st=cls(tol=1e-12,max_iter=0).solve_with_stats(lambda v:2*v,jnp.ones(3))
    assert int(st.iterations)==0 and not bool(st.converged)


def test_gmres_nonsymmetric_adjoint_and_matrix_derivative():
    A=jnp.array([[4.,2.,0.],[-1.,3.,1.],[0.,0.,2.]])
    b=jnp.array([.2,-.3,.4]);s=solvers.GMRES(tol=1e-11,max_iter=8)
    fun=lambda a,b:s.solve_with_stats(a,b)[0]
    np.testing.assert_allclose(jax.grad(lambda z:jnp.sum(fun(A,z)))(b),
                               np.linalg.solve(np.asarray(A).T,np.ones(3)),atol=1e-10)
    actual=jax.grad(lambda scale:jnp.sum(fun(scale*A,b)))(jnp.array(2.))
    expected=-jnp.sum(jnp.linalg.solve(A,b))/4
    np.testing.assert_allclose(actual,expected,atol=1e-10)


@pytest.mark.parametrize('cls',[solvers.CG,solvers.GMRES])
def test_float32_and_heterogeneous_vmap(cls):
    s=cls(tol=1e-5,atol=0,max_iter=10)
    b=jnp.array([[0.,0.,0.],[1.,2.,3.]],dtype=jnp.float32)
    x,st=jax.jit(jax.vmap(lambda b:s.solve_with_stats(lambda v:2*v,b)))(b)
    np.testing.assert_allclose(x,b/2,atol=2e-6)
    np.testing.assert_array_equal(st.iterations,[0,1])
    assert bool(jnp.all(st.converged))
