"""Exact forward spatial derivatives must match autodiff, including parameter AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from examples.spatial_jets import JetMLP as MLP


@pytest.mark.parametrize('burgers,axes', [(False,(0,1)), (True,(0,))])
def test_spatial_jet_values_and_parameter_derivatives(burgers,axes):
    model=MLP(2,5,activation='tanh',burgers_input=burgers)
    points=jnp.array([[-.2,.15],[.3,.8],[.8,.4]],dtype=jnp.float64)
    params=model.init(jax.random.PRNGKey(5),points)
    assert hasattr(model,'spatial_jet'), 'Missing exact Flax spatial derivative path'
    def reference(p):
        point=lambda z:model.apply(p,z[None,:])[0,0]
        u=jax.vmap(point)(points)
        du=jax.vmap(jax.grad(point))(points)
        h=jax.vmap(jax.hessian(point))(points)
        return u,du,sum(h[:,i,i] for i in axes)
    def optimized(p):
        return model.apply(p,points,second_axes=axes,method=model.spatial_jet)
    for expected,actual in zip(reference(params),jax.jit(optimized)(params)):
        np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
    def loss(out):
        u,du,lap=out
        return jnp.sum((u+du[:,0]+du[:,1]+lap)**2)
    expected=jax.grad(lambda p:loss(reference(p)))(params)
    actual=jax.jit(jax.grad(lambda p:loss(optimized(p))))(params)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)
    tangent=jax.tree.map(lambda a:jnp.full_like(a,.03),params)
    er=jax.jvp(reference,(params,),(tangent,))[1]
    ar=jax.jvp(optimized,(params,),(tangent,))[1]
    for a,b in zip(er,ar):
        np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-12)


@pytest.mark.parametrize('problem',['poisson','burgers'])
def test_jet_game_matches_reference_game_and_mixed_products(problem):
    from examples.cpinn_counted import make_problem
    from jax_cgd import utils
    p,d,_,ref,factor,_=make_problem(problem,6,4,4,1,123,spatial_jet=True)
    np.testing.assert_allclose(factor(p,d),ref(p,d),rtol=1e-11,atol=1e-12)
    x,xm=utils.params_to_array(p);y,ym=utils.params_to_array(d)
    fn=lambda x,y:ref(utils.params_to_dict(x,xm),utils.params_to_dict(y,ym))
    gx,gy,hxy,hyx=factor.flattened(xm,ym).linearize(x,y)
    rx,ry=jax.grad(fn,(0,1))(x,y)
    for a,b in [(gx,rx),(gy,ry),(hxy(y),utils.H_xy_vy(fn,x,y,y)),(hyx(x),utils.H_yx_vx(fn,x,y,x))]:
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)


@pytest.mark.parametrize('problem,seed',[('poisson',7),('burgers',19)])
def test_jet_updates_match_reference(problem,seed):
    from examples.cpinn_counted import make_problem
    from jax_cgd import ACGD,solvers
    args=(problem,9,4,5,2,seed)
    p,d,_,_,old,_=make_problem(*args)
    _,_,_,_,jet,_=make_problem(*args,spatial_jet=True)
    opts=[ACGD(p,d,g,lr=1e-3,beta=.99,eps=1e-8,solver=solvers.CG(tol=1e-10,atol=0,max_iter=100))
          for g in (old,jet)]
    for opt in opts: opt.step_many(10)
    np.testing.assert_allclose(opts[0].x_params,opts[1].x_params,rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(opts[0].y_params,opts[1].y_params,rtol=1e-8,atol=1e-10)
    assert all(int(o.get_stats()['failed_solves_total'])==0 for o in opts)


def test_jet_rejects_unsupported_network_and_axes():
    coords=jnp.ones((2,2))
    for model in [MLP(1,4,activation='relu'),MLP(1,4,out_features=2)]:
        p=model.init(jax.random.PRNGKey(1),coords)
        with pytest.raises(ValueError,match='scalar tanh'):
            model.apply(p,coords,method=model.spatial_jet)
    model=MLP(1,4)
    p=model.init(jax.random.PRNGKey(1),coords)
    for axes in [(),(2,),(0,0)]:
        with pytest.raises(ValueError,match='second_axes'):
            model.apply(p,coords,second_axes=axes,method=model.spatial_jet)
