import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_cgd import ACGD, BCGD, solvers, utils


def f(x,y):
    return jnp.sum(jnp.sin(x['w'])*y['w']**2)+0.1*jnp.sum(x['w']**2)


def make(cls, **kwargs):
    return cls({'w':jnp.array([.2,.5,-.1])}, {'w':jnp.array([.3,-.4,.6])}, f,
               lr=0.03,solver=solvers.CG(tol=1e-10,atol=0),**kwargs)


@pytest.mark.parametrize('cls',[BCGD,ACGD])
def test_step_many_matches_repeated_steps_and_accounting(cls):
    a,b=make(cls),make(cls)
    ks=[];mvs=[]
    for _ in range(5):
        a.step();st=a.get_stats();ks.append(int(st['krylov_iterations']));mvs.append(int(st['matvecs']))
    b.step_many(5)
    np.testing.assert_allclose(b.x_params,a.x_params,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(b.y_params,a.y_params,rtol=1e-10,atol=1e-12)
    st=b.get_stats()
    assert int(st['outer_steps'])==5 and b.timestep==5
    assert int(st['krylov_iterations_total'])==sum(ks)
    assert int(st['matvecs_total'])==sum(mvs)
    assert int(st['mixed_products_nominal_total'])==2*(sum(mvs)+5)
    assert int(st['mixed_products_total'])+int(st['mixed_products_reused_total'])==2*(sum(mvs)+5)
    assert int(st['paper_proxy_total'])==2*5+2*sum(ks)
    assert bool(st['counts_available'])


def test_fused_linearization():
    x,y=jnp.array([.1,.2,.3]),jnp.array([.4,-.3,.2])
    fun=lambda a,b:jnp.sum(jnp.sin(a)*jnp.cos(b))+jnp.sum(a*a)
    gx,gy,hxy,hyx=utils.linearize_game(fun,x,y)
    ex,ey=jax.grad(fun,(0,1))(x,y)
    np.testing.assert_allclose(gx,ex);np.testing.assert_allclose(gy,ey)
    H=jax.jacfwd(jax.grad(fun,0),1)(x,y)
    np.testing.assert_allclose(hxy(y),H@y,atol=1e-14)
    np.testing.assert_allclose(hyx(x),H.T@x,atol=1e-14)


def test_factorized_game_exact_derivatives_and_optimizer():
    from jax_cgd.games import BilinearGame
    r=lambda x:jnp.concatenate([jnp.sin(x['w']),x['w']**2])
    d=lambda y:jnp.concatenate([y['w']**3,jnp.cos(y['w'])])
    game=BilinearGame(r,d)
    x,y={'w':jnp.array([.2,-.1])},{'w':jnp.array([.1,.4])}
    opts=[ACGD(x,y,fn,lr=.01,solver=solvers.CG(tol=1e-10))
          for fn in (game,lambda x,y:game(x,y))]
    for opt in opts:opt.step_many(3)
    np.testing.assert_allclose(opts[0].x_params,opts[1].x_params,atol=1e-11)
    np.testing.assert_allclose(opts[0].y_params,opts[1].y_params,atol=1e-11)
    assert opts[0].get_stats()['factorized_game']


def test_custom_solver_reports_unknown_counts():
    class Dense:
        def solve(self,A,b,x0=None):
            return jnp.linalg.solve(jax.vmap(A,in_axes=1,out_axes=1)(jnp.eye(b.size)),b),None
    opt=BCGD({'w':jnp.ones(2)},{'w':jnp.ones(2)},f,solver=Dense())
    opt.step_many(2);st=opt.get_stats()
    assert not bool(st['counts_available'])
    assert int(st['krylov_iterations_total'])==-1
    assert int(st['paper_proxy_total'])==-1


def test_stats_count_nonconvergence():
    opt=BCGD({'w':jnp.ones(3)},{'w':jnp.array([.1,1.,10.])},f,
              lr=1.,solver=solvers.CG(tol=1e-14,max_iter=0),on_failure="accept_inexact")
    opt.step_many(2)
    assert int(opt.get_stats()['failed_solves_total'])==2
    assert not bool(opt.get_stats()['converged'])


@pytest.mark.parametrize('cls',[BCGD,ACGD])
def test_block_gmres_matches_schur_and_counts(cls):
    x={'w':jnp.array([.2,.5,-.1])};y={'w':jnp.array([.3,-.4,.6])}
    a=cls(x,y,f,lr=.03,solver=solvers.CG(tol=1e-11,atol=0))
    b=cls(x,y,f,lr=.03,solver=solvers.GMRES(tol=1e-11,atol=0),formulation='block')
    a.step_many(3);b.step_many(3)
    np.testing.assert_allclose(b.x_params,a.x_params,rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose(b.y_params,a.y_params,rtol=1e-9,atol=1e-11)
    st=b.get_stats()
    assert bool(st['paper_formula_applicable'])
    assert int(st['mixed_products_total'])==2*int(st['matvecs_total'])


def test_cg_rejects_nonsymmetric_block_system():
    with pytest.raises(ValueError,match='GMRES'):
        BCGD({'w':jnp.ones(2)},{'w':jnp.ones(2)},f,solver=solvers.CG(),formulation='block')
