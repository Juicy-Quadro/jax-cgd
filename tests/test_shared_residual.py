"""Count actual mixed products, not the number of Python traces."""
import jax
import jax.numpy as jnp
import numpy as np
from jax_cgd import ACGD, BCGD, solvers, utils


def test_true_residual_reuses_dual_reconstruction(monkeypatch):
    original=utils.linearize_game
    calls=[]
    def instrument(f,x,y):
        gx,gy,hxy,hyx=original(f,x,y)
        def left(v):
            jax.debug.callback(lambda _:calls.append('hxy'),v[0])
            return hxy(v)
        def right(v):
            jax.debug.callback(lambda _:calls.append('hyx'),v[0])
            return hyx(v)
        return gx,gy,left,right
    monkeypatch.setattr(utils,'linearize_game',instrument)
    f=lambda x,y:jnp.sum(jnp.sin(x['w'])*y['w']**2)
    opt=ACGD({'w':jnp.array([.2,.1,-.4])},{'w':jnp.array([.4,.2,-.2])},f,
             lr=.01,solver=solvers.CG(tol=1e-10,atol=0))
    opt.step_many(2)
    jax.block_until_ready((opt.x_params,opt.y_params));jax.effects_barrier()
    st=opt.get_stats()
    assert 'mixed_products_reused_total' in st, 'Missing verified-residual reuse'
    assert int(st['mixed_products_reused_total'])==2
    assert int(st['mixed_products_total'])==len(calls)
    assert bool(st['converged']) and int(st['failed_solves_total'])==0


def test_reported_schur_residual_is_true():
    x=jnp.array([.1,.2,-.5]);y=jnp.array([.3,-.4,.7])
    f=lambda a,b:jnp.sum(jnp.sin(a)*b*b)
    opt=BCGD({'w':x},{'w':y},lambda a,b:f(a['w'],b['w']),lr=.2,
             solver=solvers.CG(tol=1e-11,atol=0))
    gx,gy=jax.grad(f,(0,1))(x,y)
    H=jax.jacfwd(jax.grad(f,0),1)(x,y)
    rhs=jnp.sqrt(.2)*(gx+H@(.2*gy))
    A=jnp.eye(3)+.04*H@H.T
    opt.step()
    actual=jnp.linalg.norm(rhs-A@opt.prev_sol)
    np.testing.assert_allclose(opt.get_stats()['residual_norm'],actual,atol=3e-16)


def test_zero_iterations_and_custom_cg_subclass_are_not_miscounted():
    zero=lambda x,y:jnp.sum(x['w'])*0+jnp.sum(y['w'])*0
    opt=BCGD({'w':jnp.ones(2)},{'w':jnp.ones(2)},zero,solver=solvers.CG())
    opt.step_many(2)
    st=opt.get_stats()
    assert int(st['krylov_iterations_total'])==0
    assert int(st['mixed_products_reused_total'])==0
    assert int(st['failed_solves_total'])==0
    class CustomCG(solvers.CG):
        def _solve_for_update(self,*args,**kwargs):
            raise AssertionError('Do not bypass a custom solver subclass')
    opt=BCGD({'w':jnp.ones(2)},{'w':jnp.ones(2)},zero,solver=CustomCG())
    opt.step()
    assert int(opt.get_stats()['mixed_products_reused_total'])==0


def test_inexact_cg_still_reports_failure():
    opt=BCGD({'w':jnp.ones(3)},{'w':jnp.array([.1,1.,10.])},
             lambda a,b:jnp.sum(jnp.sin(a['w'])*b['w']**2),lr=1.,
             solver=solvers.CG(tol=1e-14,atol=0,max_iter=1),on_failure="accept_inexact")
    opt.step()
    assert not bool(opt.get_stats()['converged'])
    assert int(opt.get_stats()['failed_solves_total'])==1
    assert int(opt.get_stats()['mixed_products_reused_total'])==1
