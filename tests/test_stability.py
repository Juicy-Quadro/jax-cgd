"""Release candidate failure contracts; successful legacy trajectories unchanged."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_cgd import ACGD, BCGD, solvers


def game(x, y):
    return jnp.sum(jnp.sin(x['w']) * y['w']**2)


def make(cls=ACGD, **kw):
    return cls({'w': jnp.array([.2, .7])}, {'w': jnp.array([.3, -.4])},
               game, lr=.03, **kw)


def arrays(opt):
    return [np.array(v) for v in (opt.x_params, opt.y_params, opt.vx, opt.vy, opt.prev_sol)]


def same_numeric(opt, before):
    for got, ref in zip(arrays(opt), before):
        np.testing.assert_array_equal(got, ref)


def test_default_rejects_inexact_without_state_pollution():
    opt = make(solver=solvers.CG(tol=1e-14, atol=0., max_iter=0))
    before = arrays(opt)
    with pytest.raises(RuntimeError, match='rejected'):
        opt.step_many(4)
    same_numeric(opt, before)
    assert int(opt.timestep) == 0
    st = opt.get_stats()
    assert int(st['attempted_steps_total']) == 1
    assert int(st['rejected_steps_total']) == 1
    assert int(st['failed_solves_total']) == 1


@pytest.mark.parametrize('policy', ['stop', 'accept_inexact'])
def test_explicit_policy_finite_nonconvergence(policy):
    opt = make(on_failure=policy, solver=solvers.CG(tol=1e-14, atol=0., max_iter=0))
    before = arrays(opt)
    opt.step_many(3)
    st = opt.get_stats()
    assert not bool(st['converged'])
    if policy == 'stop':
        same_numeric(opt, before)
        assert int(opt.timestep) == 0
        assert int(st['attempted_steps_total']) == 1
    else:
        assert int(opt.timestep) == 3
        assert int(st['failed_solves_total']) == 3
        assert int(st['rejected_steps_total']) == 0


@pytest.mark.parametrize('policy', ['raise', 'stop', 'accept_inexact'])
def test_nan_gradient_is_never_committed(policy):
    bad = lambda x, y: jnp.sqrt(-jnp.sum(x['w']**2)) + jnp.sum(y['w']**2)
    opt = ACGD({'w': jnp.ones(2)}, {'w': jnp.ones(2)}, bad, on_failure=policy)
    before = arrays(opt)
    if policy == 'raise':
        with pytest.raises(RuntimeError):
            opt.step_many(3)
    else:
        opt.step_many(3)
    same_numeric(opt, before)
    assert int(opt.timestep) == 0
    assert bool(opt.get_stats()['hard_failure'])
    assert int(opt.get_stats()['attempted_steps_total']) == 1


def test_failure_keeps_successful_prefix_and_stops_chunk():
    fun = lambda x,y: jnp.sqrt(x['w']).sum() + 0*jnp.sum(y['w'])
    def build():
        return BCGD({'w': jnp.array([.01])}, {'w': jnp.array([0.])}, fun, lr=.01)
    ref, opt = build(), build()
    ref.step()
    with pytest.raises(RuntimeError):
        opt.step_many(5)
    same_numeric(opt, arrays(ref))
    assert int(opt.timestep) == 1
    assert int(opt.get_stats()['attempted_steps_total']) == 2


def test_pure_state_update_jit_does_not_mutate_wrapper():
    opt = make(solver=solvers.CG(tol=1e-9, atol=0.))
    before = arrays(opt)
    state, stats = jax.jit(lambda s: opt.update(s, steps=4))(opt.state)
    same_numeric(opt, before)
    assert int(opt.timestep) == 0
    opt.step_many(4)
    for got, ref in zip(jax.tree.leaves(state), jax.tree.leaves(opt.state)):
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
    assert int(stats['outer_steps']) == 4


def test_pure_rejection_is_status_not_python_exception():
    opt = make(solver=solvers.CG(max_iter=0))
    state, st = jax.jit(lambda s: opt.update(s, steps=5))(opt.state)
    assert int(state.step) == 0
    assert bool(st['stopped'])
    assert int(st['attempted_steps_total']) == 1


def test_constructor_rejects_nonfinite_and_nonfloat():
    for value in (jnp.array([jnp.nan]), jnp.array([1]), jnp.array([1+2j]), jnp.zeros(0)):
        with pytest.raises((TypeError, ValueError)):
            BCGD({'w': value}, {'w': jnp.ones(1)}, game)


def test_nested_parameter_tree_matches_flat_and_preserves_structure():
    xp = {'layer': {'w': jnp.array([.2, .7])}, 'bias': (jnp.array([.1]),)}
    yp = {'layer': [jnp.array([.3, -.4, .6])]}
    def nested(x, y):
        a = jnp.concatenate((x['layer']['w'], x['bias'][0]))
        return jnp.vdot(a, y['layer'][0])
    opt = ACGD(xp, yp, nested, lr=.01)
    opt.step_many(3)
    assert jax.tree.structure(opt.get_infos()[2]) == jax.tree.structure(xp)
    assert jax.tree.structure(opt.get_infos()[3]) == jax.tree.structure(yp)
    assert int(opt.timestep) == 3


def test_pure_state_vmap_and_scan():
    opt=make(solver=solvers.CG(tol=1e-9,atol=1e-12))
    state=opt.state
    batch=jax.tree.map(lambda x:jnp.stack((x,x)),state)
    actual,stats=jax.jit(jax.vmap(lambda s:opt.update(s,steps=2)))(batch)
    reference,_=opt.update(state,steps=2)
    for got,ref in zip(jax.tree.leaves(actual),jax.tree.leaves(reference)):
        np.testing.assert_allclose(got[0],ref,rtol=1e-11,atol=1e-14)
    def body(s,_):
        s,diag=opt.update(s,steps=1)
        return s,diag['converged']
    scanned,converged=jax.jit(lambda s:jax.lax.scan(body,s,None,length=4))(state)
    direct,_=opt.update(state,steps=4)
    for a,b in zip(jax.tree.leaves(scanned),jax.tree.leaves(direct)):
        np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-14)
    assert np.all(converged)


def test_custom_solver_failure_is_checked_with_true_residual():
    class Wrong:
        def solve(self,A,b,x0=None):return jnp.zeros_like(b),None
    opt=make(solver=Wrong())
    before=arrays(opt)
    with pytest.raises(RuntimeError):opt.step()
    same_numeric(opt,before)
    assert not bool(opt.get_stats()['counts_available'])
    assert int(opt.get_stats()['failed_solves_total'])==1


def test_nonfinite_solution_cannot_be_accepted_even_if_solver_claims_success():
    from jax_cgd.solvers._counted import SolverStats
    class Dishonest:
        def solve_with_stats(self,A,b,x0=None):
            return jnp.full_like(b,jnp.inf),SolverStats(jnp.int32(0),jnp.int32(0),
                jnp.int32(0),jnp.array(0.),jnp.linalg.norm(b),jnp.bool_(True),jnp.bool_(False))
    opt=make(solver=Dishonest(),on_failure='accept_inexact');before=arrays(opt)
    opt.step_many(5);same_numeric(opt,before)
    assert bool(opt.get_stats()['hard_failure'])


def test_stop_does_not_repeat_failed_attempt_inside_chunk():
    opt=make(on_failure='stop',solver=solvers.CG(max_iter=0))
    opt.step_many(1000)
    assert int(opt.get_stats()['attempted_steps_total'])==1
    opt.step_many(1000)
    assert int(opt.get_stats()['attempted_steps_total'])==2
    assert int(opt.get_stats()['outer_steps'])==0
