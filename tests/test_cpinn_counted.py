import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_cgd import ACGD,solvers
from examples.cpinn_counted import make_problem


@pytest.mark.parametrize('problem',['poisson','burgers'])
def test_flax_factorization_preserves_game_gradients_and_updates(problem):
    p,d,_,generic,factor,_=make_problem(problem,6,4,4,1,123)
    np.testing.assert_allclose(factor(p,d),generic(p,d),atol=1e-12)
    for a,b in zip(jax.tree_util.tree_leaves(jax.grad(factor,(0,1))(p,d)),
                   jax.tree_util.tree_leaves(jax.grad(generic,(0,1))(p,d))):
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
    opts=[ACGD(p,d,g,lr=1e-3,beta=.99,eps=1e-8,
               solver=solvers.CG(tol=1e-10,atol=0,max_iter=100)) for g in (generic,factor)]
    for opt in opts:opt.step_many(2)
    np.testing.assert_allclose(opts[0].x_params,opts[1].x_params,rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(opts[0].y_params,opts[1].y_params,rtol=1e-8,atol=1e-10)
    assert all(int(o.get_stats()['failed_solves_total'])==0 for o in opts)
