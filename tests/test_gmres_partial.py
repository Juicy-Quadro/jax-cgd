import jax.numpy as jnp
import numpy as np
from jax_cgd import solvers


def test_gmres_partial_cycle_uses_only_completed_basis_vectors():
    A=jnp.diag(jnp.array([1.,1.1,1.2]));b=jnp.ones(3)
    x,st=solvers.GMRES(tol=.1,atol=0.,max_iter=1,restart=3).solve_with_stats(A,b)
    Ab=A@b
    expected=b*jnp.vdot(b,Ab)/jnp.vdot(Ab,Ab)
    assert int(st.iterations)==1 and bool(st.converged)
    np.testing.assert_allclose(x,expected,rtol=1e-12,atol=1e-12)
