import jax
import jax.numpy as jnp
from functools import partial
import jax.lax as lax
from jax.scipy.sparse.linalg import gmres

class GMRES:
    def __init__(self, tol=1e-10, atol=1e-16, max_iter=1000):
        '''
        Generalized Minimal Residual Method (GMRES).
        Only works for non-singular matrices A.
        '''
        self.tol = tol
        self.atol = atol
        self.max_iter = max_iter

    def solve(self, limap, b, x0=None):
        return solve_jit(limap, b, x0, self.tol, self.atol, self.max_iter)
    
# @partial(jax.jit, static_argnums=(0,))
def solve_jit(limap, b, x0, tol, atol, max_iter):
    return gmres(limap, b, tol=tol, atol=atol, M=None, maxiter=max_iter, x0=x0)


# # Test case

# A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
# b = jnp.array([5.0, 4.0])

# def limap(x):
#     return A @ x

# import time 
# start = time.perf_counter()
# x, info = solve_jit(limap, b, None, 1e-10, 1e-16, 1000)
# print(x, info)
# print('Time taken: ', time.perf_counter() - start)

# start = time.perf_counter()
# x, info = solve_jit(limap, b, None, 1e-10, 1e-16, 1000)
# print(x, info)
# print('Time taken: ', time.perf_counter() - start)