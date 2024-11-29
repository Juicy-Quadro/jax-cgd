import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.sparse.linalg import cg
class CG:
    def __init__(self, tol=1e-10, atol=1e-16, max_iter=None):
        '''  Conjugate gradient method, implementation based on
            https://en.wikipedia.org/wiki/Conjugate_gradient_method.
            Only works for positive definite matrices A! '''

        self.tol = tol
        self.atol = atol
        self.max_iter = max_iter

    def solve(self, limap, b, x0=None):
        return solve_jit(limap, b, x0, self.tol, self.atol, self.max_iter)

# @partial(jax.jit, static_argnums=(3,4,5))
def solve_jit(limap, b, x0, tol, atol, max_iter):
    return cg(limap, b, tol=tol, atol=atol, M=None, maxiter=max_iter, x0=x0)

# # Test case

# A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
# b = jnp.array([5.0, 4.0])
# A2 = jnp.array([[4.0, 2.0], [2.0, 3.0]])
# b2 = jnp.array([6.0, 5.0])

# # def limap(x):
# #     return A @ x
# import time
# start = time.perf_counter()
# solution, info = solve_jit(A, b, None, 1e-10, 1e-16, 1000)
# print("Solution:", solution)
# print('Time taken: ', time.perf_counter() - start)
# start = time.perf_counter()
# solution, info = solve_jit(A2, b2, None, 1e-10, 1e-16, 1000)
# print("Solution:", solution)
# print('Time taken: ', time.perf_counter() - start)