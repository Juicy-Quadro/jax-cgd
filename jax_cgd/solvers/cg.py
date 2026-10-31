"""CG for SPD systems. Legacy solve() plus an explicitly counted solve."""
import math
from jax.scipy.sparse.linalg import cg
from ._counted import counted_solve


class CG:
    def __init__(self, tol=1e-10, atol=1e-16, max_iter=None):
        if not math.isfinite(tol) or not math.isfinite(atol) or min(tol, atol) < 0:
            raise ValueError('tolerances must be finite and nonnegative')
        if max_iter is not None and (not isinstance(max_iter, int) or max_iter < 0):
            raise ValueError('max_iter must be a nonnegative integer or None')
        self.tol, self.atol, self.max_iter = tol, atol, max_iter

    def solve(self, limap, b, x0=None):
        return solve_jit(limap, b, x0, self.tol, self.atol, self.max_iter)

    def solve_with_stats(self, limap, b, x0=None):
        return counted_solve(limap, b, x0, tol=self.tol, atol=self.atol,
                             maxiter=self.max_iter, method='cg')


def solve_jit(limap, b, x0, tol, atol, max_iter):
    return cg(limap, b, x0=x0, tol=tol, atol=atol, maxiter=max_iter)
