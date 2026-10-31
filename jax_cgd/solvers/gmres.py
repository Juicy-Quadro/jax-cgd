"""GMRES with explicit restart-cycle and Arnoldi-step diagnostics."""
import math
from jax.scipy.sparse.linalg import gmres
from ._counted import counted_solve


class GMRES:
    def __init__(self, tol=1e-10, atol=1e-16, max_iter=1000, restart=20):
        if not math.isfinite(tol) or not math.isfinite(atol) or min(tol, atol) < 0:
            raise ValueError('tolerances must be finite and nonnegative')
        if max_iter is not None and (not isinstance(max_iter, int) or max_iter < 0):
            raise ValueError('max_iter must be a nonnegative integer or None')
        if not isinstance(restart, int) or restart < 1:
            raise ValueError('restart must be a positive integer')
        self.tol, self.atol, self.max_iter, self.restart = tol, atol, max_iter, restart

    def solve(self, limap, b, x0=None):
        return solve_jit(limap, b, x0, self.tol, self.atol, self.max_iter, self.restart)

    def solve_with_stats(self, limap, b, x0=None):
        return counted_solve(limap, b, x0, tol=self.tol, atol=self.atol,
                             maxiter=self.max_iter, method='gmres', restart=self.restart)


def solve_jit(limap, b, x0, tol, atol, max_iter, restart=20):
    return gmres(limap, b, x0=x0, tol=tol, atol=atol, maxiter=max_iter, restart=restart)
