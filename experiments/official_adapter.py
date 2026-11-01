"""Experimental historical GMRES adapter; NOT the public safe solver.

CGDs0.4.5 cold start, one-pass MGS, Givens, projected strict-< stopping,
small-RHS return-b. Capacity is workspace only: callers must grow and retry
an unaccepted trial from the same state if it fills below official_maxiter.
No reverse-mode differentiation guarantee. No changes to package sources.
"""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import lax
from jax_cgd.solvers._counted import SolverStats

@dataclass(frozen=True)
class OfficialCodeGMRES:
    tol: float = 1e-7
    atol: float = 1e-20
    capacity: int = 128
    official_maxiter: int | None = None
    n_primal: int | None = None

    @staticmethod
    def reference_index(stats):
        return jnp.maximum(stats.iterations - 1, 0)

    def solve_with_stats(self, A, b, x0=None):
        if self.n_primal is None:
            return self._solve(A, b)
        n = self.n_primal
        forward = lambda v: jnp.concatenate((v[n:], v[:n]))
        inverse = lambda v: jnp.concatenate((v[b.size-n:], v[:b.size-n]))
        s, stats = self._solve(lambda v: forward(A(inverse(v))), -forward(b))
        return -inverse(s), stats

    def solve(self, A, b, x0=None):
        sol, stats = self.solve_with_stats(A, b, x0)
        return sol, self.reference_index(stats)

    def _solve(self, A, b):
        lim = b.size if self.official_maxiter is None else self.official_maxiter
        m = min(self.capacity, lim)
        bn = jnp.linalg.norm(b)
        short = (bn < 1e-8) | (lim == 0)
        def shortcut():
            return b, jnp.int32(0), jnp.asarray(0., b.dtype)
        def iterate():
            eps = jnp.finfo(b.dtype).eps
            v0 = jnp.where(bn > eps, b/jnp.where(bn > 0, bn, 1.), jnp.zeros_like(b))
            V = jnp.zeros((m+1, b.size), b.dtype).at[0].set(v0)
            R = jnp.zeros((m, m), b.dtype)
            cs, sn = jnp.zeros(m, b.dtype), jnp.zeros(m, b.dtype)
            g = jnp.zeros(m+1, b.dtype).at[0].set(bn)
            initial = (V, R, cs, sn, g, jnp.int32(0), jnp.bool_(False))
            def cond(s):
                return (s[5] < m) & ~s[6]
            def body(s):
                V, R, cs, sn, g, j, _ = s
                w, h = A(V[j]), jnp.zeros(m+1, b.dtype)
                def mgs(i, carry):
                    w, h = carry
                    a = jnp.dot(w, V[i])
                    return w-a*V[i], h.at[i].set(a)
                w, h = lax.fori_loop(0, j+1, mgs, (w, h))
                wn = jnp.linalg.norm(w)
                v = jnp.where(wn > eps, w/jnp.where(wn > 0, wn, 1.), jnp.zeros_like(w))
                V, h = V.at[j+1].set(v), h.at[j+1].set(wn)
                def rotate(i, h):
                    a, z = h[i], h[i+1]
                    return h.at[i].set(cs[i]*a-sn[i]*z).at[i+1].set(cs[i]*z+sn[i]*a)
                h = lax.fori_loop(0, j, rotate, h)
                rho = jnp.sqrt(h[j]*h[j]+h[j+1]*h[j+1])
                c, d = h[j]/rho, -h[j+1]/rho
                cs, sn = cs.at[j].set(c), sn.at[j].set(d)
                h = h.at[j].set(c*h[j]-d*h[j+1]).at[j+1].set(0.)
                R = R.at[:, j].set(h[:m])
                gj = g[j]
                g = g.at[j+1].set(d*gj).at[j].set(c*gj)
                residual = jnp.abs(g[j+1])
                done = (residual < self.tol*bn) | (residual < self.atol) | ~jnp.isfinite(residual)
                return V, R, cs, sn, g, j+1, done
            V, R, cs, sn, g, k, done = lax.while_loop(cond, body, initial)
            def back(s, y):
                i = k-1-s
                return y.at[i].set((g[i]-jnp.dot(R[i], y))/R[i, i])
            y = lax.fori_loop(0, k, back, jnp.zeros(m, b.dtype))
            return V[:m].T@y, k, jnp.abs(g[k])
        if m == 0:
            sol, k, proj = shortcut()
        else:
            sol, k, proj = lax.cond(short, shortcut, iterate)
        rn = jnp.linalg.norm(b-A(sol))
        finite = jnp.isfinite(rn) & jnp.isfinite(bn) & jnp.all(jnp.isfinite(sol))
        stats = SolverStats(k, k+1, jnp.asarray(k > 0, jnp.int32),
            lax.stop_gradient(rn), lax.stop_gradient(bn),
            finite & (rn <= jnp.maximum(self.tol*bn, self.atol)), ~finite)
        return sol, stats
