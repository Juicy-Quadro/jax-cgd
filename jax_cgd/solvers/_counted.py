"""Real-vector Krylov solves with device-side diagnostics (public JAX APIs)."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular


class SolverStats(NamedTuple):
    iterations: object       # actual Krylov steps, including a failed attempt
    matvecs: object          # includes warm-start and true-residual products
    restart_cycles: object  # GMRES cycles; zero for CG
    residual_norm: object   # true ||b - A x||, not the recurrence estimate
    rhs_norm: object
    converged: object
    breakdown: object


def _finish(k, mv, cycles, r, bnorm, target, failed):
    norm = jnp.linalg.norm(r)
    finite = jnp.isfinite(norm) & jnp.isfinite(bnorm)
    ok = finite & (norm <= target) & ~failed
    return SolverStats(k, mv, cycles, lax.stop_gradient(norm),
                       lax.stop_gradient(bnorm), ok, failed | ~finite)


def _initial(A, b, x0):
    if x0 is None:
        return jnp.zeros_like(b), b, jnp.int32(0)
    return x0, b - A(x0), jnp.int32(1)


def _cg(A, b, x0, tol, atol, maxiter):
    x, r, initial_mv = _initial(A, b, x0)
    bnorm = jnp.linalg.norm(b)
    target = jnp.maximum(tol * bnorm, atol)
    rr = jnp.vdot(r, r).real
    initial = (x, r, r, rr, jnp.int32(0), ~jnp.isfinite(rr))

    def cond(s):
        return (s[4] < maxiter) & (s[3] > target**2) & ~s[5]

    def body(s):
        x, r, p, rr, k, failed = s
        Ap = A(p)
        pAp = jnp.vdot(p, Ap).real
        bad = (pAp <= 0) | ~jnp.isfinite(pAp)
        alpha = jnp.where(bad, 0., rr / jnp.where(bad, 1., pAp))
        new_x = x + alpha * p
        new_r = r - alpha * Ap
        next_rr = jnp.vdot(new_r, new_r).real
        bad = bad | ~jnp.isfinite(next_rr)
        beta = next_rr / jnp.where(rr > 0, rr, 1.)
        return new_x, new_r, new_r + beta * p, next_rr, k + 1, failed | bad

    x, r, _, _, k, failed = lax.while_loop(cond, body, initial)
    # Check true residual; charge it. The initial residual is exact at k=0.
    r = lax.cond(k > 0, lambda: b - A(x), lambda: r)
    mv = initial_mv + k + (k > 0).astype(jnp.int32)
    return x, _finish(k, mv, jnp.int32(0), r, bnorm, target, failed)


def _gmres(A, b, x0, tol, atol, maxiter, restart):
    """Restarted incremental GMRES; two-pass classical Gram-Schmidt + Givens.

    maxiter caps restart cycles, matching the legacy JAX wrapper. Iterations
    counts actual Arnoldi steps, NOT restart * cycles. No preconditioner.
    """
    x, r, initial_mv = _initial(A, b, x0)
    bnorm = jnp.linalg.norm(b)
    target = jnp.maximum(tol * bnorm, atol)
    m = min(restart, b.size)
    initial = (x, r, jnp.int32(0), jnp.int32(0), initial_mv,
               ~jnp.isfinite(jnp.linalg.norm(r)))

    def cond(s):
        return (s[2] < maxiter) & (jnp.linalg.norm(s[1]) > target) & ~s[5]

    def cycle(s):
        x, r, cycles, total, mv, failed = s
        beta = jnp.linalg.norm(r)
        V = jnp.zeros((b.size, m + 1), b.dtype).at[:, 0].set(r / beta)
        R = jnp.eye(m, dtype=b.dtype)  # unused diagonals keep solve nonsingular
        cs, sn = jnp.zeros(m, b.dtype), jnp.zeros(m, b.dtype)
        g = jnp.zeros(m + 1, b.dtype).at[0].set(beta)
        state = (V, R, cs, sn, g, jnp.int32(0), jnp.bool_(False))

        def inner_cond(q):
            return (q[5] < m) & (jnp.abs(q[4][q[5]]) > target) & ~q[6]

        def inner(q):
            V, R, cs, sn, g, j, stop = q
            w = A(V[:, j])
            norm_before = jnp.linalg.norm(w)
            # Inactive V columns are zero. CGS2 reduces loss of orthogonality.
            h = V.T @ w
            w = w - V @ h
            correction = V.T @ w
            w = w - V @ correction
            h = h + correction
            hn = jnp.linalg.norm(w)
            happy = hn <= jnp.finfo(b.dtype).eps * norm_before
            h = h.at[j + 1].set(hn)
            V = V.at[:, j + 1].set(w / jnp.where(hn > 0, hn, 1.))

            def rotate(i, h):
                a, c = h[i], h[i + 1]
                return h.at[i].set(cs[i]*a + sn[i]*c).at[i+1].set(-sn[i]*a + cs[i]*c)
            h = lax.fori_loop(0, j, rotate, h)
            rho = jnp.hypot(h[j], h[j+1])
            invalid = (rho == 0) | ~jnp.isfinite(rho)
            denominator = jnp.where(invalid, 1., rho)
            c, d = h[j]/denominator, h[j+1]/denominator
            cs, sn = cs.at[j].set(c), sn.at[j].set(d)
            h = h.at[j].set(denominator).at[j+1].set(0.)
            R = R.at[:, j].set(h[:m])
            gj = g[j]
            g = g.at[j].set(c*gj).at[j+1].set(-d*gj)
            return V, R, cs, sn, g, j+1, happy | invalid

        V, R, _, _, g, k, stopped = lax.while_loop(inner_cond, inner, state)
        # Only the completed k-dimensional Krylov space enters the update.
        rhs_small = jnp.where(jnp.arange(m) < k, g[:m], 0.)
        coeff = solve_triangular(R, rhs_small, lower=False)
        new_x = x + V[:, :m] @ coeff
        new_r = b - A(new_x)
        norm = jnp.linalg.norm(new_r)
        bad = ~jnp.isfinite(norm) | ((stopped | (k == 0)) & (norm > target))
        return new_x, new_r, cycles+1, total+k, mv+k+1, failed | bad

    x, r, cycles, k, mv, failed = lax.while_loop(cond, cycle, initial)
    return x, _finish(k, mv, cycles, r, bnorm, target, failed)


def counted_solve(A, b, x0, *, tol, atol, maxiter, method, restart=20):
    b = jnp.asarray(b)
    if b.ndim != 1 or b.size == 0 or not jnp.issubdtype(b.dtype, jnp.floating):
        raise ValueError('solve_with_stats requires a nonempty real floating vector')
    if x0 is not None:
        x0 = jnp.asarray(x0, dtype=b.dtype)
        if x0.shape != b.shape:
            raise ValueError('x0 and b must have the same shape')
    mv = A if callable(A) else lambda v: A @ v
    limit = 10*b.size if maxiter is None else maxiter

    def solve(matvec, rhs, guess):
        if method == 'cg':
            return _cg(matvec, rhs, guess, tol, atol, limit)
        return _gmres(matvec, rhs, guess, tol, atol, limit, restart)

    return lax.custom_linear_solve(
        mv, b, solve=lambda a, r: solve(a, r, x0),
        transpose_solve=lambda a, r: solve(a, r, None),
        symmetric=(method == 'cg'), has_aux=True,
    )
