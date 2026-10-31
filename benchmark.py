"""Benchmark nested-HVP and cached-linearization CGD; block for completion."""
import json
import statistics
import time

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from jax_cgd import utils


def run_case(n=128, nonlinear=True):
    key = jax.random.PRNGKey(n + 104)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (n,)) * 0.15
    y = jax.random.normal(k2, (n,)) * 0.15
    W = jax.random.normal(k3, (n, n)) / jnp.sqrt(float(n))

    def f(a, b):
        if nonlinear:
            a, b = jnp.tanh(a), jnp.sin(b)
        return a @ W @ b + 0.03 * jnp.sum(a*a) - 0.05 * jnp.sum(b*b)

    eta = 0.2

    def baseline(a, b):
        gx, gy = jax.grad(f, argnums=(0, 1))(a, b)
        def mv(v):
            tmp = utils.H_yx_vx(f, a, b, eta**0.5 * v)
            return v + eta**0.5 * utils.H_xy_vy(f, a, b, eta * tmp)
        rhs = eta**0.5 * (gx + utils.H_xy_vy(f, a, b, eta * gy))
        return cg(mv, rhs, tol=1e-6, atol=0, maxiter=50)[0]

    def cached(a, b):
        gx, gy = jax.grad(f, argnums=(0, 1))(a, b)
        op = utils.Operater(f, b, a, eta, eta)
        rhs = eta**0.5 * (gx + op.hxy(eta * gy))
        return cg(op.as_function(), rhs, tol=1e-6, atol=0, maxiter=50)[0]

    data = {"dimension": n, "nonlinear": nonlinear}
    answers = {}
    for name, func in [("baseline", baseline), ("cached", cached)]:
        jit_func = jax.jit(func)
        t0 = time.perf_counter()
        answers[name] = jit_func(x, y)
        answers[name].block_until_ready()
        data[name + "_compile_s"] = time.perf_counter() - t0
        samples = []
        for _ in range(30):
            t0 = time.perf_counter()
            jit_func(x, y).block_until_ready()
            samples.append((time.perf_counter() - t0) * 1000)
        data[name + "_ms"] = statistics.median(samples)
    data["max_abs_difference"] = float(jnp.max(jnp.abs(
        answers["baseline"] - answers["cached"]
    )))
    return data


if __name__ == "__main__":
    print(json.dumps([
        run_case(n, nonlin)
        for n in (16, 64, 128) for nonlin in (False, True)
    ], indent=2))
