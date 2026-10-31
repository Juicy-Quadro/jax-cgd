"""Flax/JAX CPINN benchmarks from Zeng et al., ICLR 2023 (arXiv:2204.11144).

Run from repository root:
    python examples/poisson.py --smoke
    python examples/burgers.py --smoke
    python examples/poisson.py --steps 34001 --mode acgd
    python examples/burgers.py --steps 28001 --mode acgd --reference-mat burgers_shock.mat

The default solver is CG for the SPD zero-sum Schur complement. Use
--solver gmres to more closely match the original paper's ACGD solver.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from scipy.io import loadmat
from scipy.stats import qmc

from jax_cgd import ACGD, BCGD, solvers


class MLP(nn.Module):
    """Flax primal/dual MLP; both players are differentiated through apply()."""
    hidden_layers: int
    width: int
    out_features: int = 1
    activation: str = "tanh"
    burgers_input: bool = False

    @nn.compact
    def __call__(self, coords):
        z = coords
        if self.burgers_input:
            z = jnp.stack((coords[..., 0], 2.0 * coords[..., 1] - 1.0), axis=-1)
        for _ in range(self.hidden_layers):
            z = nn.Dense(self.width, dtype=jnp.float64, param_dtype=jnp.float64,
                         kernel_init=nn.initializers.xavier_normal())(z)
            z = jnp.tanh(z) if self.activation == "tanh" else nn.relu(z)
        return nn.Dense(self.out_features, dtype=jnp.float64,
                        param_dtype=jnp.float64,
                        kernel_init=nn.initializers.xavier_normal())(z)


def exact_poisson(coords):
    return jnp.sin(coords[..., 0]) * jnp.cos(coords[..., 1])


def sample_poisson(n_pde=5000, n_bc=50, seed=123):
    """50 independently sampled points per edge (200 total) in paper setting."""
    rng = np.random.default_rng(seed)
    def side(axis, endpoint):
        p = rng.uniform(-2.0, 2.0, (n_bc, 2))
        p[:, axis] = endpoint
        return p
    boundary = np.concatenate(
        [side(0, -2), side(0, 2), side(1, -2), side(1, 2)], axis=0
    )
    interior = qmc.scale(qmc.LatinHypercube(2, seed=seed).random(n_pde),
                         [-2, -2], [2, 2])
    return (jnp.asarray(interior, dtype=jnp.float64),
            jnp.asarray(boundary, dtype=jnp.float64))


def sample_burgers(n_pde=10000, n_bc=100, seed=123):
    """Original PINNs/CPINNs: N_u=100 selected initial/boundary points.

    N_f=10000 LHS collocation points, plus all selected initial/boundary points.
    """
    rng = np.random.default_rng(seed)
    x0 = np.linspace(-1.0, 1.0, 256)
    t0 = np.zeros_like(x0)
    ts = np.linspace(0.0, 1.0, 100)
    candidates = np.concatenate([
        np.column_stack((x0, t0)),
        np.column_stack((-np.ones_like(ts), ts)),
        np.column_stack((np.ones_like(ts), ts)),
    ], axis=0)
    idx = rng.choice(len(candidates), size=n_bc, replace=False)
    boundary = candidates[idx]
    interior = qmc.scale(qmc.LatinHypercube(2, seed=seed).random(n_pde),
                         [-1, 0], [1, 1])
    interior = np.concatenate([interior, boundary], axis=0)
    return (jnp.asarray(interior, dtype=jnp.float64),
            jnp.asarray(boundary, dtype=jnp.float64))


def burgers_boundary_value(coords):
    return jnp.where(coords[..., 1] == 0.0,
                     -jnp.sin(jnp.pi * coords[..., 0]), 0.0)


def poisson_residual(model_apply, p, coords):
    """Directional second derivatives of the primal network output."""
    def point(z):
        return model_apply(p, z[None, :])[0, 0]
    grad_point = jax.grad(point)
    ex = jnp.array([1., 0.], dtype=coords.dtype)
    ey = jnp.array([0., 1.], dtype=coords.dtype)

    def one(z):
        _, dxx = jax.jvp(lambda a: grad_point(a)[0], (z,), (ex,))
        _, dyy = jax.jvp(lambda a: grad_point(a)[1], (z,), (ey,))
        return dxx + dyy + 2.0 * exact_poisson(z)
    return jax.vmap(one)(coords)


def burgers_residual(model_apply, p, coords):
    """ut + u ux - (0.01/pi) uxx, x in [-1,1], t in [0,1]."""
    def point(z):
        return model_apply(p, z[None, :])[0, 0]
    grad_point = jax.grad(point)
    ex = jnp.array([1., 0.], dtype=coords.dtype)

    def one(z):
        u = point(z)
        du = grad_point(z)
        _, dxx = jax.jvp(lambda a: grad_point(a)[0], (z,), (ex,))
        return du[1] + u * du[0] - (0.01 / jnp.pi) * dxx
    return jax.vmap(one)(coords)


def create_problem(problem, n_pde, n_bc, width, layers, seed):
    """Return Flax models, flat-dictionary initial variables, and fixed points."""
    if problem == "poisson":
        inside, boundary = sample_poisson(n_pde, n_bc, seed)
        primal = MLP(layers, width, activation="tanh")
        # Paper Poisson discriminator: 4 hidden ReLU layers of width 50.
        dual = MLP(4 if width == 50 and layers == 3 else layers,
                   width, out_features=2, activation="relu")
        exact_boundary = exact_poisson
        residual = poisson_residual
    elif problem == "burgers":
        inside, boundary = sample_burgers(n_pde, n_bc, seed)
        primal = MLP(layers, width, activation="tanh", burgers_input=True)
        dual = MLP(layers, width, out_features=2, activation="relu",
                   burgers_input=True)
        exact_boundary = burgers_boundary_value
        residual = burgers_residual
    else:
        raise ValueError(problem)
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    pv = flatten_dict(primal.init(k1, jnp.zeros((1, 2), jnp.float64)))
    dv = flatten_dict(dual.init(k2, jnp.zeros((1, 2), jnp.float64)))

    def pred(p, locations):
        return primal.apply(unflatten_dict(p), locations)

    def disc(p, locations):
        return dual.apply(unflatten_dict(p), locations)

    target_bc = exact_boundary(boundary)

    def game(p, d):
        res = residual(pred, p, inside)
        boundary_error = pred(p, boundary)[:, 0] - target_bc
        return (jnp.mean(disc(d, inside)[:, 0] * res)
                + jnp.mean(disc(d, boundary)[:, 1] * boundary_error))

    def pinn_loss(p):
        res = residual(pred, p, inside)
        boundary_error = pred(p, boundary)[:, 0] - target_bc
        return jnp.mean(res ** 2) + jnp.mean(boundary_error ** 2)

    return pv, dv, pred, game, pinn_loss, inside, boundary


def relative_error_poisson(pred, params, grid=64):
    axis = jnp.linspace(-2., 2., grid)
    x, y = jnp.meshgrid(axis, axis, indexing="ij")
    coords = jnp.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    values = pred(params, coords)[:, 0]
    exact = exact_poisson(coords)
    return jnp.linalg.norm(values - exact) / jnp.linalg.norm(exact)


def relative_error_burgers(pred, params, reference_mat):
    """Evaluate on the paper's original burgers_shock.mat reference grid."""
    data = loadmat(reference_mat)
    xs = np.asarray(data["x"]).reshape(-1)
    ts = np.asarray(data["t"]).reshape(-1)
    exact = np.real(np.asarray(data["usol"])).T  # [nt,nx]
    xx, tt = np.meshgrid(xs, ts, indexing="xy")
    points = jnp.asarray(np.stack([xx.ravel(), tt.ravel()], axis=-1),
                         dtype=jnp.float64)
    pred_vals = jnp.concatenate([
        pred(params, batch)[:, 0]
        for batch in jnp.array_split(points, (len(points) + 1023) // 1024)
    ])
    truth = jnp.asarray(exact.ravel(), dtype=jnp.float64)
    return jnp.linalg.norm(pred_vals - truth) / jnp.linalg.norm(truth)


def train(problem, *, mode="acgd", steps=1, n_pde=5000, n_bc=50,
          width=50, layers=3, lr=1e-3, beta=0.99, eps=1e-8,
          tol=1e-6, maxiter=1000, solver="cg", seed=123,
          report_every=100, reference_mat=None, grid=64):
    pv, dv, pred, game, pinn_loss, inside, boundary = create_problem(
        problem, n_pde, n_bc, width, layers, seed
    )
    start = time.perf_counter()
    if mode in ("acgd", "bcgd"):
        solver_obj = (solvers.CG(tol=tol, max_iter=maxiter)
                      if solver == "cg" else
                      solvers.GMRES(tol=tol, max_iter=maxiter))
        cls = ACGD if mode == "acgd" else BCGD
        kwargs = dict(lr=lr, solver=solver_obj)
        if mode == "acgd":
            kwargs.update(beta=beta, eps=eps)
        opt = cls(pv, dv, game, **kwargs)
    elif mode == "adam":
        import optax
        tx = optax.adam(lr, b1=0.99, b2=0.99, eps=eps)
        opt_state = tx.init(pv)

        @jax.jit
        def adam_step(p, state):
            grads = jax.grad(pinn_loss)(p)
            updates, state = tx.update(grads, state, p)
            return optax.apply_updates(p, updates), state
    else:
        raise ValueError(mode)

    def diagnostics(params, step):
        loss = float(pinn_loss(params).block_until_ready())
        data = dict(problem=problem, mode=mode, step=step,
                    pinn_loss=loss, elapsed_s=time.perf_counter() - start)
        if problem == "poisson":
            data["relative_l2"] = float(
                relative_error_poisson(pred, params, grid).block_until_ready()
            )
        elif reference_mat is not None:
            data["relative_l2"] = float(
                relative_error_burgers(pred, params, reference_mat).block_until_ready()
            )
        return data

    output = [diagnostics(pv, 0)]
    for step in range(1, steps + 1):
        if mode in ("acgd", "bcgd"):
            opt.step()
            params = opt.get_infos()[2]
        else:
            pv, opt_state = adam_step(pv, opt_state)
            params = pv
        if step % report_every == 0 or step == steps:
            output.append(diagnostics(params, step))
    return output


def main(problem):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("acgd", "bcgd", "adam"),
                        default="acgd")
    parser.add_argument("--solver", choices=("cg", "gmres"), default="cg")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-pde", type=int, default=None)
    parser.add_argument("--n-bc", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--reference-mat", type=Path, default=None)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny correctness run, NOT a paper reproduction")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    defaults = {
        "poisson": dict(steps=34001, n_pde=5000, n_bc=50, width=50, layers=3),
        "burgers": dict(steps=28001, n_pde=10000, n_bc=100, width=60, layers=8),
    }[problem]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.smoke:
        args.steps, args.n_pde, args.n_bc = 1, 6, 4
        args.width, args.layers = 4, 1
        args.report_every, args.maxiter, args.grid = 1, 10, 12
    rows = train(problem, mode=args.mode, steps=args.steps, n_pde=args.n_pde,
                 n_bc=args.n_bc, width=args.width, layers=args.layers,
                 lr=args.lr, beta=args.beta, tol=args.tol,
                 maxiter=args.maxiter, solver=args.solver, seed=args.seed,
                 report_every=args.report_every, reference_mat=args.reference_mat,
                 grid=args.grid)
    for row in rows:
        print(json.dumps(row), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit("Use python examples/poisson.py or examples/burgers.py")
