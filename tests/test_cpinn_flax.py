"""Physical correctness checks for the Flax paper benchmarks."""
import numpy as np
import jax
import jax.numpy as jnp
from examples.cpinn_flax import (
    MLP, burgers_boundary_value, burgers_residual, create_problem,
    exact_poisson, poisson_residual, sample_burgers, sample_poisson,
)


def test_poisson_pde_manufactured_solution():
    xy = jnp.array([[0.1, 0.3], [1.3, -0.8], [-1.4, 1.2]], dtype=jnp.float64)
    exact_model = lambda ignored, p: exact_poisson(p)[:, None]
    r = poisson_residual(exact_model, None, xy)
    np.testing.assert_allclose(r, 0.0, rtol=0, atol=2e-12)


def test_poisson_collocation_and_dirichlet():
    inside, boundary = sample_poisson(n_pde=11, n_bc=7, seed=23)
    assert inside.shape == (11, 2)
    assert boundary.shape == (28, 2)
    assert np.all(np.abs(np.asarray(inside)) <= 2)
    b = np.asarray(boundary)
    assert (np.abs(b[:, 0]) == 2).sum() >= 14
    assert (np.abs(b[:, 1]) == 2).sum() >= 14


def test_burgers_collocation_initial_boundary():
    inside, boundary = sample_burgers(n_pde=15, n_bc=9, seed=24)
    assert inside.shape == (24, 2)
    assert boundary.shape == (9, 2)
    assert np.all(np.abs(np.asarray(inside[:, 0])) <= 1)
    assert np.all(np.asarray(inside[:, 1]) >= 0)
    assert np.all(np.asarray(inside[:, 1]) <= 1)
    bc = burgers_boundary_value(boundary)
    x, t = np.asarray(boundary).T
    np.testing.assert_allclose(
        bc, np.where(t == 0, -np.sin(np.pi * x), 0), atol=1e-12
    )


def test_burgers_pde_zero_solution():
    coords = jnp.array([[0.2, 0.4], [-0.8, 0.7]], dtype=jnp.float64)
    zero_model = lambda ignored, p: jnp.zeros((p.shape[0], 1), dtype=p.dtype)
    np.testing.assert_allclose(
        burgers_residual(zero_model, None, coords), 0.0, atol=1e-12
    )


def test_flax_primal_and_dual_game_gradients():
    for problem in ("poisson", "burgers"):
        pv, dv, pred, game, pinn_loss, _, _ = create_problem(
            problem, n_pde=4, n_bc=4, width=4, layers=1, seed=8
        )
        assert pred(pv, jnp.zeros((3, 2), dtype=jnp.float64)).shape == (3, 1)
        val = game(pv, dv)
        gx, gy = jax.grad(game, (0, 1))(pv, dv)
        assert jnp.isfinite(val)
        assert all(jnp.all(jnp.isfinite(v)) for v in gx.values())
        assert all(jnp.all(jnp.isfinite(v)) for v in gy.values())
        assert jnp.isfinite(pinn_loss(pv))
