import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jax_cgd import ACGD, BCGD, solvers, utils

C = jnp.array([[1.4, -0.7], [0.8, 1.3]], dtype=jnp.float32)


def loss(x, y):
    return x["w"] @ C @ y["w"] + 0.3 * (x["w"] @ x["w"]) - 0.2 * (y["w"] @ y["w"])


def make_params():
    return {"w": jnp.array([0.7, -0.4])}, {"w": jnp.array([0.2, 0.6])}


def direct_step(x, y, ex, ey):
    def f(a, b):
        return loss({"w": a}, {"w": b})
    gx, gy = jax.grad(f, (0, 1))(x, y)
    hxy = jax.jacfwd(jax.grad(f, 0), 1)(x, y)
    hyx = jax.jacfwd(jax.grad(f, 1), 0)(x, y)
    K = jnp.block([
        [jnp.eye(x.size), ex * hxy],
        [-ey * hyx, jnp.eye(y.size)],
    ])
    delta = jnp.linalg.solve(K, jnp.concatenate([-ex * gx, ey * gy]))
    return x + delta[:x.size], y + delta[x.size:]


@pytest.mark.parametrize("solver", [
    solvers.CG(tol=1e-6),
    solvers.GMRES(tol=1e-6, max_iter=20),
])
def test_bcgd_against_block_direct(solver):
    xp, yp = make_params()
    oldx, oldy = xp["w"], yp["w"]
    opt = BCGD(xp, yp, loss, lr=0.15, solver=solver)
    opt.step()
    expectedx, expectedy = direct_step(oldx, oldy, 0.15, 0.15)
    np.testing.assert_allclose(opt.x_params, expectedx, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(opt.y_params, expectedy, rtol=2e-5, atol=2e-6)
    assert int(opt.timestep) == 1
    assert all(np.isfinite(np.asarray(a)).all() for a in [opt.x_params, opt.y_params])


def test_acgd_bias_and_3_steps():
    xp, yp = make_params()
    opt = ACGD(xp, yp, loss, lr=0.04, beta=0.7, eps=1e-3,
               solver=solvers.CG(tol=1e-6))
    expected_x, expected_y = xp["w"], yp["w"]
    vx = jnp.zeros_like(expected_x)
    vy = jnp.zeros_like(expected_y)
    for t in range(1, 4):
        def f(a, b):
            return loss({"w": a}, {"w": b})
        gx, gy = jax.grad(f, (0, 1))(expected_x, expected_y)
        vx = 0.7 * vx + 0.3 * gx ** 2
        vy = 0.7 * vy + 0.3 * gy ** 2
        bias = 1 - 0.7 ** t
        ex = 0.04 * jnp.sqrt(bias) / (jnp.linalg.norm(vx) + 1e-3)
        ey = 0.04 * jnp.sqrt(bias) / (jnp.linalg.norm(vy) + 1e-3)
        expected_x, expected_y = direct_step(expected_x, expected_y, ex, ey)
        opt.step()
        np.testing.assert_allclose(opt.x_params, expected_x, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(opt.y_params, expected_y, rtol=3e-5, atol=3e-6)
        assert int(opt.timestep) == t


def test_operator_spd_and_transpose():
    xp, yp = make_params()
    x, y = xp["w"], yp["w"]
    f = lambda a, b: loss({"w": a}, {"w": b})
    op = utils.Operater(f, y, x, 0.3, 0.2)
    matrix = jax.vmap(op.as_function(), in_axes=1, out_axes=1)(jnp.eye(2))
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-6, atol=1e-6)
    assert np.min(np.linalg.eigvalsh(np.asarray(matrix))) >= 0.999
    hxy, hyx = utils.cross_hessian_matvecs(f, x, y)
    v = jnp.array([0.3, -0.4])
    w = jnp.array([-0.1, 0.4])
    np.testing.assert_allclose(
        jnp.dot(hxy(v), w), jnp.dot(v, hyx(w)), rtol=1e-6
    )


def test_default_solver_and_warm_start():
    x, y = make_params()
    opt = BCGD(x, y, loss)
    assert isinstance(opt.solver, solvers.CG)
    assert opt.prev_sol is not None
    for _ in range(3):
        opt.step()
    assert int(opt.timestep) == 3
    assert opt.prev_sol.shape == (2,)


def test_metadata_jit():
    x = {"w": jnp.array([1., 2.]), "b": jnp.ones((2, 2))}
    flat, m = utils.params_to_array(x)
    restored = jax.jit(utils.params_to_dict, static_argnums=1)(flat, m)
    for key in x:
        np.testing.assert_allclose(restored[key], x[key])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_krylov_zero_rhs(dtype):
    if dtype == jnp.float64 and not jax.config.jax_enable_x64:
        pytest.skip("x64 disabled")
    b = jnp.zeros((3,), dtype=dtype)
    for solver in (solvers.CG(), solvers.GMRES(max_iter=5)):
        sol, _ = solver.solve(lambda v: 2 * v, b, b)
        np.testing.assert_allclose(sol, b, atol=0.)


def test_vectorized_and_scanned_functional_steps():
    from jax_cgd.cgd import step_func
    xp, yp = make_params()
    opt = BCGD(xp, yp, loss, lr=0.02, solver=solvers.CG(tol=1e-6))
    single = lambda a, b, init: step_func(
        a, b, opt.x_metadata, opt.y_metadata, jnp.int32(0),
        opt.eta, opt.solver, init, opt.f
    )
    batch_x = jnp.stack([opt.x_params, opt.x_params + 0.05])
    batch_y = jnp.stack([opt.y_params, opt.y_params - 0.05])
    bx, by, *_ = jax.jit(jax.vmap(single))(
        batch_x, batch_y, jnp.zeros_like(batch_x)
    )
    assert bx.shape == (2, 2)
    assert by.shape == (2, 2)

    def body(carry, unused):
        a, b, prev = carry
        a, b, prev, _, _ = single(a, b, prev)
        return (a, b, prev), None

    final_x, final_y, prev = jax.jit(
        lambda a, b: jax.lax.scan(
            body, (a, b, jnp.zeros_like(a)), None, length=3
        )[0]
    )(opt.x_params, opt.y_params)
    assert jnp.all(jnp.isfinite(final_x))
    assert jnp.all(jnp.isfinite(final_y))
