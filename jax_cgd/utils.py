"""Utilities for flattened parameters and matrix-free mixed Hessians."""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


def params_to_array(params_dict):
    if not params_dict:
        raise ValueError("Parameter dictionary must not be empty")
    params_list = []
    metadata = []
    for key, value in params_dict.items():
        params_list.append(jnp.ravel(value))
        metadata.append((key, value.shape))
    return jnp.concatenate(params_list), tuple(metadata)


def params_to_dict(params_arr, metadata):
    result = {}
    offset = 0
    for key, shape in metadata:
        size = int(np.prod(shape))
        result[key] = lax.dynamic_slice(params_arr, (offset,), (size,)).reshape(shape)
        offset += size
    return result


def H_xy_vy(h, x, y, vy):
    return jax.jvp(
        lambda yy: jax.grad(h, argnums=0)(x, yy), (y,), (vy,)
    )[1]


def H_yx_vx(h, x, y, vx):
    return jax.jvp(
        lambda xx: jax.grad(h, argnums=1)(xx, y), (x,), (vx,)
    )[1]


def cross_hessian_matvecs(f, x, y):
    """Cache the mixed-Hessian linearization once per optimizer step.

    Reuse its exact transpose, avoiding two separately traced nested
    autodiff expressions in every Krylov iteration. Cached primal
    intermediates can increase memory usage on large neural networks.
    """
    _, hxy = jax.linearize(
        lambda yy: jax.grad(f, argnums=0)(x, yy), y
    )
    hxy_transpose = jax.linear_transpose(hxy, y)

    def hyx(vx):
        return hxy_transpose(vx)[0]

    return hxy, hyx


class Operater:
    """SPD zero-sum CGD Schur complement; historical spelling retained."""

    def __init__(self, f, y_params_arr, x_params_arr, eta_y, eta_x):
        self.f = f
        self.y_params_arr = y_params_arr
        self.x_params_arr = x_params_arr
        self.eta_y = eta_y
        self.eta_x = eta_x
        self.shape = (len(x_params_arr), len(x_params_arr))
        self.hxy, self.hyx = cross_hessian_matvecs(f, x_params_arr, y_params_arr)

    def __matmul__(self, v):
        scale = jnp.sqrt(self.eta_x)
        return v + scale * self.hxy(self.eta_y * self.hyx(scale * v))

    def as_function(self):
        return lambda v: self @ v
