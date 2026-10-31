"""Utilities for flattened model parameters and matrix-free mixed Hessians."""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


class ParameterLayout:
    """Hashable static layout for nested registered parameter PyTrees."""
    def __init__(self, treedef, shapes):
        self.treedef, self.shapes = treedef, tuple(shapes)

    def __hash__(self):
        return hash((self.treedef, self.shapes))

    def __eq__(self, other):
        return (isinstance(other, ParameterLayout) and
                self.treedef == other.treedef and self.shapes == other.shapes)

    def __repr__(self):
        return f'ParameterLayout({self.treedef!r}, {self.shapes!r})'


def params_to_array(params_dict):
    """Flatten real homogeneous float32/64 parameters, preserving flat-dict order.

    Construction is a host-side validation boundary (not a traced operation).
    Nested registered PyTrees are accepted; empty/nonfinite/integer/mixed-dtype
    parameters are rejected instead of silently promoted or truncated.
    """
    flat_dict = type(params_dict) is dict and all(
        hasattr(v, 'shape') for v in params_dict.values())
    if flat_dict:
        leaves = list(params_dict.values())
        metadata = tuple((k, tuple(v.shape)) for k, v in params_dict.items())
    else:
        leaves, treedef = jax.tree_util.tree_flatten(params_dict)
        metadata = ParameterLayout(treedef, [np.shape(v) for v in leaves])
    if not leaves or any(np.size(v) == 0 for v in leaves):
        raise ValueError('Parameter tree and all leaves must be nonempty')
    host = [np.asarray(v) for v in leaves]
    dtypes = {v.dtype for v in host}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (np.dtype('float32'), np.dtype('float64')):
        raise TypeError('Parameters require homogeneous real float32 or float64 leaves')
    if not all(np.isfinite(v).all() for v in host):
        raise ValueError('Initial parameters must be finite')
    arrays = [jnp.asarray(v) for v in leaves]
    if arrays[0].dtype != host[0].dtype:
        raise ValueError('Parameter dtype was truncated; enable JAX_ENABLE_X64 for float64')
    return jnp.concatenate([jnp.ravel(v) for v in arrays]), metadata


def params_to_dict(params_arr, metadata):
    """Restore the original parameter tree (historical function name retained)."""
    nested = isinstance(metadata, ParameterLayout)
    entries = enumerate(metadata.shapes) if nested else metadata
    result = [] if nested else {}
    offset = 0
    for key, shape in entries:
        size = int(np.prod(shape))
        value = lax.dynamic_slice(params_arr, (offset,), (size,)).reshape(shape)
        if nested:
            result.append(value)
        else:
            result[key] = value
        offset += size
    return metadata.treedef.unflatten(result) if nested else result


def H_xy_vy(h, x, y, vy):
    return jax.jvp(lambda yy: jax.grad(h, argnums=0)(x, yy), (y,), (vy,))[1]


def H_yx_vx(h, x, y, vx):
    return jax.jvp(lambda xx: jax.grad(h, argnums=1)(xx, y), (x,), (vx,))[1]


def cross_hessian_matvecs(f, x, y):
    """Linearize once and reuse H_xy*v and its exact transpose H_yx*v.

    Avoid retracing two nested autodiff expressions on every Krylov iteration.
    Like any `linearize`, this caches primal intermediates: for very large
    networks, compare memory consumption to the uncached HVP approach.
    """
    _, hxy = jax.linearize(lambda yy: jax.grad(f, argnums=0)(x, yy), y)
    hxy_transpose = jax.linear_transpose(hxy, y)

    def hyx(vx):
        return hxy_transpose(vx)[0]

    return hxy, hyx


class Operater:
    """Symmetric positive-definite zero-sum CGD Schur complement.

    Keep the historical spelling/API for backwards compatibility.
    """
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


def linearize_game(f, x, y):
    """Joint gradients + H_xy linearization in one primal evaluation.

    The y-gradient is auxiliary: do not build the unused H_yy tangent.
    Specialized separable games can supply their exact linearization.
    """
    if hasattr(f, 'linearize'):
        return f.linearize(x, y)

    def grad_x_aux(yy):
        return jax.grad(f, argnums=(0, 1))(x, yy)

    gx, hxy, gy = jax.linearize(grad_x_aux, y, has_aux=True)
    transpose = jax.linear_transpose(hxy, y)
    return gx, gy, hxy, lambda v: transpose(v)[0]
