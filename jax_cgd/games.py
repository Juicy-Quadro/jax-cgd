"""Exact structure-aware linearizations for residual/discriminator games."""
import jax
import jax.numpy as jnp
from . import utils


class BilinearGame:
    """f(x,y) = <primal(x), dual(y)>; each feature map can be nonlinear.

    Put sampling weights into either feature map. This factors H_xy as
    J_primal.T @ J_dual WITHOUT forming either Jacobian or changing the game.
    Both feature maps must return same-sized real one-dimensional arrays.
    """
    def __init__(self, primal, dual):
        self.primal, self.dual = primal, dual

    def __call__(self, x, y):
        return jnp.vdot(self.primal(x), self.dual(y)).real

    def flattened(self, x_meta, y_meta):
        return BilinearGame(
            lambda x: self.primal(utils.params_to_dict(x, x_meta)),
            lambda y: self.dual(utils.params_to_dict(y, y_meta)),
        )

    def linearize(self, x, y):
        r, Jr = jax.linearize(self.primal, x)
        d, Jd = jax.linearize(self.dual, y)
        if r.ndim != 1 or r.shape != d.shape:
            raise ValueError('BilinearGame feature maps must return matching vectors')
        Tr = jax.linear_transpose(Jr, x)
        Td = jax.linear_transpose(Jd, y)
        return Tr(d)[0], Td(r)[0], lambda v: Tr(Jd(v))[0], lambda v: Td(Jr(v))[0]
