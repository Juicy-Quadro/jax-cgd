"""Exact spatial derivatives for the reference scalar Flax tanh MLP.

The inherited forward method and Dense_i variable layout are unchanged.
"""
import jax.numpy as jnp
from flax import linen as nn
from examples.cpinn_flax import MLP


class JetMLP(MLP):
    @nn.compact
    def spatial_jet(self, coords, *, second_axes=(0, 1)):
        """Exact batched spatial gradient and selected Hessian trace.

        For the scalar tanh MLP only. Uses the same Dense_i parameters as
        __call__; all expressions remain differentiable with respect to them.
        No finite differences, stop-gradient, or PDE approximation is used.
        """
        if self.activation != "tanh" or self.out_features != 1:
            raise ValueError("spatial_jet requires a scalar tanh MLP")
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("spatial_jet requires coordinates of shape (N, 2)")
        if not second_axes or any(i not in (0, 1) for i in second_axes):
            raise ValueError("second_axes must be a nonempty subset of (0, 1)")
        if len(set(second_axes)) != len(second_axes):
            raise ValueError("second_axes cannot contain duplicates")
        z = coords
        scale = jnp.array([1., 2. if self.burgers_input else 1.], dtype=coords.dtype)
        if self.burgers_input:
            z = z * scale - jnp.array([0., 1.], dtype=coords.dtype)
        dz = jnp.broadcast_to(jnp.diag(scale), (coords.shape[0], 2, 2))
        lap = jnp.zeros_like(z)
        for i in range(self.hidden_layers + 1):
            dense = nn.Dense(self.width if i < self.hidden_layers else 1,
                             dtype=jnp.float64, param_dtype=jnp.float64,
                             kernel_init=nn.initializers.xavier_normal(), name=f"Dense_{i}")
            z = dense(z)
            kernel = dense.variables["params"]["kernel"]
            dz = dz @ kernel
            lap = lap @ kernel
            if i < self.hidden_layers:
                z = jnp.tanh(z)
                first = 1. - z*z
                second = -2. * z * first
                lap = first*lap + second*jnp.sum(dz[:, second_axes, :]**2, axis=1)
                dz = first[:, None, :] * dz
        return z[:, 0], dz[:, :, 0], lap[:, 0]
