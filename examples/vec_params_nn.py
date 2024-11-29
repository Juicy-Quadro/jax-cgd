import jax
import jax.numpy as jnp
import jax_cgd as jcgd
from jax import grad, random, vmap, jit, value_and_grad
from jax.scipy.sparse.linalg import gmres, cg  # Use conjugate gradient solver
from functools import partial
jax.config.update("jax_enable_x64", True)
jcgd.utils.params_arr

def init_weights_dict(layer_sizes, key, method="xavier_uniform"):
    keys = random.split(key, len(layer_sizes) - 1)
    params = {}
    for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if method == "xavier_uniform":
            limit = jnp.sqrt(6 / (m + n))  # Xavier uniform initialization
            params[f"W{i}"] = random.uniform(keys[i], (m, n), minval=-limit, maxval=limit)  # Weights
            params[f"b{i}"] = jnp.zeros(n)  # Bias
        elif method == "xavier_normal":
            stddev = jnp.sqrt(2 / (m + n))  # Xavier normal initialization
            params[f"W{i}"] = random.normal(keys[i], (m, n)) * stddev  # Weights
            params[f"b{i}"] = jnp.zeros(n)  # Bias
        elif method == "zeros":
            params[f"W{i}"] = jnp.zeros((m, n))  # Weights
            params[f"b{i}"] = jnp.zeros(n)  # Bias
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    return params

# Forward pass for the network using params dictionary
@partial(jit, static_argnums=(2,))
def forward_pass(params, inputs, activationf=jax.nn.tanh):
    activations = inputs
    num_layers = len(params) // 2
    for i in range(num_layers - 1):
        w = params[f"W{i}"]
        b = params[f"b{i}"]
        z = jnp.dot(activations, w) + b
        activations = activationf(z)  # tanh activation function
    # Linear output
    final_w = params[f"W{num_layers - 1}"]
    final_b = params[f"b{num_layers - 1}"]
    output = jnp.dot(activations, final_w) + final_b
    return output

# Forward pass for the network using flat params array
@partial(jit, static_argnums=(2, 3))
def forward_pass_new(params_arr, inputs, activationf, metadata):
    params = jcgd.utils.params_to_dict(params_arr, metadata)
    return forward_pass(params, inputs, activationf)


layer_sizes = [2, 4, 1]
key = random.PRNGKey(0)
params_dict = init_weights_dict(layer_sizes, key)
params_arr, metadata = jcgd.utils.params_to_array(params_dict)
print(forward_pass_new(params_arr, jnp.array([1.0, 2.0]), jax.nn.relu, metadata))

