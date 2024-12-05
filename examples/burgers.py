import jax_cgd as jcgd
import jax
from jax import random, jit
import jax.numpy as jnp
from functools import partial
import numpy as np
from scipy.stats.qmc import Sobol

class SobolSequenceGenerator:
    def __init__(self, n_samples, dim, scramble=True, seed=0):
        self.n_samples = n_samples
        self.dim = dim
        self.sobol = Sobol(d=dim, scramble=scramble,seed=seed)
        self.sequence = None

    def generate_sequence(self):
        samples = self.sobol.random(n=self.n_samples)
        self.sequence = jnp.array(samples)

    def get_sequence(self):
        if self.sequence is None:
            self.generate_sequence()
        return self.sequence
      
jax.config.update("jax_enable_x64", True)

devices = jax.devices()
ID = 1
gpu_device = devices[ID]
# solve the Burgers equation with PINN
# settings
nu = 0.01/np.pi
x_min = 0
x_max = 1
t_min = 0
t_max = 1.2

key = jax.random.PRNGKey(0)
n_inner = 8192
n_initial = 256
n_boundary = 128

def generate_inner_points(key, n_inner, x_min, x_max, t_min, t_max):
    ssg = SobolSequenceGenerator(n_inner, 2)
    tx = ssg.get_sequence()
    x = tx[:, [0]] * (x_max - x_min) + x_min
    t = tx[:, [1]] * (t_max - t_min) + t_min
    return x, t

def generate_lboundary_points(key, n_boundary, x_min, x_max, t_min, t_max):
    t = jnp.linspace(t_min, t_max, n_boundary).reshape(-1, 1)
    x = jnp.ones((n_boundary, 1)) * x_min
    return x, t

def generate_rboundary_points(key, n_boundary, x_min, x_max, t_min, t_max):
    t = jnp.linspace(t_min, t_max, n_boundary).reshape(-1, 1)
    x = jnp.ones((n_boundary, 1)) * x_max
    return x, t

def generate_initial_points(key, n_initial, x_min, x_max, t_min, t_max):
    x = jnp.linspace(x_min, x_max, n_initial).reshape(-1, 1)
    t = jnp.zeros((n_initial, 1))
    return x, t
    

def ini_condition(x):
    return jnp.sin(2 * np.pi * x) / np.pi

x_inner, t_inner = generate_inner_points(key, n_inner, x_min, x_max, t_min, t_max)
x_lboundary, t_lboundary = generate_lboundary_points(key, n_boundary, x_min, x_max, t_min, t_max)
x_rboundary, t_rboundary = generate_rboundary_points(key, n_boundary, x_min, x_max, t_min, t_max)
x_initial, t_initial = generate_initial_points(key, n_initial, x_min, x_max, t_min, t_max)

x_inner = jax.device_put(x_inner, gpu_device)
t_inner = jax.device_put(t_inner, gpu_device)
x_lboundary = jax.device_put(x_lboundary, gpu_device)
t_lboundary = jax.device_put(t_lboundary, gpu_device)
x_rboundary = jax.device_put(x_rboundary, gpu_device)
t_rboundary = jax.device_put(t_rboundary, gpu_device)
x_initial = jax.device_put(x_initial, gpu_device)
t_initial = jax.device_put(t_initial, gpu_device)

def init_weights_dict(layer_sizes, key, method="xavier_uniform",adaptive_act=False):
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
    
generater_layersize = [2] + [64]*8 + [1]
params_g = init_weights_dict(generater_layersize, key)
discriminator_layersize = [2] + [64]*8 + [2]
params_d = init_weights_dict(discriminator_layersize, key)

params_g = jax.device_put(params_g, gpu_device)
params_d = jax.device_put(params_d, gpu_device)

@jax.jit
def discriminator_w(x, t, params_d):
    xt = jnp.concatenate([x, t], axis=1)
    return forward_pass(params_d, xt, activationf=jax.nn.relu)


# print(forward_pass_g(params_g, jnp.array([[1, 1],[2, 2]])))

# define the field function u
@jax.jit
def u(x, t, params_g):
    xt = jnp.concatenate([x, t], axis=1)
    u_ = forward_pass(params_g, xt)
    # alpha = jax.nn.tanh(t)
    return u_ #* alpha + ini_condition(x)

@jax.jit
def residual_lbd(xl, tl, params_g):
    u_l = u(xl, tl, params_g)
    return u_l

@jax.jit
def residual_rbd(xr, tr, params_g):
    u_r = u(xr, tr, params_g)
    return u_r

@jax.jit
def residual_ini(x, t, params_g):
    u_ini = u(x, t, params_g)
    return u_ini - ini_condition(x)

@jax.jit
def residual_inner(x, t, params_g):
    u_sum = lambda x, t: jnp.sum(u(x, t, params_g))
    u_x = jax.grad(u_sum, argnums=0)
    u_t = jax.grad(u_sum, argnums=1)
    u_x_sum = lambda x, t: jnp.sum(u_x(x, t))
    u_xx = jax.grad(u_x_sum, argnums=0)
    res_func = u_t(x, t).reshape(-1,1) + u(x, t, params_g) * u_x(x, t).reshape(-1,1) - nu * u_xx(x, t).reshape(-1,1)
    return res_func

# print(residual(x, t, params_g))
@jax.jit
def l2_loss(x_in, t_in, x_r,t_r, x_l, t_l,x_ini, t_ini, params_g):
    res_inner = residual_inner(x_in, t_in, params_g)
    res_rbd = residual_rbd(x_r, t_r, params_g)
    res_lbd = residual_lbd(x_l, t_l, params_g)
    res_ini = residual_ini(x_ini, t_ini, params_g)
    res_inner = jnp.abs(res_inner)
    res_rbd = jnp.abs(res_rbd)
    res_lbd = jnp.abs(res_lbd)
    res_ini = jnp.abs(res_ini)
    return jnp.mean(res_inner), jnp.mean(res_rbd), jnp.mean(res_lbd), jnp.mean(res_ini)

# print(l2_loss(x, t, params_g))
# print(jax.grad(l2_loss, argnums=2)(x, t, params_g))
@jax.jit
def discriminator_loss(x_in, t_in, x_r,t_r, x_l, t_l,x_ini, t_ini, params_g, params_d):
    # res_inner = residual_inner(x_in, t_in, params_g)
    # weight_inner = jax.nn.softplus(discriminator_w(x_in, t_in, params_d)[:,[0]])
    # res_rbd = residual_rbd(x_r, t_r, params_g)
    # weight_rbd = jax.nn.softplus(discriminator_w(x_r, t_r, params_d)[:,[1]])
    # res_lbd = residual_lbd(x_l, t_l, params_g)
    # weight_lbd = jax.nn.softplus(discriminator_w(x_l, t_l, params_d)[:,[2]])
    # res_ini = residual_ini(x_ini, t_ini, params_g) 
    # weight_ini = jax.nn.softplus(discriminator_w(x_ini, t_ini, params_d)[:,[3]])
    # res_inner = jnp.abs(res_inner)
    # res_rbd = jnp.abs(res_rbd)
    # res_lbd = jnp.abs(res_lbd)
    # res_ini = jnp.abs(res_ini)
    res_inner = residual_inner(x_in, t_in, params_g)
    weight_inner = discriminator_w(x_in, t_in, params_d)[:,[0]]
    res_rbd = residual_rbd(x_r, t_r, params_g)
    weight_rbd = discriminator_w(x_r, t_r, params_d)[:,[1]]
    res_lbd = residual_lbd(x_l, t_l, params_g)
    weight_lbd = discriminator_w(x_l, t_l, params_d)[:,[1]]
    res_ini = residual_ini(x_ini, t_ini, params_g)
    weight_ini = discriminator_w(x_ini, t_ini, params_d)[:,[1]]
    
    return jnp.mean(res_ini * weight_ini) + jnp.mean(res_inner * weight_inner) + jnp.mean(res_rbd * weight_rbd) + jnp.mean(res_lbd * weight_lbd)

# print(discriminator_loss(x, t, params_g, params_d))

f = jax.jit(lambda params_g_, params_d_: discriminator_loss(x_inner, t_inner, x_rboundary, t_rboundary, x_lboundary, t_lboundary, x_initial, t_initial, params_g_, params_d_))

optimizer = jcgd.ACGD(params_g, params_d, f, solver=jcgd.solvers.CG(tol=1e-9,atol=1e-9),lr=1e-2,eps=1e-8)

l2_loss_list = []
discriminator_loss_list = []

for i in range(20000):
    optimizer.step()
    x_params, y_params, x_params_dict, y_params_dict = optimizer.get_infos()
    l2_loss_inner_temp, l2_loss_rbd_temp, l2_loss_lbd_temp, l2_loss_ini_temp = l2_loss(x_inner, t_inner, x_rboundary, t_rboundary, x_lboundary, t_lboundary, x_initial, t_initial, x_params_dict)
    discriminator_loss_temp = f(x_params_dict, y_params_dict)
    l2_loss_list.append([l2_loss_inner_temp.item(), l2_loss_rbd_temp.item(), l2_loss_lbd_temp.item(), l2_loss_ini_temp.item()])
    discriminator_loss_list.append(discriminator_loss_temp.item())
    print(f"iter: {i}, mae_loss_ini: {l2_loss_ini_temp.item()}, mae_loss_rbd: {l2_loss_rbd_temp.item()}, mae_loss_lbd: {l2_loss_lbd_temp.item()}, mae_loss_inner: {l2_loss_inner_temp.item()}, discriminator_loss: {discriminator_loss_temp.item()}")
    
