import jax_cgd as jcgd
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from jax.tree_util import tree_flatten, tree_unflatten
from functools import partial
import numpy as np
from tqdm import tqdm


# define a simple neural network through flax
class MLP(nn.Module):
    layersizes: list
    act: nn.activation = nn.tanh

    def setup(self):
        self.layers = [nn.Dense(features=size, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros) for size in self.layersizes]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x

generater_layersize = [1, 64, 64, 1]
generator = MLP(generater_layersize, act=nn.swish)
discriminator_layersize = [1, 64, 64, 1]
discriminator = MLP(discriminator_layersize, act=nn.swish)
ex_input = jnp.ones((1, 1))
params_g = generator.init(jax.random.PRNGKey(0), ex_input)
params_d = discriminator.init(jax.random.PRNGKey(0), ex_input)

# dataset

x = (jnp.linspace(0 , 2, 100) * jnp.pi).reshape(-1, 1)
y = (jnp.sin(x)).reshape(-1, 1)
    
# define the forward pass for the neural network
@jax.jit
def forward_pass_g(params, inputs):
    return generator.apply(params, inputs)

@jax.jit
def forward_pass_d(params, inputs):
    return discriminator.apply(params, inputs)



@jax.jit
def dp_loss(params_g, params_d, x, y):
    p_out = forward_pass_g(params_g, x)
    d_weight = forward_pass_d(params_d, x)
    residual = y - p_out
    return jnp.mean(residual * d_weight)

f_dict_input = lambda p_g, p_d: dp_loss(p_g, p_d, x, y)

@jax.jit
def l2_loss(params_g, x, y):
    return jnp.mean((forward_pass_g(params_g, x) - y) ** 2)

acgd = jcgd.ACGD(params_g, params_d, f_dict_input,lr=0.01, solver=jcgd.solvers.CG())

print("Start training")
print("Initial l2_loss:", l2_loss(params_g, x, y))

closs = []
l2loss = []

for i in tqdm(range(8000), desc="Training", unit="step"):
    acgd.step()
    x_params, y_params, x_params_dict, y_params_dict = acgd.get_infos()
    closs_temp = f_dict_input(x_params_dict, y_params_dict).item()
    l2_loss_temp = l2_loss(x_params_dict, x, y).item()
    # tqdm.write(f"Current closs: {closs_temp}, Current l2_loss: {l2_loss_temp}")
    closs.append(closs_temp)
    l2loss.append(l2_loss_temp)
    
print("Final l2_loss:", l2_loss(x_params_dict, x, y))
    
import matplotlib.pyplot as plt
# plt.plot(closs, label="closs")
plt.plot(l2loss, label="l2loss")
plt.legend()
plt.yscale("log")
plt.show()

plt.plot(x, y, label="true")
plt.plot(x, forward_pass_g(x_params_dict, x), label="pred")
plt.plot(x, forward_pass_d(params_g, x), label="initial")
plt.legend()
plt.show()