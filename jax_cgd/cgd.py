"""Basic competitive gradient descent for real differentiable zero-sum games."""
from functools import partial
import jax
import jax.numpy as jnp
from . import utils
from ._optimizer import OptimizerBase, advance


class BCGD(OptimizerBase):
    def __init__(self,x_params_dict,y_params_dict,f_dict_input,lr=1e-3,solver=None,*,formulation="schur",on_failure="raise"):
        self._init(x_params_dict,y_params_dict,f_dict_input,lr,solver,adaptive=False,formulation=formulation,on_failure=on_failure)


@partial(jax.jit,static_argnums=(2,3,5,6,8))
def step_func(x_params,y_params,x_metadata,y_metadata,timestep,eta,solver,prev_sol,f):
    """Legacy functional interface; use the class for cumulative diagnostics."""
    (x,y,_,_,prev,_),_=advance(x_params,y_params,jnp.zeros_like(x_params),
        jnp.zeros_like(y_params),prev_sol,timestep,eta=eta,beta=.9,eps=1e-3,
        solver=solver,f=f,adaptive=False)
    return x,y,prev,utils.params_to_dict(x,x_metadata),utils.params_to_dict(y,y_metadata)
