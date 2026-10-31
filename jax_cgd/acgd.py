"""Adaptive competitive gradient descent with per-parameter RMS scaling."""
from functools import partial
import jax
from . import utils
from ._optimizer import OptimizerBase, advance


class ACGD(OptimizerBase):
    def __init__(self,x_params_dict,y_params_dict,f_dict_input,
                 lr=1e-3,beta=.9,eps=1e-3,solver=None,*,formulation="schur"):
        self._init(x_params_dict,y_params_dict,f_dict_input,lr,solver,
                   adaptive=True,beta=beta,eps=eps,formulation=formulation)


@partial(jax.jit,static_argnums=(2,3,7,8,9,10,12))
def step_func(x_params,y_params,x_metadata,y_metadata,vx,vy,timestep,
              eta,beta,eps,solver,prev_sol,f):
    """Legacy functional interface, retaining its seven return values."""
    (x,y,vx,vy,prev,_),_=advance(x_params,y_params,vx,vy,prev_sol,timestep,
        eta=eta,beta=beta,eps=eps,solver=solver,f=f,adaptive=True)
    return x,y,vx,vy,prev,utils.params_to_dict(x,x_metadata),utils.params_to_dict(y,y_metadata)
