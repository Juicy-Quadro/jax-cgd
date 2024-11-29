import jax
import jax.numpy as jnp
from jax import grad, jit
from . import utils
from . import solvers
# import time
import jax.lax as lax
from functools import partial

class BCGD(object):
    def __init__(self, x_params_dict, y_params_dict, f_dict_input, lr=1e-3, solver=None):
        """_summary_

        Args:
            x_params_dict (_type_): _description_
            y_params_dict (_type_): _description_
            f_dict_input (_type_): _description_
            lr (_type_, optional): _description_. Defaults to 1e-3.
            beta (float, optional): _description_. Defaults to 0.9.
            eps (_type_, optional): _description_. Defaults to 1e-3.
            solver (_type_, optional): _description_. Defaults to None.
        """
        
        self.x_params_dict = x_params_dict
        self.y_params_dict = y_params_dict
        self.eta = lr
        
        if solver is None:
            self.solver = solvers.GMRES()
        else:
            self.solver = solver

        self.x_params, self.x_metadata = utils.params_to_array(x_params_dict)
        self.y_params, self.y_metadata = utils.params_to_array(y_params_dict)

        self.x_metadata = jax.device_get(self.x_metadata)
        self.y_metadata = jax.device_get(self.y_metadata)

        # Count number of parameters
        self.n_x = self.x_params.shape[0]
        self.n_y = self.y_params.shape[0]


        # Initialize timestep

        self.timestep = 0
        self.prev_sol = None

        self.f_dict_input = f_dict_input
        self.f = lambda x, y: f_dict_input(utils.params_to_dict(x, self.x_metadata), utils.params_to_dict(y, self.y_metadata))

    def get_infos(self):
        """
        Get the current parameters of the model.

        Returns:
            x_params_dict, y_params_dict
        """
        return self.x_params,self.y_params, self.x_params_dict, self.y_params_dict
         

    def step(self):
        """
        Perform a single optimization step by calling the external step_func.
        """
        (self.x_params,
         self.y_params,
         self.prev_sol,
         self.x_params_dict,
         self.y_params_dict) = step_func(
            self.x_params,
            self.y_params,
            self.x_metadata,
            self.y_metadata,
            self.timestep,
            self.eta,
            self.solver,
            self.prev_sol,
            self.f
        )

@partial(jax.jit, static_argnums=(2,3,5,6,8))
def step_func(x_params, y_params, x_metadata, y_metadata, timestep, eta, solver, prev_sol, f):
    """
    Perform a single optimization step for the ACGD optimizer.

    Args:
        x_params: Flattened array of x parameters.
        y_params: Flattened array of y parameters.
        x_metadata: Metadata for reconstructing x_params_dict.
        y_metadata: Metadata for reconstructing y_params_dict.
        vx: Second moment estimate for x parameters.
        vy: Second moment estimate for y parameters.
        timestep: Current timestep.
        eta: Learning rate.
        beta: Exponential decay rate for second moment estimates.
        eps: Small constant for numerical stability.
        solver: Linear algebra solver for solving linear systems.
        prev_sol: Previous solution for the solver.
        f: Objective function. Callable.

    Returns:
        Updated x_params, y_params, vx, vy, prev_sol, x_params_dict, y_params_dict.
    """
    timestep += 1

    # # Define the function f
    # f = lambda x, y: f_dict_input(
    #     params_to_dict(x, x_metadata), params_to_dict(y, y_metadata)
    # )

    # Compute gradients
    dfdx_val = lax.stop_gradient(jax.grad(f, argnums=0)(x_params, y_params))
    dfdy_val = lax.stop_gradient(jax.grad(f, argnums=1)(x_params, y_params))


    # Compute the Hessian-vector product
    A1 = utils.Operater(f, y_params, x_params, eta, eta)
    b1 = eta ** 0.5 * lax.stop_gradient(
        dfdx_val + utils.H_xy_vy(f, x_params, y_params, eta * dfdy_val)
    )
    prev_sol, _ = solver.solve(A1.as_function(), b1, prev_sol)
    dx = -eta ** 0.5 * prev_sol
    dy = eta * (dfdy_val + utils.H_yx_vx(f, x_params, y_params, dx))

    # Update parameters
    x_params += dx
    y_params += dy

    # Update parameter dictionaries
    x_params_dict = utils.params_to_dict(x_params, x_metadata)
    y_params_dict = utils.params_to_dict(y_params, y_metadata)

    return x_params, y_params, prev_sol, x_params_dict, y_params_dict

    
   
