"""Basic competitive gradient descent for differentiable zero-sum games."""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from . import solvers, utils


class BCGD:
    def __init__(self, x_params_dict, y_params_dict, f_dict_input, lr=1e-3, solver=None):
        self.x_params_dict = x_params_dict
        self.y_params_dict = y_params_dict
        self.eta = lr
        self.solver = solvers.CG(tol=1e-6, atol=0.0) if solver is None else solver
        self.x_params, self.x_metadata = utils.params_to_array(x_params_dict)
        self.y_params, self.y_metadata = utils.params_to_array(y_params_dict)
        self.n_x, self.n_y = self.x_params.shape[0], self.y_params.shape[0]
        self.timestep = 0
        # Keep the warm-start argument shape stable from the first call.
        self.prev_sol = jnp.zeros_like(self.x_params)
        self.f_dict_input = f_dict_input
        x_meta, y_meta = self.x_metadata, self.y_metadata
        self.f = lambda x, y: f_dict_input(
            utils.params_to_dict(x, x_meta), utils.params_to_dict(y, y_meta)
        )

    def get_infos(self):
        return self.x_params, self.y_params, self.x_params_dict, self.y_params_dict

    def step(self):
        (self.x_params, self.y_params, self.prev_sol,
         self.x_params_dict, self.y_params_dict) = step_func(
            self.x_params, self.y_params, self.x_metadata, self.y_metadata,
            self.timestep, self.eta, self.solver, self.prev_sol, self.f
        )
        self.timestep += 1


@partial(jax.jit, static_argnums=(2, 3, 5, 6, 8))
def step_func(x_params, y_params, x_metadata, y_metadata, timestep, eta, solver, prev_sol, f):
    gx, gy = jax.grad(f, argnums=(0, 1))(x_params, y_params)
    gx, gy = lax.stop_gradient(gx), lax.stop_gradient(gy)
    operator = utils.Operater(f, y_params, x_params, eta, eta)
    scale = jnp.sqrt(eta)
    rhs = scale * (gx + operator.hxy(eta * gy))
    solution, _ = solver.solve(operator.as_function(), rhs, prev_sol)
    dx = -scale * solution
    dy = eta * (gy + operator.hyx(dx))
    new_x, new_y = x_params + dx, y_params + dy
    return (
        new_x, new_y, solution,
        utils.params_to_dict(new_x, x_metadata),
        utils.params_to_dict(new_y, y_metadata),
    )
