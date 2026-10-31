"""Adaptive competitive gradient descent for differentiable zero-sum games."""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from . import solvers, utils


class ACGD:
    def __init__(self, x_params_dict, y_params_dict, f_dict_input,
                 lr=1e-3, beta=0.9, eps=1e-3, solver=None):
        self.x_params_dict = x_params_dict
        self.y_params_dict = y_params_dict
        self.eta, self.beta, self.eps = lr, beta, eps
        self.solver = solvers.CG(tol=1e-6, atol=0.0) if solver is None else solver
        self.x_params, self.x_metadata = utils.params_to_array(x_params_dict)
        self.y_params, self.y_metadata = utils.params_to_array(y_params_dict)
        self.n_x, self.n_y = self.x_params.shape[0], self.y_params.shape[0]
        self.vx = jnp.zeros_like(self.x_params)
        self.vy = jnp.zeros_like(self.y_params)
        self.timestep = 0
        self.prev_sol = jnp.zeros_like(self.x_params)
        self.f_dict_input = f_dict_input
        x_meta, y_meta = self.x_metadata, self.y_metadata
        self.f = lambda x, y: f_dict_input(
            utils.params_to_dict(x, x_meta), utils.params_to_dict(y, y_meta)
        )

    def get_infos(self):
        return self.x_params, self.y_params, self.x_params_dict, self.y_params_dict

    def step(self):
        (self.x_params, self.y_params, self.vx, self.vy, self.prev_sol,
         self.x_params_dict, self.y_params_dict) = step_func(
            self.x_params, self.y_params, self.x_metadata, self.y_metadata,
            self.vx, self.vy, self.timestep, self.eta, self.beta, self.eps,
            self.solver, self.prev_sol, self.f
        )
        self.timestep += 1


@partial(jax.jit, static_argnums=(2, 3, 7, 8, 9, 10, 12))
def step_func(x_params, y_params, x_metadata, y_metadata, vx, vy, timestep,
              eta, beta, eps, solver, prev_sol, f):
    gx, gy = jax.grad(f, argnums=(0, 1))(x_params, y_params)
    gx, gy = lax.stop_gradient(gx), lax.stop_gradient(gy)
    vx = beta * vx + (1 - beta) * gx ** 2
    vy = beta * vy + (1 - beta) * gy ** 2
    # Keep the original scalar norm-based ACGD learning-rate convention.
    bias = 1 - beta ** (timestep + 1)
    eta_x = eta * jnp.sqrt(bias) / (jnp.linalg.norm(vx) + eps)
    eta_y = eta * jnp.sqrt(bias) / (jnp.linalg.norm(vy) + eps)
    operator = utils.Operater(f, y_params, x_params, eta_y, eta_x)
    scale = jnp.sqrt(eta_x)
    rhs = scale * (gx + operator.hxy(eta_y * gy))
    solution, _ = solver.solve(operator.as_function(), rhs, prev_sol)
    dx = -scale * solution
    dy = eta_y * (gy + operator.hyx(dx))
    new_x, new_y = x_params + dx, y_params + dy
    return (
        new_x, new_y, vx, vy, solution,
        utils.params_to_dict(new_x, x_metadata),
        utils.params_to_dict(new_y, y_metadata),
    )
