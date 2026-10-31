"""Shared compiled zero-sum update and device-side work accounting."""
from functools import partial
import math
import jax
import jax.numpy as jnp
from jax import lax
from . import solvers, utils
from .games import BilinearGame
from .solvers._counted import SolverStats


def advance(x, y, vx, vy, previous, t, *, eta, beta, eps, solver, f, adaptive, formulation="schur"):
    gx, gy, hxy, hyx = utils.linearize_game(f, x, y)
    gx, gy = lax.stop_gradient(gx), lax.stop_gradient(gy)
    if adaptive:
        vx, vy = beta*vx + (1-beta)*gx**2, beta*vy + (1-beta)*gy**2
        bias = 1-beta**(t+1)
        ex = eta*jnp.sqrt(bias)/(jnp.sqrt(vx)+eps)
        ey = eta*jnp.sqrt(bias)/(jnp.sqrt(vy)+eps)
    else:
        ex = ey = eta
    scale = jnp.sqrt(ex)
    if formulation == 'schur':
        A = lambda v: v + scale*hxy(ey*hyx(scale*v))
        rhs = scale*(gx + hxy(ey*gy))
    else:
        def A(v):
            a,b = v[:x.size],v[x.size:]
            return jnp.concatenate((a+ex*hxy(b),b-ey*hyx(a)))
        rhs = jnp.concatenate((-ex*gx,ey*gy))
    reuse = formulation == 'schur' and type(solver) is solvers.CG
    if reuse:
        solution, stats = solver._solve_for_update(A, rhs, previous)
    elif hasattr(solver, 'solve_with_stats'):
        solution, stats = solver.solve_with_stats(A, rhs, previous)
    else:
        solution, _ = solver.solve(A, rhs, previous)
        norm, bn = jnp.linalg.norm(rhs-A(solution)), jnp.linalg.norm(rhs)
        # Custom solvers have no reliable counts or tolerance contract.
        stats = SolverStats(jnp.int32(-1), jnp.int32(-1), jnp.int32(-1),
                            norm, bn, jnp.bool_(False), ~jnp.isfinite(norm))
    if formulation == 'schur':
        dx = -scale*solution
        if reuse:
            # This product is needed both by the true residual and dy.
            dual_product = hyx(-dx)
            norm = lax.cond(stats.iterations > 0,
                lambda: jnp.linalg.norm(rhs - (solution + scale*hxy(ey*dual_product))),
                lambda: stats.residual_norm)
            target = jnp.maximum(solver.tol*stats.rhs_norm, solver.atol)
            bad = stats.breakdown | ~jnp.isfinite(norm)
            stats = stats._replace(
                matvecs=stats.matvecs+(stats.iterations>0).astype(jnp.int32),
                residual_norm=lax.stop_gradient(norm),
                converged=(norm<=target) & ~bad, breakdown=bad)
            dy = ey*(gy-dual_product)
        else:
            dy = ey*(gy+hyx(dx))
    else:
        dx,dy = solution[:x.size],solution[x.size:]
    return (x+dx, y+dy, vx, vy, solution, t+1), stats


@partial(jax.jit, static_argnames=('x_meta','y_meta','solver','f','adaptive','steps','formulation'))
def run_steps(state, *, x_meta, y_meta, solver, f, eta, beta, eps, adaptive, steps, formulation="schur"):
    *params, total_k, total_mv, failed, reused = state
    available = hasattr(solver, 'solve_with_stats')
    empty = SolverStats(jnp.int32(0), jnp.int32(0), jnp.int32(0),
                        jnp.asarray(0., params[0].dtype), jnp.asarray(0., params[0].dtype),
                        jnp.bool_(True), jnp.bool_(False))

    def body(_, carry):
        params, tk, tm, failed, reused, _ = carry
        params, st = advance(*params, eta=eta,beta=beta,eps=eps,solver=solver,
                             f=f,adaptive=adaptive,formulation=formulation)
        tk, tm = tk+st.iterations, tm+st.matvecs
        failed = failed + (~st.converged if available else st.breakdown)
        if formulation == 'schur' and type(solver) is solvers.CG:
            reused = reused + (st.iterations > 0)
        return params, tk, tm, failed, reused, st

    params, tk, tm, failed, reused, last = lax.fori_loop(
        0, steps, body, (tuple(params), total_k, total_mv, failed, reused, empty))
    t = params[5]
    stats = dict(
        outer_steps=t, krylov_iterations=last.iterations,
        krylov_iterations_total=tk if available else jnp.asarray(-1,tk.dtype),
        matvecs=last.matvecs, matvecs_total=tm if available else jnp.asarray(-1,tm.dtype),
        restart_cycles=last.restart_cycles, residual_norm=last.residual_norm,
        relative_residual=last.residual_norm/jnp.where(last.rhs_norm>0,last.rhs_norm,1.),
        converged=last.converged, breakdown=last.breakdown,
        counts_available=jnp.bool_(available),
        failed_solves_total=failed if available else jnp.asarray(-1,failed.dtype),
        # Schur form: one Hxy and one Hyx per A, plus RHS/reconstruction.
        mixed_products_total=2*(tm+(t if formulation=='schur' else 0))-reused if available else jnp.asarray(-1,tm.dtype),
        mixed_products_reused_total=reused,
        mixed_products_nominal_total=2*(tm+(t if formulation=='schur' else 0)) if available else jnp.asarray(-1,tm.dtype),
        # A proxy only: Schur/cached products are not paper full-block GMRES work.
        paper_proxy_total=2*(tk+t) if available else jnp.asarray(-1,tk.dtype),
        factorized_game=jnp.bool_(isinstance(f,BilinearGame)),
        paper_formula_applicable=jnp.bool_(formulation=='block' and isinstance(solver,solvers.GMRES)),
    )
    return ((*params, tk, tm, failed, reused), stats,
            utils.params_to_dict(params[0],x_meta), utils.params_to_dict(params[1],y_meta))


class OptimizerBase:
    def _init(self, xp, yp, game, lr, solver, *, adaptive, beta=0.9, eps=1e-3, formulation="schur"):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('lr must be positive and finite')
        if adaptive and (not 0 <= beta < 1 or not math.isfinite(eps) or eps <= 0):
            raise ValueError('ACGD requires 0 <= beta < 1 and positive finite eps')
        self.eta, self.beta, self.eps, self.adaptive = lr, beta, eps, adaptive
        if formulation not in ('schur','block'):
            raise ValueError('formulation must be schur or block')
        self.formulation = formulation
        if solver is None:
            solver = (solvers.CG(tol=1e-6,atol=0.) if formulation=='schur'
                      else solvers.GMRES(tol=1e-6,atol=0.))
        if formulation=='block' and isinstance(solver,solvers.CG):
            raise ValueError('The block system is nonsymmetric; use GMRES, not CG')
        self.solver = solver
        self.x_params_dict, self.y_params_dict, self.f_dict_input = xp, yp, game
        self.x_params, self.x_metadata = utils.params_to_array(xp)
        self.y_params, self.y_metadata = utils.params_to_array(yp)
        self.n_x, self.n_y = self.x_params.size, self.y_params.size
        xm,ym = self.x_metadata,self.y_metadata
        self.f = (game.flattened(xm,ym) if isinstance(game,BilinearGame) else
                  lambda x,y: game(utils.params_to_dict(x,xm),utils.params_to_dict(y,ym)))
        self.vx, self.vy = jnp.zeros_like(self.x_params),jnp.zeros_like(self.y_params)
        self.prev_sol = (jnp.zeros_like(self.x_params) if formulation=='schur'
                         else jnp.zeros_like(jnp.concatenate((self.x_params,self.y_params))))
        self.timestep = 0
        # 64-bit totals where supported, avoiding overflow in long paper runs.
        zero = jnp.asarray(0, dtype=jnp.int64 if jax.config.jax_enable_x64 else jnp.int32)
        self._totals = (zero,zero,zero,zero,zero)
        self._stats = None

    def get_infos(self):
        return self.x_params,self.y_params,self.x_params_dict,self.y_params_dict

    def get_stats(self):
        """Device arrays; transfer only when logging. None before the first step.

        paper_proxy_total is NOT a measured forward/backward-pass counter.
        Failed/inexact solves are reported but updates are not silently skipped.
        """
        return self._stats

    def step(self):
        self.step_many(1)

    def step_many(self, steps):
        """Compile a fixed chunk of updates without per-step Python dispatch.

        Does not retain a trajectory. Log/validate at chunk boundaries.
        Different chunk lengths may require separate compilation.
        """
        if not isinstance(steps,int) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        if steps == 0:
            return
        t,tk,tm,failed,reused = self._totals
        state=(self.x_params,self.y_params,self.vx,self.vy,self.prev_sol,t,tk,tm,failed,reused)
        state,self._stats,self.x_params_dict,self.y_params_dict=run_steps(
            state,x_meta=self.x_metadata,y_meta=self.y_metadata,solver=self.solver,
            f=self.f,eta=self.eta,beta=self.beta,eps=self.eps,
            adaptive=self.adaptive,steps=steps,formulation=self.formulation)
        self.x_params,self.y_params,self.vx,self.vy,self.prev_sol,t,tk,tm,failed,reused=state
        self._totals=(t,tk,tm,failed,reused)
        self.timestep += steps
