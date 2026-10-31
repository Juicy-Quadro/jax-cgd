"""Shared compiled zero-sum update and device-side work accounting."""
from functools import partial
import math
import jax
import jax.numpy as jnp
from jax import lax
from . import solvers, utils
from .games import BilinearGame
from .solvers._counted import SolverStats
from .state import OptimizerState, SolverFailure


def advance(x, y, vx, vy, previous, t, *, eta, beta, eps, solver, f, adaptive, formulation="schur"):
    # With x64 enabled, integer time counters must not promote float32 rates.
    eta, beta, eps = (jnp.asarray(v, dtype=x.dtype) for v in (eta, beta, eps))
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
        # Unknown counts do not mean unknown residual: verify custom solvers too.
        target = jnp.maximum(getattr(solver, 'tol', 1e-6)*bn,
                             getattr(solver, 'atol', 0.))
        bad = ~jnp.isfinite(norm) | ~jnp.isfinite(bn)
        stats = SolverStats(jnp.int32(-1), jnp.int32(-1), jnp.int32(-1),
                            norm, bn, (norm <= target) & ~bad, bad)
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
    proposal = (x+dx, y+dy, vx, vy, solution, t+1)
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v))
                              for v in (*proposal[:5], gx, gy)]))
    stats = stats._replace(converged=stats.converged & finite,
                           breakdown=stats.breakdown | ~finite)
    return proposal, stats


@partial(jax.jit, static_argnames=('solver', 'f', 'adaptive', 'steps',
                                  'formulation', 'on_failure'))
def run_steps(state, *, solver, f, eta, beta, eps, adaptive, steps,
              formulation='schur', on_failure='raise'):
    """Pure guarded transition. Never raises on device; inspect ``stopped``."""
    available = hasattr(solver, 'solve_with_stats')
    empty = SolverStats(jnp.int32(0), jnp.int32(0), jnp.int32(0),
                        jnp.asarray(0., state.x.dtype), jnp.asarray(0., state.x.dtype),
                        jnp.bool_(True), jnp.bool_(False))

    def cond(carry):
        current, _, stopped = carry
        return ((current.attempted_total-state.attempted_total < steps) & ~stopped)

    def body(carry):
        current, _, _ = carry
        proposal, st = advance(*current[:6], eta=eta, beta=beta, eps=eps,
                              solver=solver, f=f, adaptive=adaptive,
                              formulation=formulation)
        # Hard failure is never accepted, including under accept_inexact.
        hard = st.breakdown | ~jnp.isfinite(st.residual_norm) | ~jnp.isfinite(st.rhs_norm)
        st = st._replace(breakdown=hard, converged=jnp.asarray(st.converged) & ~hard)
        accept = ~hard & (st.converged | (on_failure == 'accept_inexact'))
        params = lax.cond(accept, lambda: proposal, lambda: tuple(current[:6]))
        tk = current.krylov_total+st.iterations if available else jnp.full_like(current.krylov_total, -1)
        tm = current.matvec_total+st.matvecs if available else jnp.full_like(current.matvec_total, -1)
        reused = current.reused_total
        if formulation == 'schur' and type(solver) is solvers.CG:
            reused = reused + (st.iterations > 0)
        new = OptimizerState(*params, tk, tm, current.failed_total+(~st.converged | hard),
                             reused, current.attempted_total+1,
                             current.rejected_total+~accept)
        return new, st, ~accept

    result, last, stopped = lax.while_loop(cond, body, (state, empty, jnp.bool_(False)))
    t, tk, tm, attempts = result.step, result.krylov_total, result.matvec_total, result.attempted_total
    nominal = 2*(tm+(attempts if formulation == 'schur' else 0))
    unknown = jnp.full_like(tk, -1)
    stats = dict(
        outer_steps=t, attempted_steps_total=attempts,
        rejected_steps_total=result.rejected_total,
        accepted_steps=t-state.step, attempted_steps=attempts-state.attempted_total,
        stopped=stopped, hard_failure=last.breakdown,
        krylov_iterations=last.iterations, krylov_iterations_total=tk,
        matvecs=last.matvecs, matvecs_total=tm,
        restart_cycles=last.restart_cycles, residual_norm=last.residual_norm,
        relative_residual=last.residual_norm/jnp.where(last.rhs_norm>0, last.rhs_norm, 1.),
        converged=last.converged, breakdown=last.breakdown,
        counts_available=jnp.bool_(available), failed_solves_total=result.failed_total,
        mixed_products_total=nominal-result.reused_total if available else unknown,
        mixed_products_reused_total=result.reused_total,
        mixed_products_nominal_total=nominal if available else unknown,
        paper_proxy_total=2*(tk+attempts) if available else unknown,
        factorized_game=jnp.bool_(isinstance(f, BilinearGame)),
        paper_formula_applicable=jnp.bool_(formulation=='block' and isinstance(solver, solvers.GMRES)),
    )
    return result, stats


class OptimizerBase:
    def _init(self, xp, yp, game, lr, solver, *, adaptive, beta=.9, eps=1e-3,
              formulation='schur', on_failure='raise'):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('lr must be positive and finite')
        if adaptive and (not 0 <= beta < 1 or not math.isfinite(eps) or eps <= 0):
            raise ValueError('ACGD requires 0 <= beta < 1 and positive finite eps')
        if on_failure not in ('raise', 'stop', 'accept_inexact'):
            raise ValueError('on_failure must be raise, stop or accept_inexact')
        self.eta, self.beta, self.eps, self.adaptive = float(lr), float(beta), float(eps), adaptive
        self.on_failure = on_failure
        if formulation not in ('schur', 'block'):
            raise ValueError('formulation must be schur or block')
        self.formulation = formulation
        if solver is None:
            solver = (solvers.CG(tol=1e-6, atol=0.) if formulation=='schur'
                      else solvers.GMRES(tol=1e-6, atol=0.))
        if formulation=='block' and isinstance(solver, solvers.CG):
            raise ValueError('The block system is nonsymmetric; use GMRES, not CG')
        for attr, default in (('tol', 1e-6), ('atol', 0.)):
            value = getattr(solver, attr, default)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Custom solver tolerances must be finite and nonnegative')
        self.solver = solver
        self.x_params_dict, self.y_params_dict, self.f_dict_input = xp, yp, game
        self.x_params, self.x_metadata = utils.params_to_array(xp)
        self.y_params, self.y_metadata = utils.params_to_array(yp)
        if self.x_params.dtype != self.y_params.dtype:
            raise TypeError('Both players must use the same real floating dtype')
        self.n_x, self.n_y = self.x_params.size, self.y_params.size
        xm, ym = self.x_metadata, self.y_metadata
        self.f = (game.flattened(xm, ym) if isinstance(game, BilinearGame) else
                  lambda x,y: game(utils.params_to_dict(x, xm), utils.params_to_dict(y, ym)))
        self.vx, self.vy = jnp.zeros_like(self.x_params), jnp.zeros_like(self.y_params)
        self.prev_sol = (jnp.zeros_like(self.x_params) if formulation=='schur'
                         else jnp.zeros_like(jnp.concatenate((self.x_params, self.y_params))))
        zero = jnp.asarray(0, dtype=jnp.int64 if jax.config.jax_enable_x64 else jnp.int32)
        self._totals = (zero,)*7
        self.timestep = zero
        self._stats = None

    @property
    def state(self):
        """Immutable JAX PyTree, usable with ``update`` / jit / vmap / scan."""
        t, tk, tm, failed, reused, attempts, rejected = self._totals
        return OptimizerState(self.x_params, self.y_params, self.vx, self.vy,
                              self.prev_sol, t, tk, tm, failed, reused, attempts, rejected)

    def update(self, state, *, steps=1):
        """Pure transition; no wrapper mutation or host-side exception/sync.

        Even with on_failure='raise', this returns ``stats['stopped']`` instead
        of raising inside JIT. Use state produced by this optimizer or a
        validated checkpoint. Objective/configuration must not be mutated
        after tracing. Full optimizer meta-gradients are not promised.
        """
        if not isinstance(state, OptimizerState):
            raise TypeError('state must be OptimizerState')
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        return run_steps(state, solver=self.solver, f=self.f, eta=self.eta,
                         beta=self.beta, eps=self.eps, adaptive=self.adaptive,
                         steps=steps, formulation=self.formulation, on_failure=self.on_failure)

    def _install_state(self, state, stats):
        # Reconstruct both parameter trees before changing any live attributes.
        xp = utils.params_to_dict(state.x, self.x_metadata)
        yp = utils.params_to_dict(state.y, self.y_metadata)
        self.x_params, self.y_params, self.vx, self.vy, self.prev_sol = state[:5]
        self._totals = tuple(state[5:])
        self.timestep = state.step
        self._stats = stats
        self.x_params_dict, self.y_params_dict = xp, yp

    def get_infos(self):
        return self.x_params, self.y_params, self.x_params_dict, self.y_params_dict

    def get_stats(self):
        """Device arrays; counts include rejected attempts. None before first step."""
        return self._stats

    def step(self):
        self.step_many(1)

    def step_many(self, steps):
        """Apply a chunk; reject unsafe updates and stop on its first rejection.

        Successful prefix remains committed. Default raise policy synchronizes
        one scalar at the chunk boundary, then raises SolverFailure if needed.
        ``stop`` and ``accept_inexact`` require the caller to inspect stats.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        if steps == 0:
            return
        state, stats = self.update(self.state, steps=steps)
        self._install_state(state, stats)
        if self.on_failure == 'raise' and bool(jax.device_get(stats['stopped'])):
            raise SolverFailure('Optimizer update rejected; successful prefix retained. '
                                'Inspect get_stats() before retrying.')

    def state_dict(self):
        """Versioned host copies of complete numerical state and configuration."""
        from ._checkpoint import state_dict
        return state_dict(self)

    def load_state_dict(self, snapshot):
        """Validate an entire snapshot before mutation; same config/tree required."""
        from ._checkpoint import load_state_dict
        load_state_dict(self, snapshot)

    def save_checkpoint(self, path):
        """Atomically write NPZ+JSON, without executable objects or pickle."""
        from ._checkpoint import save_checkpoint
        save_checkpoint(self, path)

    def load_checkpoint(self, path):
        """Load a compatible checkpoint; caller recreates matching objective/data."""
        from ._checkpoint import load_checkpoint
        load_checkpoint(self, path)
