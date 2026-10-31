"""Bounded convergence attempt; does not modify the published 2.0.0rc1 optimizer.

A wall-budget stop is NOT a completed paper reproduction. Schur-CG nominal
work is NOT interchangeable with the paper's full-block GMRES pass budget.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat
import flax
import jax_cgd
from jax_cgd import ACGD, solvers
from jax_cgd.state import SolverFailure
from examples.cpinn_counted import make_problem
from examples.cpinn_flax import exact_poisson


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--problem', choices=['poisson', 'burgers'], required=True)
    p.add_argument('--mode', choices=['acgd', 'adam'], default='acgd')
    p.add_argument('--formulation', choices=['schur', 'block'], default='schur')
    p.add_argument('--eps', type=float, default=1e-8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--steps', type=int, default=48000)
    p.add_argument('--seconds', type=float, default=600)
    p.add_argument('--chunk', type=int, default=10)
    p.add_argument('--n-pde', type=int)
    p.add_argument('--n-bc', type=int)
    p.add_argument('--width', type=int)
    p.add_argument('--layers', type=int)
    p.add_argument('--reference-mat', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.chunk < 1 or a.steps < 1 or a.seconds <= 0:
        p.error('positive limits required')
    jax.config.update('jax_enable_x64', True)
    defaults = (5000, 50, 50, 3) if a.problem == 'poisson' else (10000, 100, 60, 8)
    for name, value in zip(['n_pde', 'n_bc', 'width', 'layers'], defaults):
        if getattr(a, name) is None:
            setattr(a, name, value)
    a.output.mkdir(parents=True, exist_ok=True)
    meta = dict(problem=a.problem, mode=a.mode, formulation=a.formulation, eps=a.eps,
                lr=a.lr, beta=.99, seed=123, dtype='float64', rtol=1e-7, atol=1e-20,
                n_pde=a.n_pde, n_bc_argument=a.n_bc, width=a.width, layers=a.layers,
                requested_steps=a.steps, wall_budget_seconds=a.seconds,
                python=platform.python_version(), jax=jax.__version__, flax=flax.__version__,
                package_version=jax_cgd.__version__, package_file=jax_cgd.__file__,
                devices=[str(d) for d in jax.devices()],
                source_sha='df67eb6c6309014b6c2752cfadd59a7483c181b2',
                protocol='current Flax example; not exact authors RNG/initialization',
                gmres_max_cycles=50, gmres_restart=20, cg_maxiter=1000)
    (a.output / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({'event': 'metadata', **meta}), flush=True)
    start = time.perf_counter()
    pv, dv, pred, generic, game, loss = make_problem(
        a.problem, a.n_pde, a.n_bc, a.width, a.layers, 123, spatial_jet=True)
    if a.problem == 'poisson':
        axis = jnp.linspace(-2., 2., 256)
        xx, yy = jnp.meshgrid(axis, axis, indexing='ij')
        coords = jnp.stack([xx.ravel(), yy.ravel()], -1)
        truth = exact_poisson(coords)
        n_inside = a.n_pde
        meta['evaluation_grid'] = [256, 256]
    else:
        if a.reference_mat is None:
            raise ValueError('Burgers needs official reference .mat')
        data = loadmat(a.reference_mat)
        xs, ts = data['x'].ravel(), data['t'].ravel()
        xx, tt = np.meshgrid(xs, ts, indexing='xy')
        coords = jnp.asarray(np.column_stack([xx.ravel(), tt.ravel()]))
        truth = jnp.asarray(np.real(data['usol']).T.ravel())
        n_inside = a.n_pde + 456
        meta['reference_sha256'] = hashlib.sha256(a.reference_mat.read_bytes()).hexdigest()
        meta['evaluation_grid'] = list(np.real(data['usol']).T.shape)
    @jax.jit
    def metrics(params):
        prediction = pred(params, coords)[:, 0]
        r = game.primal(params)
        return dict(relative_l2=jnp.linalg.norm(prediction-truth)/jnp.linalg.norm(truth),
                    pinn_loss=jnp.sum(r*r), pde_mse=jnp.sum(r[:n_inside]**2),
                    boundary_mse=jnp.sum(r[n_inside:]**2))
    opt = None
    if a.mode == 'acgd':
        linear = (solvers.CG(1e-7, 1e-20, 1000) if a.formulation == 'schur'
                  else solvers.GMRES(1e-7, 1e-20, 50, 20))
        opt = ACGD(pv, dv, game, lr=a.lr, solver=linear,
                   formulation=a.formulation, beta=.99, eps=a.eps)
        meta.update(primal_parameters=opt.n_x, dual_parameters=opt.n_y)
        params = opt.get_infos()[2]
    else:
        import optax
        tx = optax.adam(a.lr, b1=.99, b2=.99, eps=1e-8)
        state = tx.init(pv)
        params = pv
        def objective(p):
            r = game.primal(p)
            return jnp.vdot(r, r)
        @jax.jit
        def adam_chunk(params, state):
            def one(_, carry):
                pp, ss = carry
                updates, ss = tx.update(jax.grad(objective)(pp), ss, pp)
                return optax.apply_updates(pp, updates), ss
            return jax.lax.fori_loop(0, a.chunk, one, (params, state))
    (a.output / 'metadata.json').write_text(json.dumps(meta, indent=2))
    step = 0
    rows = []
    def record(event):
        m = {k: float(v) for k, v in jax.device_get(metrics(params)).items()}
        row = dict(event=event, step=step, wall_s=time.perf_counter()-start, **m)
        if opt is not None and opt.get_stats() is not None:
            row.update({k: np.asarray(v).item() for k, v in jax.device_get(opt.get_stats()).items()})
        rows.append(row)
        with (a.output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    record('initial')
    status = 'budget_exhausted'
    try:
        while step < a.steps and time.perf_counter()-start < a.seconds:
            n = min(a.chunk, a.steps-step)
            if opt is not None:
                try:
                    opt.step_many(n)
                except SolverFailure:
                    params = opt.get_infos()[2]
                    step = int(opt.timestep)
                    record('solver_failure')
                    status = 'solver_failure'
                    break
                params = opt.get_infos()[2]
                step = int(opt.timestep)
            else:
                if n != a.chunk:
                    break
                params, state = adam_chunk(params, state)
                step += a.chunk
            jax.block_until_ready(params)
            record('progress')
        if step >= a.steps:
            status = 'step_target_completed'
    except Exception:
        status = 'execution_error'
        (a.output / 'exception.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    if opt is not None:
        opt.save_checkpoint(a.output / 'final.npz')
    else:
        np.savez(a.output / 'adam_parameters.npz',
                 **{str(i):np.asarray(v) for i,v in enumerate(jax.tree.leaves(params))})
    final = dict(status=status, final_step=step, wall_s=time.perf_counter()-start,
                 final_relative_l2=rows[-1]['relative_l2'],
                 best_logged_relative_l2=min(r['relative_l2'] for r in rows),
                 complete_requested_training=step >= a.steps)
    (a.output / 'summary.json').write_text(json.dumps(final, indent=2)+'\n')
    print(json.dumps(final), flush=True)

if __name__ == '__main__':
    main()
