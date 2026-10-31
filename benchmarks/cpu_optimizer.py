"""Equal-accuracy CPU ablation against TWO immutable source checkouts.

Run from repository root; baselines are extracted with git archive, not replaced
by a dense linear solver. This measures optimizer updates, not time-to-accuracy.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import os
import jax
import numpy as np
import flax
from jax_cgd import ACGD, solvers
from examples.cpinn_counted import make_problem


def load_source(path, name):
    package = Path(path) / 'jax_cgd'
    spec = importlib.util.spec_from_file_location(name, package/'__init__.py',
        submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy', required=True)
    p.add_argument('--counted', required=True)
    p.add_argument('--problem', choices=('poisson','burgers'), default='poisson')
    p.add_argument('--n-pde', type=int, default=256)
    p.add_argument('--n-bc', type=int, default=16)
    p.add_argument('--width', type=int, default=50)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--repeats', type=int, default=11)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--tol', type=float, default=1e-8)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.steps,a.repeats)<1:
        p.error('steps/repeats must be positive')
    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'cpu':
        p.error('CPU acceptance benchmark must run on CPU')
    legacy = load_source(a.legacy,'legacy_cgd')
    counted = load_source(a.counted,'counted_cgd')
    args=(a.problem,a.n_pde,a.n_bc,a.width,a.layers,a.seed)
    pv,dv,_,generic,factor,_ = make_problem(*args)
    _,_,_,_,jet,_ = make_problem(*args,spatial_jet=True)
    np.testing.assert_allclose(jet(pv,dv),generic(pv,dv),rtol=1e-11,atol=1e-12)
    old_factor = counted.games.BilinearGame(factor.primal,factor.dual)
    kw = dict(lr=1e-3,beta=.99,eps=1e-8)
    configs = [
        ('legacy',legacy.ACGD(pv,dv,generic,solver=legacy.solvers.CG(a.tol,0.,200),**kw),False),
        ('counted_generic',counted.ACGD(pv,dv,generic,solver=counted.solvers.CG(a.tol,0.,200),**kw),False),
        ('counted_factor',counted.ACGD(pv,dv,old_factor,solver=counted.solvers.CG(a.tol,0.,200),**kw),False),
        ('counted_chunk',counted.ACGD(pv,dv,old_factor,solver=counted.solvers.CG(a.tol,0.,200),**kw),True),
        ('candidate_generic',ACGD(pv,dv,generic,solver=solvers.CG(a.tol,0.,200),**kw),False),
        ('candidate_factor',ACGD(pv,dv,factor,solver=solvers.CG(a.tol,0.,200),**kw),False),
        ('candidate_jet',ACGD(pv,dv,jet,solver=solvers.CG(a.tol,0.,200),**kw),False),
        ('candidate_chunk',ACGD(pv,dv,jet,solver=solvers.CG(a.tol,0.,200),**kw),True),
    ]
    snapshots=[dict(opt.__dict__) for _,opt,_ in configs]
    times={name:[] for name,_,_ in configs}
    first={}; answers={}; stats={}; max_diff={}
    def execute(i):
        name,opt,chunk = configs[i]
        opt.__dict__.update(snapshots[i])
        begin = time.perf_counter()
        if chunk:
            opt.step_many(a.steps)
        else:
            for _ in range(a.steps):
                opt.step()
        jax.block_until_ready((opt.x_params,opt.y_params))
        seconds=time.perf_counter()-begin
        vec=np.concatenate((np.asarray(opt.x_params),np.asarray(opt.y_params)))
        if not np.all(np.isfinite(vec)):
            raise AssertionError(f'{name}: nonfinite parameters')
        if name=='legacy':
            answers['legacy']=vec
        else:
            np.testing.assert_allclose(vec,answers['legacy'],rtol=2e-7,atol=2e-9)
            max_diff[name]=max(max_diff.get(name,0.),float(np.max(np.abs(vec-answers['legacy']))))
            st=jax.device_get(opt.get_stats())
            stats[name]={key:value.item() for key,value in st.items()}
            if int(st['failed_solves_total']):
                raise AssertionError(f'{name}: inexact solves invalidate comparison')
        return seconds
    for i,(name,_,_) in enumerate(configs):
        first[name]=execute(i)
        execute(i)
    rng=np.random.default_rng(9123)
    for _ in range(a.repeats):
        for i in rng.permutation(len(configs)):
            times[configs[i][0]].append(execute(i))
    rows=[]
    for name,_,_ in configs:
        samples=np.asarray(times[name])*1000/a.steps
        rows.append(dict(variant=name,median_step_ms=float(np.median(samples)),
            samples_step_ms=samples.tolist(),first_chunk_s=first[name],
            max_parameter_difference=max_diff.get(name,0.),stats=stats.get(name)))
    med={r['variant']:r['median_step_ms'] for r in rows}
    pairs={'full_vs_counted_chunk':('counted_chunk','candidate_chunk'),
           'full_vs_legacy':('legacy','candidate_chunk'),
           'core_generic':('counted_generic','candidate_generic'),
           'core_factor':('counted_factor','candidate_factor'),
           'jet_only':('candidate_factor','candidate_jet')}
    comparisons={key:med[old]/med[new] for key,(old,new) in pairs.items()}
    try:
        cpu=subprocess.check_output(['lscpu'],text=True)
    except (OSError,subprocess.SubprocessError):
        cpu='unavailable'
    output=dict(python=platform.python_version(),jax=jax.__version__,flax=flax.__version__,
        cpu=cpu,devices=[str(d) for d in jax.devices()],affinity=sorted(os.sched_getaffinity(0)),
        dtype='float64',tol=a.tol,atol=0.,maxiter=200,problem=a.problem,
        n_pde=a.n_pde,n_bc=a.n_bc,width=a.width,layers=a.layers,seed=a.seed,
        steps=a.steps,repeats=a.repeats,rows=rows,speedups=comparisons,
        legacy_sha='b9320efa1d5ef66e88a2a87f215482671ccfcf82',
        counted_sha='78d2f7c275ad155cc206f3b0261763e08f4a43bb')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output),flush=True)


if __name__=='__main__':
    main()
