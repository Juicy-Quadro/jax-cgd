"""A/B benchmark against an immutable checkout, NOT a dense-solver surrogate.

python -m benchmarks.cpinn_optimizer --baseline /tmp/baseline --output results.json
Use git archive b9320efa1d5ef66e88a2a87f215482671ccfcf82 to prepare the baseline.
All variants use the same Flax parameters, samples, dtype and CG tolerance.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import flax
from jax_cgd import ACGD,solvers
from examples.cpinn_counted import make_problem


def load_baseline(path):
    path=Path(path)/'jax_cgd'
    spec=importlib.util.spec_from_file_location('baseline_cgd',path/'__init__.py',
                                               submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True)
    p.add_argument('--problem',choices=['poisson','burgers'],default='poisson')
    p.add_argument('--n-pde',type=int,default=128)
    p.add_argument('--n-bc',type=int,default=16)
    p.add_argument('--width',type=int,default=16)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--repeats',type=int,default=9)
    p.add_argument('--tol',type=float,default=1e-8)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();jax.config.update('jax_enable_x64',True)
    if min(a.steps,a.repeats)<1:p.error('positive steps/repeats required')
    base=load_baseline(a.baseline)
    pv,dv,pred,generic,factor,loss=make_problem(a.problem,a.n_pde,a.n_bc,a.width,a.layers,123)
    np.testing.assert_allclose(generic(pv,dv),factor(pv,dv),rtol=1e-12,atol=1e-12)
    args=dict(lr=1e-3,beta=.99,eps=1e-8)
    configs=[
        ('baseline',base.ACGD(pv,dv,generic,solver=base.solvers.CG(a.tol,0.,200),**args),False),
        ('counted_generic',ACGD(pv,dv,generic,solver=solvers.CG(a.tol,0.,200),**args),False),
        ('counted_factorized',ACGD(pv,dv,factor,solver=solvers.CG(a.tol,0.,200),**args),False),
        ('counted_factorized_chunk',ACGD(pv,dv,factor,solver=solvers.CG(a.tol,0.,200),**args),True),
    ]
    snapshots=[dict(opt.__dict__) for _,opt,_ in configs]
    times={name:[] for name,_,_ in configs};first={};answers={};last_stats={}

    def one(i):
        name,opt,chunk=configs[i]
        opt.__dict__.update(snapshots[i])  # restore identical initial state, outside timer
        tick=time.perf_counter()
        if chunk:opt.step_many(a.steps)
        else:
            for _ in range(a.steps):opt.step()
        jax.block_until_ready((opt.x_params,opt.y_params))
        seconds=time.perf_counter()-tick
        answers[name]=np.concatenate((np.asarray(opt.x_params),np.asarray(opt.y_params)))
        if name!='baseline':
            st=jax.device_get(opt.get_stats());last_stats[name]={k:v.item() for k,v in st.items()}
            if int(st['failed_solves_total'])!=0:
                raise AssertionError(f'{name}: inexact solve; cannot report same-accuracy speedup')
        return seconds

    for i,(name,_,_) in enumerate(configs):
        first[name]=one(i);one(i)  # compilation/first-run and extra unmeasured warm-up
    rng=np.random.default_rng(19)
    for _ in range(a.repeats):
        for i in rng.permutation(len(configs)):
            name=configs[i][0];times[name].append(one(i))
    rows=[]
    for name,opt,_ in configs:
        np.testing.assert_allclose(answers[name],answers['baseline'],rtol=2e-6,atol=2e-8)
        row=dict(variant=name,first_chunk_s=first[name],
                 median_step_ms=1000*statistics.median(times[name])/a.steps,
                 min_step_ms=1000*min(times[name])/a.steps,
                 max_step_ms=1000*max(times[name])/a.steps,
                 max_parameter_difference=float(np.max(np.abs(answers[name]-answers['baseline']))))
        row['speedup_vs_baseline']=statistics.median(times['baseline'])/statistics.median(times[name])
        row['stats']=last_stats.get(name)
        rows.append(row)
    output=dict(baseline_sha='b9320efa1d5ef66e88a2a87f215482671ccfcf82',
                python=platform.python_version(),jax=jax.__version__,flax=flax.__version__,
                devices=[str(d) for d in jax.devices()],dtype='float64',
                problem=a.problem,n_pde=a.n_pde,n_bc=a.n_bc,width=a.width,layers=a.layers,
                steps_per_sample=a.steps,repeats=a.repeats,tol=a.tol,atol=0.,rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)


if __name__=='__main__':main()
