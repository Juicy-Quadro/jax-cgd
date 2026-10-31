"""Flax CPINN training with counted work and compiled update chunks.

Run from repository root: python -m examples.cpinn_counted --problem poisson --smoke
Counts implement the paper's *nominal* work convention, not a runtime AD profiler.
The original examples/cpinn_flax.py is retained as the generic reference.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
from flax.traverse_util import unflatten_dict
from jax_cgd import ACGD, BCGD, solvers
from jax_cgd.games import BilinearGame
from examples.spatial_jets import JetMLP
from examples.cpinn_flax import (
    MLP, create_problem, poisson_residual, burgers_residual, exact_poisson,
    burgers_boundary_value, relative_error_poisson, relative_error_burgers,
)


def make_problem(problem, n_pde, n_bc, width, layers, seed, *, spatial_jet=False):
    """Same points/Flax variables as the reference, with an exact factored game."""
    if min(n_pde,n_bc,width,layers) < 1:
        raise ValueError('sample counts, width and layers must be positive')
    pv,dv,pred,generic,loss,inside,boundary=create_problem(
        problem,n_pde,n_bc,width,layers,seed)
    burgers=problem=='burgers'
    dual=MLP(4 if not burgers and width==50 and layers==3 else layers,
             width,out_features=2,activation='relu',burgers_input=burgers)
    residual=burgers_residual if burgers else poisson_residual
    target=(burgers_boundary_value if burgers else exact_poisson)(boundary)
    points=jnp.concatenate((inside,boundary))
    npde,nbc=inside.shape[0],boundary.shape[0]
    pi,bi=(1,0) if burgers else (0,1)

    primal = JetMLP(layers, width, activation="tanh", burgers_input=burgers)

    def primal_features(p):
        if spatial_jet:
            u, du, lap = primal.apply(unflatten_dict(p), inside,
                second_axes=(0,) if burgers else (0,1), method=primal.spatial_jet)
            res = (du[:,1] + u*du[:,0] - (0.01/jnp.pi)*lap if burgers
                   else lap + 2*exact_poisson(inside))
        else:
            res = residual(pred,p,inside)
        return jnp.concatenate((res/jnp.sqrt(float(npde)),
                                (pred(p,boundary)[:,0]-target)/jnp.sqrt(float(nbc))))

    def dual_features(d):
        # One batched discriminator call, rather than separate PDE/BC calls.
        values=dual.apply(unflatten_dict(d),points)
        return jnp.concatenate((values[:npde,pi]/jnp.sqrt(float(npde)),
                                values[npde:,bi]/jnp.sqrt(float(nbc))))

    return pv,dv,pred,generic,BilinearGame(primal_features,dual_features),loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--problem',choices=['poisson','burgers'],default='poisson')
    p.add_argument('--mode',choices=['acgd','bcgd','adam'],default='acgd')
    p.add_argument('--formulation',choices=['schur','block'],default='schur')
    p.add_argument('--solver',choices=['cg','gmres'],default=None)
    p.add_argument('--generic',action='store_true',help='Disable exact game factorization')
    p.add_argument('--autodiff',action='store_true',help='Use the reference nested-autodiff spatial derivatives instead of exact MLP jets')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--report-every',type=int,default=20)
    p.add_argument('--n-pde',type=int,default=None)
    p.add_argument('--n-bc',type=int,default=None)
    p.add_argument('--width',type=int,default=None)
    p.add_argument('--layers',type=int,default=None)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--tol',type=float,default=1e-7)
    p.add_argument('--atol',type=float,default=1e-20)
    p.add_argument('--maxiter',type=int,default=1000)
    p.add_argument('--restart',type=int,default=20)
    p.add_argument('--reference-mat',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    jax.config.update('jax_enable_x64',True)
    defaults=(5000,50,50,3) if a.problem=='poisson' else (10000,100,60,8)
    for name,default in zip(('n_pde','n_bc','width','layers'),defaults):
        if getattr(a,name) is None:setattr(a,name,default)
    if a.smoke:
        a.steps,a.report_every,a.n_pde,a.n_bc,a.width,a.layers=2,1,6,4,4,1
        a.maxiter=20
    if a.steps < 0 or a.report_every <= 0:
        p.error('steps must be nonnegative and report-every positive')
    pv,dv,pred,generic,game,loss=make_problem(a.problem,a.n_pde,a.n_bc,a.width,a.layers,a.seed,
        spatial_jet=not a.autodiff and not a.generic)
    game=generic if a.generic else game
    loss=jax.jit(loss)
    if a.mode=='adam':
        import optax
        tx=optax.adam(a.lr,b1=.99,b2=.99,eps=1e-8)
        opt_state=tx.init(pv)
        @partial(jax.jit,static_argnums=2)
        def adam_chunk(params,state,n):
            def body(_,carry):
                params,state=carry
                updates,state=tx.update(jax.grad(loss)(params),state,params)
                return optax.apply_updates(params,updates),state
            return jax.lax.fori_loop(0,n,body,(params,state))
    else:
        name=a.solver or ('gmres' if a.formulation=='block' else 'cg')
        linear=(solvers.CG(a.tol,a.atol,a.maxiter) if name=='cg' else
                solvers.GMRES(a.tol,a.atol,a.maxiter,a.restart))
        cls=ACGD if a.mode=='acgd' else BCGD
        kwargs=dict(beta=.99,eps=1e-8) if a.mode=='acgd' else {}
        opt=cls(pv,dv,game,lr=a.lr,solver=linear,formulation=a.formulation,**kwargs)
    error=jax.jit(lambda params:relative_error_poisson(pred,params,grid=64))
    begin=time.perf_counter();training_s=0.;rows=[]

    def record(step,params):
        row=dict(problem=a.problem,mode=a.mode,step=step,
                 pinn_loss=float(loss(params)),training_s_including_compile=training_s)
        if a.problem=='poisson':row['relative_l2']=float(error(params))
        elif a.reference_mat:
            row['relative_l2']=float(relative_error_burgers(pred,params,a.reference_mat))
        if a.mode=='adam':
            row.update(paper_proxy_total=step,outer_steps=step,krylov_iterations_total=0)
        elif step:
            row.update({k:v.item() for k,v in jax.device_get(opt.get_stats()).items()})
        else:row.update(paper_proxy_total=0,outer_steps=0,krylov_iterations_total=0)
        row['wall_s_including_diagnostics']=time.perf_counter()-begin
        row['spatial_derivatives']='autodiff' if a.autodiff or a.generic or a.mode=='adam' else 'exact_tanh_jet'
        row['work_axis']='paper_nominal_not_measured_AD_passes'
        row['formulation']=a.formulation if a.mode!='adam' else 'pinn'
        rows.append(row);print(json.dumps(row),flush=True)

    record(0,pv)
    for start in range(0,a.steps,a.report_every):
        n=min(a.report_every,a.steps-start)
        tick=time.perf_counter()
        if a.mode=='adam':
            pv,opt_state=adam_chunk(pv,opt_state,n)
            jax.block_until_ready((pv,opt_state));params=pv
        else:
            opt.step_many(n)
            jax.block_until_ready((opt.x_params,opt.y_params));params=opt.get_infos()[2]
        training_s+=time.perf_counter()-tick
        record(start+n,params)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
