"""Native released CG/GMRES controls, explicitly NOT historical-solver mode."""
from experiments.official_reference import *
import argparse, time, platform

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=200);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    q=fixture('poisson',5000,50,50,3);ev,truth=eval_arrays(q);points=jnp.asarray(ev)
    pred,disc=q['pred'],q['disc'];inside,bc,target=q['insidej'],q['bcj'],q['target']
    def game(x,y):return jnp.mean(disc(y,inside)[:,0]*poisson_residual(pred,x,inside))+jnp.mean(disc(y,bc)[:,1]*(pred(x,bc)[:,0]-target))
    @jax.jit
    def metrics(x):
        r=poisson_residual(pred,x,inside)
        return jnp.linalg.norm(pred(x,points)[:,0]-jnp.asarray(truth))/jnp.linalg.norm(jnp.asarray(truth)),jnp.mean(r*r),jnp.mean((pred(x,bc)[:,0]-target)**2)
    np.savez_compressed(a.output/'fixture.npz',inside=np.array(inside),boundary=np.array(bc),bc_target=np.array(target),eval_points=ev,eval_truth=truth,initial_P=tvec(q['P']),initial_D=tvec(q['D']))
    metadata=dict(python=sys.version,jax=jax.__version__,torch=torch.__version__,flax=__import__('flax').__version__,device=str(jax.devices()),platform=platform.platform(),n_pde=5000,n_bc=200,dtype='float64',lr=.001,beta=.99,eps=1e-8,tol=1e-7,atol=1e-20,requested_steps=a.steps,notes='Unmodified released solvers. Block cold1000 one cycle; Schur warm with exact jets. No claim both implement identical finite-tolerance stopping.')
    (a.output/'environment.json').write_text(json.dumps(metadata,indent=2))
    for variant in ('native_block_cold','native_schur_warm_jet'):
        form='block' if variant=='native_block_cold' else 'schur'
        solver=solvers.GMRES(tol=1e-7,atol=1e-20,restart=1000,max_iter=1) if form=='block' else solvers.CG(tol=1e-7,atol=1e-20,max_iter=1000)
        objective=game if form=='block' else BilinearGame(q['jfeat'],q['dfeat'])
        opt=ACGD(q['pv'],q['dv'],objective,lr=.001,beta=.99,eps=1e-8,solver=solver,formulation=form,on_failure='raise')
        out=a.output/variant;out.mkdir();rows=[];start=time.perf_counter();reason='requested_steps';step=0
        def record(t):
            er,pe,be=map(float,metrics(opt.x_params_dict));st=opt.get_stats()
            row=dict(step=t,l2=er,pde_mse=pe,bc_mse=be,elapsed_s=time.perf_counter()-start)
            if st is not None:row.update(krylov_total=int(st['krylov_iterations_total']),failed_total=int(st['failed_solves_total']),last_relative_residual=float(st['relative_residual']))
            rows.append(row)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(variant,json.dumps(row),flush=True)
            np.savez_compressed(out/'last_params.npz',P=jvec(opt.x_params_dict,q['ppairs']),D=jvec(opt.y_params_dict,q['dpairs']))
        record(0)
        try:
            for step in range(1,a.steps+1):
                if form=='block':opt.prev_sol=jnp.zeros_like(opt.prev_sol)
                opt.step()
                if step==1 or step%10==0 or step==a.steps:record(step)
        except Exception as error:
            import traceback
            reason=type(error).__name__; (out/'error.txt').write_text(traceback.format_exc());step=int(opt.timestep)
        if rows[-1]['step']!=step:record(step)
        opt.save_checkpoint(out/'optimizer.npz')
        (out/'summary.json').write_text(json.dumps(dict(variant=variant,completed_steps=step,stop_reason=reason,final=rows[-1]),indent=2))
        print('SUMMARY',variant,reason,step,flush=True)
        if reason!='requested_steps':raise SystemExit(2)

if __name__=='__main__':main()
