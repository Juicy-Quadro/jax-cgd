"""Full-data official-code trial; resource limits are not convergence claims."""
from experiments.official_reference import *
from experiments.official_adapter import OfficialCodeGMRES
from dataclasses import replace
import argparse, time, tempfile, contextlib, platform, resource

@contextlib.contextmanager
def observe_reference(records):
    module=importlib.import_module('CGDs.gmres_acgd');gm=importlib.import_module('CGDs.gmres_torch')
    original=module.GMRES;real_torch=gm.torch
    class Proxy:
        buffers=[]
        def __getattr__(self,name):return getattr(real_torch,name)
        def zeros(self,*args,**kw):
            shape=args[0] if args and isinstance(args[0],(tuple,list)) else args
            if len(shape)==2 and shape[0]==shape[1] and shape[0]**2*8>64*1024**2:
                if str(kw.get('device','cpu'))!='cpu':raise RuntimeError('Sparse H mapping is CPU-only')
                f=tempfile.NamedTemporaryFile(prefix='cpinn-H-',suffix='.bin')
                m=np.memmap(f.name,dtype=np.float64,mode='w+',shape=tuple(shape));self.buffers.append((m,f))
                return torch.from_numpy(m)
            return real_torch.zeros(*args,**kw)
    proxy=Proxy()
    def observed(*a,**kw):
        calls=[0];A=kw['Avp'];bn=float(torch.linalg.norm(kw['b']))
        def mv(v):calls[0]+=1;return A(v)
        kw['Avp']=mv;gm.torch=proxy
        try:
            sol,info=original(*a,**kw)
            rn=float(torch.linalg.norm(kw['b']-A(sol)))
            records.append(dict(arnoldi=calls[0],historical_index=int(info[0]),rhs_norm=bn,
                true_relative_residual=rn/max(bn,1e-300),filebacked_H=bool(proxy.buffers),
                extra_diagnostic_matvecs=1))
            return sol,info
        finally:
            gm.torch=real_torch
            for m,f in proxy.buffers:f.close()
            proxy.buffers=[]
    module.GMRES=observed
    try:yield
    finally:module.GMRES=original;gm.torch=real_torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--problem',choices=['poisson','burgers'],required=True)
    parser.add_argument('--steps',type=int,default=None)
    parser.add_argument('--seconds',type=float,default=1200.)
    parser.add_argument('--paired-prefix',type=int,default=0,help='0: pair throughout; positive: prefix then JAX only')
    parser.add_argument('--report-every',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    n,bc,w,dep=(5000,50,50,3) if a.problem=='poisson' else (10000,100,60,8)
    target=34001 if a.problem=='poisson' else 28001
    requested=target if a.steps is None else a.steps
    t0=time.perf_counter();q=fixture(a.problem,n,bc,w,dep)
    P,D=q['P'],q['D'];nx,ny=len(tvec(P)),len(tvec(D));limit=1000 if a.problem=='poisson' else nx+ny
    pred,disc=q['pred'],q['disc'];inside,bd,targetbc=q['insidej'],q['bcj'],q['target']
    def game(p,d):
        r=q['resfun'](pred,p,inside)
        return jnp.mean(q['sign']*disc(d,inside)[:,q['pi']]*r)+jnp.mean(disc(d,bd)[:,q['bi']]*(pred(p,bd)[:,0]-targetbc))
    solver=OfficialCodeGMRES(capacity=128,official_maxiter=limit,n_primal=nx)
    opt=ACGD(q['pv'],q['dv'],game,lr=.001,beta=.99,eps=1e-8,solver=solver,formulation='block',on_failure='accept_inexact')
    ref=CGDs.GACGD(D.parameters(),P.parameters(),lr_x=.001,lr_y=.001,beta=.99,eps=1e-8,tol=1e-7,atol=1e-20,max_iter=1000 if a.problem=='poisson' else None,track_cond=lambda *_:True)
    eva,truth=eval_arrays(q);evj=jnp.asarray(eva);norm=np.linalg.norm(truth)
    @jax.jit
    def evaluate(p):
        r=q['resfun'](pred,p,inside)
        return jnp.linalg.norm(pred(p,evj)[:,0]-jnp.asarray(truth))/norm,jnp.mean(r*r),jnp.mean((pred(p,bd)[:,0]-targetbc)**2)
    frozen=dict(inside=np.array(inside),boundary=np.array(bd),bc_target=np.array(targetbc),eval_points=eva,eval_truth=truth,
        initial_P=tvec(P),initial_D=tvec(D),scale=np.array(q['scale']),shift=np.array(q['shift']))
    np.savez_compressed(out/'fixture.npz',**frozen)
    env=dict(python=sys.version,jax=jax.__version__,torch=torch.__version__,flax=__import__('flax').__version__,
        devices=str(jax.devices()),platform=platform.platform(),problem=a.problem,n_pde=len(inside),n_bc=len(bd),n_P=nx,n_D=ny,
        official_target=target,requested_steps=requested,paired_prefix=a.paired_prefix,seconds_budget=a.seconds,
        official_maxiter=limit,source_sha='df67eb6c6309014b6c2752cfadd59a7483c181b2',
        cpinn_sha='ef0d6a4504e5a5fd69e3d0583f4c821b33436c2d',reference_optimizer='CGDs0.4.5',
        fixture_sha256=hashlib.sha256((out/'fixture.npz').read_bytes()).hexdigest(),dtype='float64',
        lr=.001,beta=.99,eps=1e-8,tol=1e-7,atol=1e-20,
        note='Original Burgers D RNG is unspecified; explicit123 used and tensors shared. Sparse-file H for memory only; not fair timing benchmark.')
    (out/'environment.json').write_text(json.dumps(env,indent=2));print('ENV',json.dumps(env),flush=True)
    rows=[];rlog=[];jt=rt=hist=retry=step=0;reason='requested_steps'
    def record(t,paired):
        l2,pde,bce=map(float,evaluate(opt.x_params_dict))
        row=dict(step=t,jax_l2=l2,jax_pde_mse=pde,jax_bc_mse=bce,elapsed_s=time.perf_counter()-t0,paired=paired,
                 jax_k_total=jt,torch_k_total=rt,torch_historical_total=hist,workspace=opt.solver.capacity,retry_extra_krylov=retry)
        if paired:
            with torch.no_grad():
                points=torch.tensor(eva);v=P(points[:,[0]],points[:,[1]]).numpy().ravel()
            row['torch_l2']=float(np.linalg.norm(v-truth)/norm)
            row['P_param_diff']=delta_metrics(jvec(opt.x_params_dict,q['ppairs']),tvec(P))
            row['D_param_diff']=delta_metrics(jvec(opt.y_params_dict,q['dpairs']),tvec(D))
        if t:
            st=opt.get_stats();row.update(jax_inner=int(st['krylov_iterations']),jax_relative_residual=float(st['relative_residual']),
                jax_failed_total=int(st['failed_solves_total']),jax_rejected_total=int(st['rejected_steps_total']))
            if paired:row['torch_solve']=rlog[-1]
        rows.append(row)
        with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print('ROW',json.dumps(row),flush=True)
        states=dict(torch_P=tvec(P),torch_D=tvec(D),step=np.asarray(t))
        if ref.state['sq_exp_avg_x'] is not None:
            states.update(torch_vD=ref.state['sq_exp_avg_x'].numpy(),torch_vP=ref.state['sq_exp_avg_y'].numpy())
        for name,arr in zip(opt.state._fields,opt.state):states['jax_'+name]=np.asarray(arr)
        np.savez_compressed(out/'last_state.npz',**states)
    record(0,True)
    try:
      with observe_reference(rlog):
        for t in range(1,requested+1):
            paired=a.paired_prefix==0 or t<=a.paired_prefix
            if paired:
                ref.zero_grad();value=q['tloss']();ref.step(-value,value)
                rt+=rlog[-1]['arnoldi'];hist+=rlog[-1]['historical_index']
            old=opt.state
            while True:
                opt.step();st=opt.get_stats();k=int(st['krylov_iterations']);cap=opt.solver.capacity
                if k>=cap and cap<limit:
                    retry+=k;opt._install_state(old,None);opt.solver=replace(opt.solver,capacity=min(2*cap,limit))
                    print('WORKSPACE_GROW',cap,opt.solver.capacity,flush=True);continue
                break
            if bool(st['stopped']):reason='numerical_rejection';break
            jt+=k;step=t
            if t==1 or t%a.report_every==0 or t==requested:record(t,paired)
            if time.perf_counter()-t0>a.seconds:reason='wall_budget';break
    except Exception as error:
        import traceback
        (out/'error.txt').write_text(traceback.format_exc());reason='exception:'+type(error).__name__
        print(traceback.format_exc(),flush=True)
    if rows[-1]['step']!=step:record(step,a.paired_prefix==0 or step<=a.paired_prefix)
    summary=dict(environment=env,completed_steps=step,stop_reason=reason,elapsed_s=time.perf_counter()-t0,final=rows[-1],
        best_jax_recorded=min(r['jax_l2'] for r in rows),completed_official_length=step>=target,
        maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        max_param_abs_diff=max(max(r['P_param_diff']['max_abs'],r['D_param_diff']['max_abs']) for r in rows if r['paired']))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'reference_solver.json').write_text(json.dumps(rlog,indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)
    if reason.startswith('exception:') or reason=='numerical_rejection':raise SystemExit(2)

if __name__=='__main__':main()
