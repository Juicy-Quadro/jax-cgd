"""Diagnostic only: same official Poisson state, not a changed training protocol."""
from experiments.official_reference import *
import argparse,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--fixture',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    q=fixture('poisson',5000,50,50,3)
    snap=np.load(a.snapshot,allow_pickle=False);ff=np.load(a.fixture,allow_pickle=False)
    for key,arr in [('inside',q['insidej']),('boundary',q['bcj']),('bc_target',q['target']),('initial_P',tvec(q['P'])),('initial_D',tvec(q['D']))]:
        assert np.array_equal(ff[key],np.asarray(arr)),key
    pred,disc=q['pred'],q['disc'];inside,bc,target=q['insidej'],q['bcj'],q['target']
    def game(x,y):
        return jnp.mean(disc(y,inside)[:,0]*poisson_residual(pred,x,inside))+jnp.mean(disc(y,bc)[:,1]*(pred(x,bc)[:,0]-target))
    factor=BilinearGame(q['jfeat'],q['dfeat'])
    template=ACGD(q['pv'],q['dv'],game,lr=.001,beta=.99,eps=1e-8,formulation='block')
    point={k:jnp.asarray(snap['jax_'+k]) for k in ('x','y','vx','vy','step')}
    state=template.state._replace(**point)
    f=template.f
    @jax.jit
    def vectors(x,y,vx,vy,t):
        gx,gy,hxy,hyx=utils.linearize_game(f,x,y)
        vx=.99*vx+.01*gx**2;vy=.99*vy+.01*gy**2
        ex=.001*jnp.sqrt(1-.99**(t+1))/(jnp.sqrt(vx)+1e-8)
        ey=.001*jnp.sqrt(1-.99**(t+1))/(jnp.sqrt(vy)+1e-8)
        return gx,gy,ex,ey
    gx,gy,ex,ey=vectors(state.x,state.y,state.vx,state.vy,state.step)
    @jax.jit
    def block_residual(dx,dy):
        _,_,hxy,hyx=utils.linearize_game(f,state.x,state.y)
        r=jnp.concatenate((dx+ex*hxy(dy)+ex*gx,dy-ey*hyx(dx)-ey*gy))
        b=jnp.concatenate((-ex*gx,ey*gy))
        return jnp.linalg.norm(r)/jnp.linalg.norm(b)
    results={};updates={}
    for name,form,obj,tol in [('block_1e7','block',game,1e-7),('schur_generic_1e7','schur',game,1e-7),('schur_jet_1e7','schur',factor,1e-7),('block_1e12','block',game,1e-12),('schur_jet_1e12','schur',factor,1e-12)]:
        solver=solvers.GMRES(tol=tol,atol=1e-20,restart=1000,max_iter=1) if form=='block' else solvers.CG(tol=tol,atol=1e-20,max_iter=1000)
        opt=ACGD(q['pv'],q['dv'],obj,lr=.001,beta=.99,eps=1e-8,solver=solver,formulation=form,on_failure='stop')
        s=opt.state._replace(**point);t0=time.perf_counter();ns,stats=opt.update(s)
        jax.block_until_ready(ns);assert not bool(stats['stopped']),name
        dx,dy=ns.x-s.x,ns.y-s.y
        updates[name]=np.concatenate((np.asarray(dx),np.asarray(dy)))
        results[name]=dict(tol=tol,k=int(stats['krylov_iterations']),reported_relative_residual=float(stats['relative_residual']),true_full_block_relative_residual=float(block_residual(dx,dy)),elapsed_with_compile_s=time.perf_counter()-t0)
        print(name,json.dumps(results[name]),flush=True)
    accurate=updates['block_1e12']
    for name,u in updates.items():
        results[name]['max_abs_increment_difference_to_tight_block']=float(np.max(np.abs(u-accurate)))
        results[name]['relative_increment_difference_to_tight_block']=float(np.linalg.norm(u-accurate)/np.linalg.norm(accurate))
    jet=ACGD(q['pv'],q['dv'],factor,lr=.001,beta=.99,eps=1e-8)
    @jax.jit
    def products():
        g0,h0,A0,B0=utils.linearize_game(f,state.x,state.y);g1,h1,A1,B1=utils.linearize_game(jet.f,state.x,state.y)
        vx=jnp.sin(jnp.arange(state.x.size,dtype=state.x.dtype));vy=jnp.cos(jnp.arange(state.y.size,dtype=state.y.dtype))
        return dict(gx=jnp.max(jnp.abs(g0-g1)),gy=jnp.max(jnp.abs(h0-h1)),hxy=jnp.max(jnp.abs(A0(vy)-A1(vy))),hyx=jnp.max(jnp.abs(B0(vx)-B1(vx))))
    derivative_checks={k:float(v) for k,v in products().items()}
    result=dict(source_step=int(state.step),mode='all candidates cold, one common state; diagnostic tightened tolerances not official training settings',solves=results,jet_vs_nested_derivatives=derivative_checks)
    (a.output/'same_state_diagnostics.json').write_text(json.dumps(result,indent=2));np.savez_compressed(a.output/'increments.npz',**updates)
    print('SUMMARY',json.dumps(result),flush=True)
if __name__=='__main__':main()
