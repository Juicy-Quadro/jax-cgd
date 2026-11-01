"""Load frozen official definitions without running plotting/training top-level code.
Original numerical method bodies and CGDs0.4.5 are unchanged. Modern CPU runtime.
Burgers D's unspecified starting RNG is explicitly seeded123, then tensors shared.
"""
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import sys, json, math, ast, importlib, hashlib, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as tnn
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import unflatten_dict
from scipy.io import loadmat
from pyDOE import lhs
import CGDs
from CGDs.gmres_torch import GMRES as torch_gmres
from jax_cgd import ACGD, solvers, utils
from jax_cgd.games import BilinearGame
from examples.cpinn_flax import MLP, poisson_residual, burgers_residual, exact_poisson
from examples.spatial_jets import JetMLP
ROOT = Path(__file__).resolve().parents[1]
OFF = ROOT/'vendor/CPINN'
torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
jax.config.update('jax_enable_x64', True)

def definitions(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    ns = dict(torch=torch, nn=tnn, autograd=torch.autograd, np=np, math=math,
              device=torch.device('cpu'), lhs=lhs)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns

def arrays(pairs):
    result = {}
    for name, layer in pairs:
        result[('params',name,'kernel')] = jnp.asarray(layer.weight.detach().numpy().T.copy())
        result[('params',name,'bias')] = jnp.asarray(layer.bias.detach().numpy().copy())
    return result

def jvec(tree,pairs):
    return np.concatenate([np.r_[np.asarray(tree[('params',name,'kernel')]).T.ravel(),
        np.asarray(tree[('params',name,'bias')]).ravel()] for name,_ in pairs])

def tvec(model):
    return np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])

def delta_metrics(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return dict(max_abs=float(np.max(np.abs(a-b))),
        relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)))

class SharedDiscriminator(nn.Module):
    width: int
    layers: int
    @nn.compact
    def __call__(self,z):
        z=nn.Dense(self.width,name='Dense_0',dtype=jnp.float64,param_dtype=jnp.float64)(z)
        shared=nn.Dense(self.width,name='Shared',dtype=jnp.float64,param_dtype=jnp.float64)
        for _ in range(self.layers-1):
            z=shared(nn.relu(z))
        return nn.Dense(2,name='Output',dtype=jnp.float64,param_dtype=jnp.float64)(nn.relu(z))

def fixture(problem,n_pde,n_bc,width,layers):
    torch.manual_seed(123)
    if problem=='poisson':
        util=definitions(OFF/'Poisson/utils.py');orig=definitions(OFF/'Poisson/networks.py')
        u=lambda xy:np.sin(xy[:,0])*np.cos([xy[:,1]])
        ff=lambda x,y:-2*np.sin(x)*np.cos(y)
        _,bc,ub,inside,ffv=util['trainingData'](np.array([-2,-2]),np.array([2,2]),n_bc,n_pde,u,ff,123)
        P=orig['PINN_Poisson_2d'](np.array([2]+[width]*layers+[1]),bc[:,[0]],bc[:,[1]],ub,ffv,
            bc[:,[0]],bc[:,[1]],ub,ffv,inside[:,[0]],inside[:,[1]],123)
        D=orig['Discriminator'](2,width//2,2)
        ppairs=[(f'Dense_{i}',m) for i,m in enumerate(P.linears)]
        dpairs=[(f'Dense_{i}',m) for i,m in enumerate(m for m in D.map if isinstance(m,tnn.Linear))]
        dm=MLP(4,width,2,'relu',False);scale=np.array([1.,1.]);shift=np.array([0.,0.])
        def tloss():
            r=P.test_PDE()
            return (D(P.x_inside_train,P.y_inside_train)[:,[0]]*r).mean()+(D(P.x_bc,P.y_bc)[:,[1]]*(P(P.x_bc,P.y_bc)-P.u_bc)).mean()
    else:
        orig=definitions(OFF/'Burger/BurgerTrainGACGD.py')
        data=loadmat(OFF/'Burger/burgers_shock.mat')
        xx,tt=np.meshgrid(data['x'].ravel(),data['t'].ravel());exact=np.real(data['usol']).T
        points=np.c_[xx.ravel(),tt.ravel()];lo=points.min(0);hi=points.max(0)
        pool=np.vstack((np.c_[xx[0,:],tt[0,:]],np.c_[xx[:,0],tt[:,0]],np.c_[xx[:,-1],tt[:,-1]]))
        targets=np.r_[exact[0,:],exact[:,0],exact[:,-1]][:,None]
        np.random.seed(123)
        ins=lo+(hi-lo)*lhs(2,n_pde);ins=np.vstack((ins,pool))
        idx=np.random.choice(len(pool),n_bc,replace=False)
        bc=torch.tensor(pool[idx]);ub=torch.tensor(targets[idx]);inside=torch.tensor(ins)
        D=orig['Discriminator']([2,layers,width,2])
        P=orig['PINN_Burger'](bc.numpy(),ub.numpy(),inside.numpy(),[2,layers,width,1],lo,hi,.01/np.pi,points,exact.ravel()[:,None])
        ppairs=[(f'Dense_{i}',m) for i,m in enumerate(P.linears)]
        dpairs=[('Dense_0',D.linears[0]),('Shared',D.linears[2]),('Output',D.linears[-1])]
        dm=SharedDiscriminator(width,layers);scale=2/(hi-lo);shift=-1-scale*lo
        def tloss():
            r=P.PDE_residual();_,diff=P.loss_BC()
            C=(D(P.x_u,P.t_u)[:,[0]]*diff).mean()+(D(P.x_f,P.t_f)[:,[1]]*r).mean()
            return -C
    pv,dv=arrays(ppairs),arrays(dpairs)
    pm=MLP(layers,width,1,'tanh',False);jm=JetMLP(layers,width,1,'tanh',False)
    s=jnp.asarray(scale);shiftj=jnp.asarray(shift)
    insidej=jnp.asarray(inside.numpy());bcj=jnp.asarray(bc.numpy());target=jnp.asarray(ub.numpy().ravel())
    pred=lambda p,z:pm.apply(unflatten_dict(p),s*z+shiftj)
    disc=lambda d,z:dm.apply(unflatten_dict(d),z)
    resfun=poisson_residual if problem=='poisson' else burgers_residual
    pi,bi,sign=(0,1,1.) if problem=='poisson' else (1,0,-1.)
    def jfeat(p):
        u,du,lap=jm.apply(unflatten_dict(p),insidej*s+shiftj,
            second_axes=(0,1) if problem=='poisson' else (0,),method=jm.spatial_jet)
        r=lap+2*exact_poisson(insidej) if problem=='poisson' else s[1]*du[:,1]+u*s[0]*du[:,0]-(.01/np.pi)*s[0]**2*lap
        return jnp.concatenate((r/jnp.sqrt(float(len(insidej))),
            (pred(p,bcj)[:,0]-target)/jnp.sqrt(float(len(bcj)))))
    def dfeat(d):
        v=disc(d,jnp.concatenate((insidej,bcj)))
        return jnp.concatenate((sign*v[:len(insidej),pi]/jnp.sqrt(float(len(insidej))),
            v[len(insidej):,bi]/jnp.sqrt(float(len(bcj)))))
    return locals()

def eval_arrays(q):
    if q['problem']=='poisson':
        util=definitions(OFF/'Poisson/utils.py')
        _,_,xy,ut,_,_,_,_=util['testingData'](np.array([-2,-2]),np.array([2,2]),
            lambda x,y:np.sin(x)*np.cos(y),lambda x,y:-2*np.sin(x)*np.cos(y),256)
        return np.asarray(xy),ut.numpy().ravel()
    data=loadmat(OFF/'Burger/burgers_shock.mat');xx,tt=np.meshgrid(data['x'].ravel(),data['t'].ravel())
    return np.c_[xx.ravel(),tt.ravel()],np.real(data['usol']).T.ravel()
