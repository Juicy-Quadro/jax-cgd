"""Versioned, atomic, non-pickle checkpoints and reproducible resume."""
import copy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_cgd import ACGD, BCGD, solvers


def fun(x,y):
    return jnp.sum(x['w']*y['w'])+.25*jnp.sum(x['w']**2-y['w']**2)


def make(cls=ACGD, dtype=jnp.float64, **kw):
    args=dict(lr=.01,solver=solvers.CG(tol=1e-8 if dtype==jnp.float64 else 1e-5,
                                    atol=1e-12 if dtype==jnp.float64 else 1e-7))
    if cls is ACGD: args.update(beta=.9,eps=.1)
    args.update(kw)
    return cls({'w':jnp.array([.2,.7],dtype=dtype)}, {'w':jnp.array([.4,-.3],dtype=dtype)},fun,**args)


def assert_state_equal(a,b):
    for x,y in zip(jax.tree.leaves(a.state),jax.tree.leaves(b.state)):
        np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('cls',[BCGD,ACGD])
def test_checkpoint_resume_matches_uninterrupted(tmp_path,cls):
    full, part, restored=make(cls),make(cls),make(cls)
    full.step_many(40);part.step_many(17)
    path=tmp_path/'restart.npz';part.save_checkpoint(path)
    restored.load_checkpoint(path);assert_state_equal(restored,part)
    restored.step_many(23);assert_state_equal(restored,full)


def test_state_dict_is_detached_copy_and_initial_roundtrip():
    a,b=make(),make()
    snap=a.state_dict();b.load_state_dict(snap);assert_state_equal(a,b)
    snap['state']['x'][0]=123
    assert float(a.x_params[0])!=123


@pytest.mark.parametrize('field,value',[
    ('vx',np.array([-1.,0.])),('vy',np.array([np.nan,0.])),
    ('x',np.array([1.,2.],dtype=np.float32)),('previous',np.ones(3)),
    ('step',np.array(-1,dtype=np.int64)),('attempted_total',np.array(99,dtype=np.int64)),
])
def test_invalid_checkpoint_never_partially_mutates(field,value):
    a,b=make(),make();a.step_many(3);b.step_many(2)
    before=copy.deepcopy(b.state_dict());bad=a.state_dict();bad['state'][field]=value
    with pytest.raises((ValueError,TypeError)):
        b.load_state_dict(bad)
    after=b.state_dict()
    for key in before['state']:
        np.testing.assert_array_equal(after['state'][key],before['state'][key])


def test_config_layout_schema_mismatch_rejected():
    source=make();source.step()
    for field,value in [('schema_version',999),('layout',{}),('config',{})]:
        snap=source.state_dict();snap[field]=value
        with pytest.raises(ValueError):make().load_state_dict(snap)
    with pytest.raises(ValueError,match='config'):
        make(lr=.1).load_state_dict(source.state_dict())


def test_rejected_update_checkpoint_preserves_failure_evidence(tmp_path):
    a=make(on_failure='stop',solver=solvers.CG(max_iter=0))
    a.step_many(10);path=tmp_path/'failed.npz';a.save_checkpoint(path)
    b=make(on_failure='stop',solver=solvers.CG(max_iter=0))
    b.load_checkpoint(path);assert_state_equal(a,b)
    assert int(b.get_stats()['failed_solves_total'])==1
    assert bool(b.get_stats()['stopped'])


def test_file_contains_no_pickle_or_object_arrays(tmp_path):
    a=make();a.step();path=tmp_path/'safe.npz';a.save_checkpoint(path)
    with np.load(path,allow_pickle=False) as data:
        assert '__manifest__' in data
        assert all(data[k].dtype.kind!='O' for k in data.files)
    path.write_bytes(b'broken checkpoint')
    before=make();snapshot=before.state_dict()
    with pytest.raises(ValueError):before.load_checkpoint(path)
    np.testing.assert_array_equal(before.x_params,snapshot['state']['x'])


def test_failed_atomic_save_keeps_existing_checkpoint(tmp_path,monkeypatch):
    import os
    a=make();path=tmp_path/'model.npz';a.save_checkpoint(path);old=path.read_bytes()
    def fail(*args):raise OSError('simulated rename failure')
    monkeypatch.setattr(os,'replace',fail)
    with pytest.raises(OSError):a.save_checkpoint(path)
    assert path.read_bytes()==old
    assert list(tmp_path.iterdir())==[path]


@pytest.mark.parametrize('cls',[BCGD,ACGD])
@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
def test_long_bilinear_trajectory_and_resume(tmp_path,cls,dtype):
    full,half,resumed=make(cls,dtype),make(cls,dtype),make(cls,dtype)
    full.step_many(1000);half.step_many(500)
    path=tmp_path/'long.npz';half.save_checkpoint(path);resumed.load_checkpoint(path)
    resumed.step_many(500);assert_state_equal(full,resumed)
    assert int(full.get_stats()['failed_solves_total'])==0
    assert int(full.timestep)==1000
    assert np.isfinite(np.asarray(full.x_params)).all()
    assert float(jnp.linalg.norm(full.x_params)+jnp.linalg.norm(full.y_params))<.1


def test_nested_tree_and_block_gmres_resume(tmp_path):
    def make_nested():
        return ACGD({'a':(jnp.array([.2,.1]),)},[{'b':jnp.array([.3,.6])}],
          lambda x,y:jnp.vdot(x['a'][0],y[0]['b']),lr=.01,eps=.1,
          solver=solvers.GMRES(tol=1e-9,atol=1e-12),formulation='block')
    a,b=make_nested(),make_nested();a.step_many(3)
    a.save_checkpoint(tmp_path/'nested.npz');b.load_checkpoint(tmp_path/'nested.npz')
    assert_state_equal(a,b);a.step_many(4);b.step_many(4);assert_state_equal(a,b)


def test_nonfinite_failure_diagnostics_roundtrip(tmp_path):
    def build():
        return BCGD({'w':jnp.ones(2)},{'w':jnp.ones(2)},
          lambda x,y:jnp.sqrt(-jnp.sum(x['w']**2))+jnp.sum(y['w']),on_failure='stop')
    a,b=build(),build();a.step();a.save_checkpoint(tmp_path/'nonfinite.npz')
    b.load_checkpoint(tmp_path/'nonfinite.npz');assert_state_equal(a,b)
    assert bool(b.get_stats()['hard_failure'])


def test_bad_manifest_is_valueerror_without_mutation(tmp_path):
    a=make();before=a.state_dict();path=tmp_path/'bad.npz'
    np.savez(path,__manifest__=np.asarray('null'))
    with pytest.raises(ValueError):a.load_checkpoint(path)
    np.testing.assert_array_equal(a.x_params,before['state']['x'])
