"""Versioned optimizer snapshots. NPZ numeric arrays + JSON; never pickle."""
import json
import os
from pathlib import Path
import tempfile
import zipfile
import jax
import jax.numpy as jnp
import numpy as np
from . import solvers
from ._version import __version__
from .state import OptimizerState

_SCHEMA = 1


def _config(opt):
    if type(opt.solver) not in (solvers.CG, solvers.GMRES):
        raise TypeError('Checkpoint configuration supports built-in CG/GMRES only')
    linear = {key: getattr(opt.solver, key) for key in ('tol', 'atol', 'max_iter')}
    if type(opt.solver) is solvers.GMRES:
        linear['restart'] = opt.solver.restart
    return dict(optimizer=type(opt).__module__+'.'+type(opt).__qualname__,
                lr=opt.eta, beta=opt.beta, eps=opt.eps, adaptive=opt.adaptive,
                formulation=opt.formulation, on_failure=opt.on_failure,
                solver=type(opt.solver).__name__, solver_options=linear)


def _layout(opt):
    return dict(x=repr(opt.x_metadata), y=repr(opt.y_metadata),
                dtype=str(opt.x_params.dtype), counter_dtype=str(opt.state.step.dtype))


def state_dict(opt):
    """Host copies; does not serialize the objective, data or caller PRNG state."""
    host = lambda value: np.array(jax.device_get(value), copy=True)
    return dict(schema_version=_SCHEMA, package_version=__version__,
                config=_config(opt), layout=_layout(opt),
                state={key: host(value) for key, value in opt.state._asdict().items()},
                stats=None if opt._stats is None else {k: host(v) for k,v in opt._stats.items()})


def _validated(opt, snapshot):
    required = {'schema_version', 'package_version', 'config', 'layout', 'state', 'stats'}
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise ValueError('Invalid checkpoint fields')
    if type(snapshot['schema_version']) is not int or snapshot['schema_version'] != _SCHEMA:
        raise ValueError('Unsupported checkpoint schema_version')
    if snapshot['package_version'] != __version__:
        raise ValueError('Checkpoint package_version mismatch; explicit migration is required')
    if snapshot['config'] != _config(opt):
        raise ValueError('Checkpoint config mismatch')
    if snapshot['layout'] != _layout(opt):
        raise ValueError('Checkpoint parameter layout or dtype mismatch')
    data = snapshot['state']
    reference = opt.state._asdict()
    if not isinstance(data, dict) or set(data) != set(reference):
        raise ValueError('Invalid state fields')
    values = {}
    for key, ref in reference.items():
        value = np.asarray(data[key])
        if value.shape != ref.shape or value.dtype != np.dtype(ref.dtype):
            raise ValueError(f'Checkpoint shape/dtype mismatch: {key}')
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite checkpoint state: {key}')
        if key not in ('x', 'y', 'previous') and np.any(value < 0):
            raise ValueError(f'Negative moment/counter: {key}')
        values[key] = value.copy()
    n = {key: int(values[key]) for key in OptimizerState._fields[5:]}
    if (n['step']+n['rejected_total'] != n['attempted_total'] or
        not n['rejected_total'] <= n['failed_total'] <= n['attempted_total'] or
        n['reused_total'] > n['attempted_total'] or
        n['krylov_total'] > n['matvec_total']):
        raise ValueError('Inconsistent checkpoint counters')
    stats = snapshot['stats']
    if stats is None:
        if n['attempted_total'] != 0:
            raise ValueError('Missing diagnostics for a noninitial checkpoint')
    else:
        # Zero-step transition creates the schema without evaluating the game.
        _, schema = opt.update(opt.state, steps=0)
        if not isinstance(stats, dict) or set(stats) != set(schema):
            raise ValueError('Invalid checkpoint diagnostic fields')
        for key, ref in schema.items():
            value = np.asarray(stats[key])
            if value.shape != () or value.dtype != np.dtype(ref.dtype):
                raise ValueError(f'Invalid diagnostic dtype/shape: {key}')
        if not bool(stats['counts_available']):
            raise ValueError('Built-in solver checkpoint must have available counts')
        correspond = dict(outer_steps='step', attempted_steps_total='attempted_total',
                          rejected_steps_total='rejected_total', krylov_iterations_total='krylov_total',
                          matvecs_total='matvec_total', failed_solves_total='failed_total',
                          mixed_products_reused_total='reused_total')
        for stat, counter in correspond.items():
            if int(stats[stat]) != n[counter]:
                raise ValueError(f'Diagnostic/state counter mismatch: {stat}')
        for key, value in stats.items():
            arr = np.asarray(value)
            if arr.dtype.kind == 'f':
                if np.isfinite(arr):
                    if arr < 0:
                        raise ValueError('Negative residual diagnostic')
                elif not (bool(stats['hard_failure']) and not bool(stats['converged'])):
                    raise ValueError('Nonfinite diagnostic without hard failure')
            elif arr.dtype.kind in 'iu' and int(arr) < 0:
                raise ValueError('Negative diagnostic counter')
        if bool(stats['converged']) and bool(stats['breakdown']):
            raise ValueError('Conflicting convergence diagnostics')
        stats = {key: jnp.asarray(value) for key, value in stats.items()}
    # Conversion is completed before any live state is installed.
    state = OptimizerState(**{key: jnp.asarray(value) for key,value in values.items()})
    return state, stats


def load_state_dict(opt, snapshot):
    state, stats = _validated(opt, snapshot)
    opt._install_state(state, stats)


def save_checkpoint(opt, path):
    """Atomically replace a checkpoint. A failed write preserves the old file."""
    snapshot = state_dict(opt)
    _validated(opt, snapshot)
    manifest = {k: v for k,v in snapshot.items() if k not in ('state', 'stats')}
    manifest['has_stats'] = snapshot['stats'] is not None
    payload = {'state__'+key: value for key,value in snapshot['state'].items()}
    if snapshot['stats'] is not None:
        payload.update({'stats__'+key: value for key,value in snapshot['stats'].items()})
    payload['__manifest__'] = np.asarray(json.dumps(manifest, allow_nan=False, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name+'.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def load_checkpoint(opt, path):
    """Read NPZ with allow_pickle=False; validate all fields before mutation.

    Use trusted files: disallowing pickle prevents code deserialization, not
    arbitrary resource exhaustion from malicious/oversized archives.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            manifest_array = archive['__manifest__']
            if manifest_array.shape != () or manifest_array.dtype.kind not in 'US':
                raise ValueError('Invalid checkpoint manifest')
            manifest = json.loads(str(manifest_array.item()))
            if not isinstance(manifest, dict):
                raise ValueError('Checkpoint manifest must be an object')
            has_stats = manifest.pop('has_stats')
            if type(has_stats) is not bool:
                raise ValueError('Invalid has_stats flag')
            names = set(archive.files)
            if len(names) != len(archive.files):
                raise ValueError('Duplicate checkpoint members')
            state_names = {'state__'+key for key in OptimizerState._fields}
            extra = names - state_names - {'__manifest__'}
            if not state_names <= names or any(not k.startswith('stats__') for k in extra):
                raise ValueError('Unexpected checkpoint arrays')
            if bool(extra) != has_stats:
                raise ValueError('Inconsistent diagnostic manifest')
            snapshot = dict(manifest, state={key: archive['state__'+key].copy()
                                            for key in OptimizerState._fields},
                            stats={key[7:]: archive[key].copy() for key in extra} if has_stats else None)
    except (OSError, KeyError, TypeError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f'Cannot read checkpoint: {exc}') from exc
    load_state_dict(opt, snapshot)
