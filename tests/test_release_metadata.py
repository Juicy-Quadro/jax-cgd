"""Source and installed distribution must identify the same release candidate."""
from pathlib import Path
import importlib.metadata as metadata
import tomllib
import jax_cgd


def test_pyproject_is_single_metadata_source():
    path=Path(__file__).parents[1]/'pyproject.toml'
    assert path.exists(), 'Missing modern build/metadata declaration'
    data=tomllib.loads(path.read_text())
    assert data['project']['dynamic']==['version']
    assert data['tool']['setuptools']['dynamic']['version']['attr']=='jax_cgd._version.__version__'
    assert data['project']['requires-python']=='>=3.11,<3.14'


def test_installed_metadata_matches_runtime():
    assert metadata.version('jax-cgd')==jax_cgd.__version__=='2.0.0rc1'
    req=metadata.requires('jax-cgd')
    assert any(r.startswith('jax<0.13,>=0.9.0') or r.startswith('jax>=0.9.0,<0.13') for r in req)
    assert all('extra ==' in r for r in req if r.startswith(('flax','optax')))
