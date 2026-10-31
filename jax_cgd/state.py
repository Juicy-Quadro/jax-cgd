"""Immutable JAX PyTree state for the public pure optimizer transition."""
from typing import NamedTuple
import jax


class OptimizerState(NamedTuple):
    """Numerical state only; objectives, data and RNG belong to the caller.

    ``step`` counts accepted updates. Work/attempt counters also include
    rejected attempts; a failed step never changes the first six fields.
    """
    x: jax.Array
    y: jax.Array
    vx: jax.Array
    vy: jax.Array
    previous: jax.Array
    step: jax.Array
    krylov_total: jax.Array
    matvec_total: jax.Array
    failed_total: jax.Array
    reused_total: jax.Array
    attempted_total: jax.Array
    rejected_total: jax.Array


class SolverFailure(RuntimeError):
    """An update was rejected; the successful prefix and diagnostics are retained."""
