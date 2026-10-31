# Stability release candidate design

Base: b3271dd2b0333843077dcfd5c9d4f5a0ae945be8. User requested stability and
release-candidate preparation. Work in release/stability-rc; do not publish to
PyPI or merge main in this task. Version: 2.0.0rc1, because failure defaults,
Python support floor and previously corrected update rules are behavioral changes.

## Choices
A packaging-only release would leave unsafe updates; a full optimizer rewrite
would make the existing numerical evidence harder to reuse. Choose incremental
hardening of the tested update kernel, with explicit JAX-array state and a small
pure state-transition API. No new optimization algorithm, no relaxed tolerance.

## Contracts
- Policies: raise (default), stop, accept_inexact. Hard failures (breakdown or
  nonfinite candidate/gradient/moment/solution) are never committed. A rejected
  attempt stops its chunk. Successful prefix remains committed. RMS, warm start
  and accepted timestep only advance on accepted steps. Attempts/work/failures
  count actual attempts, including rejected attempts.
- Pure update(state, steps=...) returns new state and diagnostics, never raises
  on device. The class wrapper raises only at chunk boundary under raise policy.
- Parameter trees: real homogeneous float32/float64 array leaves; preserve flat
  dictionary order for legacy metadata; validate emptiness, dtype and finite
  initial values. Nested registered PyTrees supported without depending on Flax.
- Checkpoints: schema-versioned NPZ+JSON, no pickle. Full numeric state, counters
  and diagnostics. Atomic file replacement. Strict metadata/config/dtype checks
  before mutation. Caller supplies compatible game/data; checkpoint does not
  serialize executable code, data, PRNG or unowned training-loop state.
- Standalone solve/solve_with_stats retain signatures and implicit solver AD.
  Old step_func remains low-level proposal API, documented not failure guarded.
- Wheel/sdist tests outside checkout; single version source, honest support
  matrix, separate optional Flax dependencies, no automatic GPU installation.

## Non-goals / release gates
No general-sum/complex support, no automatic retry/precision changes, no promise
of whole-optimizer meta-gradients, GPU performance or full CPINN reproduction.
Tests: red/green failure injection, exact resume, corrupted checkpoint rejection,
PyTree/jit/vmap/scan, long bilinear trajectories, float32/64 and all prior tests.
Artifact build, metadata check, installed-wheel tests, Python/JAX CI matrix.
Any missing GPU or platform evidence remains an explicit release blocker/limit.
