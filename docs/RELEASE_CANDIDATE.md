# 2.0.0rc1 — stability release candidate (not published)

## Scope and installation

Real, differentiable, two-player zero-sum games. This RC is prepared on a branch;
no PyPI upload, Git release/tag or main merge is implied. The earlier PyPI version
is 1.0.4 and old Git metadata 0.1.4 was inconsistent. Behavioral changes justify a
2.x release candidate rather than silently replacing the 1.x distribution.

Install a built wheel, or `pip install -e '.[examples,test]'` in a source checkout.
`jax_cgd.__version__` and distribution metadata have one source `_version.py`.
Flax/Optax are optional extras, not core imports. Examples are available in the
source distribution/checkout; they are not installed as a top-level wheel package.

Declared target: Python >=3.11,<3.14; JAX >=0.9,<0.13; float32/float64. Linux CPU is
the RC validation target. CI checks minimum JAX 0.9.0, reference 0.10.2 and newest
within the declared range on Python 3.11/3.12/3.13 respectively. This is NOT a claim
that every version combination or Windows/macOS/TPU has been tested. See PR/logs
for actual outcomes; a configured workflow alone is not evidence of a pass.

GPU remains a release-validation gap. Prior L4 results were on commit 78d2f7c,
not this RC. For GPU tests that use host debug callbacks use
`JAX_PLATFORMS=cuda,cpu`, then assert actual compute/result devices are GPU.
Do not silently fall back to CPU or automatically install/change GPU drivers.

## Numerical failure policy

`ACGD(..., on_failure="raise")` / `BCGD(...)` is the default:
- Verify the linear solve's true residual and all candidate numerical state.
- Reject nonconvergence, breakdown or nonfinite proposals; stop at first rejection.
- The successful prefix of `step_many` is committed, NOT rolled back as a whole.
- Failed proposal leaves parameters, RMS, warm start and accepted timestep unchanged.
- Counters/diagnostics retain the attempted work; default raises SolverFailure at
  the chunk boundary (one host synchronization). Catch it and inspect get_stats.

`on_failure="stop"` has the same rejection rule without a Python raise. Inspect
`stopped` and `accepted_steps`; it does NOT repeat a failing update for the rest of
the chunk. `accept_inexact` commits a finite nonconverged proposal intentionally,
but never commits hard breakdown/nonfinite state. No hidden retries or relaxed
precision/tolerance. Per-update safety adds work; rebenchmark for performance.

A finite update can still worsen the objective or leave a problem's mathematical
domain; this package cannot infer every objective constraint. If the next attempt
becomes invalid, it is rejected. This is not a line search or convergence proof.

`outer_steps` is accepted updates. `attempted_steps_total` includes rejections;
`paper_proxy_total` charges real attempts, not just accepted updates, and remains
an algorithmic proxy, not measured AD calls. Under vmap, masked work can execute
on already-converged lanes: counters represent logical per-system work.

Custom solvers with `solve` only have their true residual checked with their
`tol`/`atol`, or defaults 1e-6/0. Counts are -1. Custom `solve_with_stats` providers
must honor the SolverStats contract. Public standalone `solve()` return semantics
are unchanged. Low-level legacy `step_func` is a proposal helper and does not
provide the guarded state policy; use the class or pure update for that contract.

## PyTrees and pure state transitions

Nested registered PyTrees with nonempty, real homogeneous float32/float64 array
leaves are supported. Both players use the same dtype. Flat ordinary dictionaries
retain insertion order. Constructor validation runs on the host; initialization
is not a JIT operation. No silent mixed-dtype promotion or float64 truncation.

```python
import jax
from jax_cgd import ACGD, solvers
opt = ACGD(x, y, game, solver=solvers.CG(tol=1e-8, atol=1e-12))
state = opt.state
advance = jax.jit(lambda s: opt.update(s, steps=20))
state, diagnostics = advance(state)  # pure: does NOT mutate opt
# Pure update returns stopped even with policy='raise'; never raises on device.
# Inspect diagnostics before continuing or trusting an experiment.
```

State is an immutable NamedTuple PyTree. Only pass valid states from a matching
optimizer/validated checkpoint; this is not a verifier for arbitrary forged raw
state. Keep objective/configuration fixed after JIT tracing. Caller owns data and
PRNG. Whole-optimizer reverse-mode/meta-gradients remain outside the contract;
implicit differentiation of standalone converged linear solves is tested separately.

## Checkpoints and restart

```python
opt.step_many(20)
opt.save_checkpoint('results/optimizer.npz')
other = ACGD(x, y, game, solver=solvers.CG(tol=1e-8, atol=1e-12))
other.load_checkpoint('results/optimizer.npz')
other.step_many(20)
```

`state_dict()` returns detached host copies. `load_state_dict()` validates schema,
package version, optimizer/solver configuration, tree layout, shapes, dtype,
finite numerical state, nonnegative moments and counter consistency before any
live state is replaced. Changed config is rejected, not silently coerced.
Checkpoint support is for exact built-in CG/GMRES types; custom solvers need a
separate explicit serialization contract. Checkpoints include warm start, RMS,
accepted timestep, attempted work and last diagnostics (including failure evidence).

NPZ arrays + JSON only; `allow_pickle=False`; atomic same-directory rename preserves
an existing checkpoint if writing fails. Use trusted checkpoint files: no pickle
does not prevent all malformed-file/resource-exhaustion attacks. Recreate the SAME
objective, data, external RNG and software environment for meaningful resume;
those objects are not serialized and their identity cannot be inferred from shapes.
Exact restart was tested for deterministic CPU trajectories; bitwise agreement
across hardware, JAX versions or changed input data is not promised.

## Before a stable release

Record completed installed-wheel/matrix tests and artifact hashes; run this exact
RC on GPU; investigate failures without deleting tests; measure safety overhead;
run representative long CPINN training and same-budget comparisons separately.
No speed or full-paper-accuracy result from an older SHA transfers automatically.
