# Changelog

## 2.0.0rc1 (unreleased)

This is an unpublished release candidate, NOT a stable release. The previous
PyPI release is 1.0.4; older Git metadata incorrectly said 0.1.4. Use the Git SHA
for old experiment provenance. No tag or PyPI publication is performed here.

### Breaking/safety changes
- Default `on_failure="raise"` rejects an inexact solve, stops the chunk and
  raises `SolverFailure` at its boundary. Successful prefix remains committed.
- `stop` exposes status without a host raise; `accept_inexact` explicitly permits
  finite nonconverged proposals. No policy commits breakdown/nonfinite updates.
- RMS state, accepted step and warm start only advance on accepted updates.
  Work/attempt/failure counters include rejected attempts.
- Python 3.11-3.13; JAX >=0.9,<0.13. Version-range metadata is not a claim that
  every combination has been tested. See docs/RELEASE_CANDIDATE.md.
- Inputs must have homogeneous float32/float64 leaves and finite initial values.
  Nested registered PyTrees are now accepted; flat-dict order stays compatible.

### Added/fixed
- Immutable `OptimizerState`; pure `opt.update(state, steps=...)` for JAX transforms.
- Schema-versioned state_dict and atomic non-pickle NPZ checkpoints with strict
  configuration, layout, dtype and numeric-state validation before restore.
- Float32 ACGD rates no longer promote to float64 when x64 counters are enabled.
- Single source version; wheel/sdist validation and installed-wheel CI matrix.
- Failure injection, long-run/resume, corrupt-file and PyTree regression tests.

### Inherited from unreleased main
- Correct persistent timesteps and elementwise RMS (old-main trajectories change).
- Counted CG/GMRES and true residuals; default SPD Schur-CG, optional block GMRES.
- Flax CPINN examples and exact tanh spatial derivatives. Prior CPU benchmarks
  concern their pinned commits, not the extra safety checks introduced here.

### Not promised
No full optimizer meta-gradients, general-sum/complex games, preconditioning,
automatic retries, full-paper accuracy, RC GPU verification or universal speedup.
