# Stability RC Implementation Plan

Goal: protect state on numerical failure and prepare an installable, auditable RC.
Spec: docs/superpowers/specs/2026-10-31-stability-rc.md.
Tech: public JAX APIs, NumPy, setuptools, pytest; optional Flax/Optax.

- [x] Failure injection tests first; show old implementation failure.
- [x] Guard proposals in compiled chunks; pure state API; class failure policy.
- [x] Test and add validated parameter PyTrees, state snapshots, atomic NPZ resume.
- [x] Add long-run/dtype and corrupted-checkpoint regression tests.
- [x] Single 2.0.0rc1 version, pyproject metadata, clean artifact tests and CI.
- [ ] Run local CPU and remote wheel/matrix validation; inspect evidence.
- [ ] Review diff; create PR and deliver artifacts, without publish or merge.

Review risks: zero-iteration/nonfinite-gradient failure, rejection after accepted
prefix, counter correctness, partial checkpoint load, false wheel imports from
checkout, missing CPU backend for GPU debug callbacks. No subagent tool available;
perform explicit inline review and report that limitation.

Baseline: fresh full local run timed out after 120 seconds (21 tests completed),
not counted as a full pass. Prior 52-test CI is historical evidence only.

Local evidence: 11 failure tests first failed on old main; checkpoint and metadata
RED tests also recorded. Full candidate suite: 89 passed in 114.17s (JAX0.9.0.1,
Flax0.12.0, Python3.13.5, CPU). Built wheel and sdist successfully. Review found
and fixed float32 rate promotion when x64 counters are enabled, and malformed
JSON manifest handling. Remote installed-wheel matrix remains pending.
