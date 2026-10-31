> Historical notes for PR #1, not the current feature list. Counted solvers and
> Flax benchmarks were subsequently added. Current behavior is documented in
> [WORK_ACCOUNTING](docs/WORK_ACCOUNTING.md) and
> [CPU_OPTIMIZATION](docs/CPU_OPTIMIZATION.md).

# Matrix-free CGD/ACGD optimization patch

This change is based on the repository main branch at
77059bbfa2dab3d04ec247083761bee9626037df.

## Changes

- Fix BCGD/ACGD step counters: the previous code did not persist timestep
  updates, so adaptive bias correction was wrong after the first step.
- Compute both gradients in one jax.grad call.
- Use jax.linearize and jax.linear_transpose to reuse the same mixed-Hessian
  linearization across Krylov matvecs, RHS construction and the final update.
- Keep the warm-start array shape stable, avoiding a first/second-step
  recompilation caused by switching from None to an array.
- For real zero-sum objectives and positive scalar learning rates, the CGD
  Schur complement is SPD, so use CG as the default solver rather than GMRES.
  A user-provided GMRES solver is still accepted.
- Keep the public BCGD/ACGD class interfaces and historical Operater spelling.

## Validation

Local CPU: JAX 0.9.0.1, Python 3.13, x64 enabled:
python -m pytest -q tests/test_optimizer.py => 9 passed.

Run the synthetic microbenchmark with:
python benchmark.py

## Limitations

- The default relative tolerance is now 1e-6 rather than 1e-10. Set an
  explicit CG/GMRES solver for tighter scientific error requirements.
- Cached jax.linearize intermediates can increase memory footprint on
  large neural networks; compare GPU peak memory before adopting.
- The existing flat dictionary interface does not support arbitrary nested
  FrozenDicts. A PyTree-native API is separate future work.
- This change does not implement counted Krylov iterations; JAX native
  solver info is not an iteration-count report.
- The benchmark is synthetic and not proof of overall speedup for PINNs,
  large models or GPU execution.
