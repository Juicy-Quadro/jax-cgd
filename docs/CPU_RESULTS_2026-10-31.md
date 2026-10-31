# CPU acceptance results, 2026-10-31

Measured source: c1ca09edf82c317c89b4ecafec4db6a1bde8869d.
CI run and raw JSON artifacts:
https://github.com/Juicy-Quadro/jax-cgd/actions/runs/41209009692

## Protocol and scope

GitHub-hosted CPU, JAX 0.10.2, Flax 0.12.8, Python 3.11.17, float64.
Two independent runner jobs per case; 11 randomized timing repetitions per
job, each containing 5 or 10 optimizer updates. Same Flax data/initial state,
CG tol=1e-8, atol=0, max_iter=200. Every sample restores the initial state;
compilation/warmup are separated and completion is synchronized.

Each comparison is paired on the same runner. Different jobs used AMD EPYC
7763, AMD EPYC 9V74 or Intel Xeon 6973P-C hosts. These names describe hosts,
not an allocation of all their cores. Raw outputs include CPU affinity.

Old counted reference: 78d2f7c275ad155cc206f3b0261763e08f4a43bb.
Earlier non-counting reference: b9320efa1d5ef66e88a2a87f215482671ccfcf82.
Neither should be confused with original main 77059bb, whose ACGD timestep
and RMS formula defects make a pure speed comparison misleading.

## Warm median milliseconds per update

Ranges span the two independent jobs; speedups are ratios within each job,
not ratios between endpoints measured on different machines.

| Case | Old counted chunk (ms) | New exact-jet chunk (ms) | Speedup vs counted | Speedup vs earlier non-counting |
|---|---:|---:|---:|---:|
| Poisson 2x16, 128 PDE points | 1.873-1.891 | 1.018-1.239 | 1.53-1.84x | 1.80x |
| Poisson 3x50, 256 PDE points | 36.874-36.888 | 20.707-20.796 | 1.77-1.78x | 1.84-1.90x |
| Poisson 3x50, 2048 PDE points | 238.552-243.385 | 121.503-122.856 | 1.96-1.98x | 2.28-2.33x |
| Burgers 2x16, 32 LHS +456 candidates | 5.303-6.317 | 3.908-4.224 | 1.36-1.50x | 1.32-1.36x |

Geometric mean across all eight paired speedups: **1.70082x**. Minimum 1.35719x.
The proposed >=1.15x geometric-mean and no >5% regression gate is met for
these targeted optimized CPINN cases. This is not a universal speed guarantee.

## Ablation and numerical checks

Core residual reuse alone: generic mode **1.048-1.101x**, factored mode
**0.990-1.101x**. One small factored case was ~1% slower. Thus the full
1.36-1.98x gain must NOT be attributed to the general-purpose optimizer alone.
Exact spatial derivatives contribute **1.234-1.726x** versus the same new
factored core without jets. Compiled chunks also affect small-case dispatch.

All counted samples reported zero failed solves. Cumulative Krylov steps
matched between old counted and new optimized paths in every case:
Poisson-small 40/10 updates, Poisson-256 26/5, Poisson-2048 27/5,
Burgers-small 41/10. Maximum parameter discrepancy across all variants and
cases was **2.945e-12** (the largest Poisson case); smaller cases <=2.995e-14.
No precision reduction, tolerance relaxation or removal of residual checking.

## Verification and review

Local JAX 0.9.0.1 / Flax 0.12.0 CPU: 52 passed after the final module refactor.
Remote Python 3.11 and 3.12 correctness jobs passed; 3.11 log:
52 passed in 64.30s. Full-block GMRES and exact-jet Burgers smoke also passed.
All four workflow groups on measured source completed successfully.
New source/test/benchmark blobs were checked against locally tested files.

Inline review (no separate reviewer agent available) checked true-residual
semantics, custom solver dispatch, no-iteration cases, accounting, implicit
linear-solve AD, Flax parameter reuse, time input rescaling, reference examples,
matched benchmark state and merge scope. Core changes do not alter Krylov
recurrences; public solve_with_stats still verifies residuals independently.

## Limits

These are short optimizer-runtime tests, not time-to-accuracy or a reproduced
paper training curve. Jets specialize the example scalar tanh MLP. Adam
still uses its reference spatial autodiff path, so no Adam-runtime superiority
is claimed. GPU behavior, GPU peak memory, large Burgers paper architecture,
long-run stability and whole-optimizer meta-gradients were not established.
A single-case run may differ due to compilation, workload and host variance.
