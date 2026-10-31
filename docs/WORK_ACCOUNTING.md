# Counted Krylov work and optimized Flax CPINNs

## What the paper counts

arXiv:2204.11144v2 section 3.1 and Figure 1 say **Forward Passes**, not literal
backward calls: Adam costs 1 per update; ACGD costs `2 + 2*k`, with `k` the
GMRES inner iteration count. Thus `F_Adam=T` and `F_ACGD=2*T+2*sum(k)`.
https://arxiv.org/html/2204.11144v2#S3.SS1

This is a nominal accounting convention. Cached JVP/VJP maps, fused Flax calls,
Schur vs full-block systems, restart residuals, AD evaluation, batched masking,
validation and compilation are not equivalent operations. Report wall time,
accuracy, dtype, tolerance, samples and memory separately. Do not label the
counter as measured backward/forward calls or physical GPU kernel launches.

The reference cgds-package GMRES currently returns the zero-based loop index
`j` and forwards it to `num_iter`. An actually executed first iteration must
count as 1, not 0. This does not establish how published plots were generated.
https://github.com/devzhk/cgds-package/blob/master/src/CGDs/gmres_torch.py

## Solver API (legacy API retained)

```python
from jax_cgd import solvers
solver = solvers.CG(tol=1e-8, atol=0., max_iter=200)
x, stats = solver.solve_with_stats(A, b, x0)
# stats.iterations, matvecs, restart_cycles, residual_norm, rhs_norm,
#       converged, breakdown
```

`solve()` retains JAX native return semantics, NOT iteration counts.
`solve_with_stats()` accepts real, nonempty, 1D floating vectors; the optimizers
already flatten their parameters into this format. It supports `jit`, `vmap`
and implicit differentiation through `custom_linear_solve(has_aux=True)`.
No private JAX internals or callbacks are used in production.

CG `max_iter` caps inner steps. GMRES `max_iter` retains the wrapper's meaning:
**restart cycles**; `restart` limits the number of inner steps per cycle. Total
GMRES work is the actual Arnoldi count, not `restart * cycles`. The counted
GMRES uses incremental Givens rotations and two-pass Gram-Schmidt rather than
JAX's legacy default batched least-squares mode. Keep this algorithmic
variation in mind in performance comparisons. No preconditioner is supplied.

Counters include attempted breakdown steps. `matvecs` includes the initial
warm-start product and true-residual verification. CG verifies the true final
residual; GMRES does so at every restart. A non-converged solve is reported,
not silently called successful. The optimizer retains the previous behavior
of applying inexact updates; inspect `failed_solves_total` in scientific runs.
Gradients require both primal and adjoint solves to converge sufficiently.

## Optimizer API and performance

```python
from jax_cgd import ACGD
opt = ACGD(x, y, game, solver=solver)
opt.step()
stats = opt.get_stats()      # device arrays; None before the first step
opt.step_many(100)           # compiled chunk, no Python dispatch per update
stats = opt.get_stats()
```

Existing `get_infos()` and functional `step_func()` return signatures remain.
`step_many()` keeps only the final state, not a trajectory. Different chunk
lengths may compile separately. Read stats only when logging; they remain on
the device in the inner loop. Custom solvers exposing only `solve()` remain
accepted, but their counts and cumulative failure count are -1 (unknown).

Fields: `outer_steps`, `krylov_iterations`, `krylov_iterations_total`, `matvecs`,
`matvecs_total`, `restart_cycles`, `residual_norm`, `relative_residual`,
`converged`, `breakdown`, `failed_solves_total`, `counts_available`,
`mixed_products_total`, `paper_proxy_total`, `paper_formula_applicable`,
`factorized_game`. A zero RHS uses absolute residual in `relative_residual`.

For Schur solves each matrix product uses two mixed-Hessian maps; RHS and
reconstruction add two maps per optimizer step:
`mixed_products_total = 2*matvecs_total + 2*outer_steps`.
For full-block solves it is `2*matvecs_total`. These are logical map calls;
a `vmap` batch may physically execute masked work for converged systems.

The default remains the SPD Schur solve. A **Schur iteration is not a
full-block GMRES iteration**. For a paper-style full system use:

```python
opt = ACGD(x, y, game,
           solver=solvers.GMRES(tol=1e-7, atol=1e-20, restart=20, max_iter=1000),
           formulation='block')
```

`paper_proxy_total=2*T+2*sum(k)` is always explicitly labeled a proxy.
`paper_formula_applicable=True` only identifies full-block GMRES, not exact
runtime equivalence or strict reproduction (warm starts, restarts, cached AD
and initialization can differ from the authors). CG is rejected for the
nonsymmetric full-block system. CGD/ACGD formulas, rates and default tolerances
are otherwise unchanged from the parent feature branch.

Performance changes: reuse gradients from the mixed-Hessian linearization
rather than evaluate them twice; run compiled multi-step chunks; optionally
factor a separable game `r(x).T d(y)` using `BilinearGame`:
`H_xy=J_r.T J_d`, `H_yx=J_d.T J_r`. This is exact, with no dense Jacobians.
Both feature maps may be nonlinear. Caching can increase peak memory.
The optimizer's inherited stop-gradient convention is retained: implicit AD
of the linear solver is supported, not a claim of full optimizer meta-gradients.

## Flax example and reproducible benchmarks

The original examples stay available as controls. The new runner uses the
same points, models, initialization and loss, with an exact optional factorization:

```bash
python -m examples.cpinn_counted --problem poisson --smoke
python -m examples.cpinn_counted --problem poisson --formulation block --solver gmres --generic --steps 100 --output results/block.json
python -m examples.cpinn_counted --problem poisson --mode adam --steps 100 --output results/adam.json
python -m examples.cpinn_counted --problem burgers --steps 100 --reference-mat burgers_shock.mat
```

JSON rows include loss, relative L2 (Burgers needs the reference), cumulative
counts, and separately timed training vs total wall time. They do not assert
paper accuracy or equal-budget superiority. Initial diagnostics count as zero
training work. Match the logged nominal budget, not just the update index.

`benchmarks/cpinn_optimizer.py` compares the actual pre-change source at
`b9320efa1d5ef66e88a2a87f215482671ccfcf82` to counted generic, counted factored,
and counted factored/chunked execution. Same inputs, float64, CG tolerance
`1e-8`, absolute tolerance 0, cap 200. Timed samples restore the same initial
state, alternate randomized order, synchronize, and separate first execution
from warmed timings. It asserts parameter agreement and zero reported failed
solves. CI uploads all timing samples' min/median/max and environment metadata.
These are short CPU benchmarks, not full-paper convergence or GPU memory tests.
