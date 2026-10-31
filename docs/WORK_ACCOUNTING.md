# Work accounting (2.0.0rc1)

## Counters are not measured AD passes

CPINN arXiv:2204.11144v2 section 3.1 uses the nominal Forward Passes convention:
Adam T; ACGD 2*T+2*sum(k), where k is GMRES inner iterations.
https://arxiv.org/html/2204.11144v2#S3.SS1

`paper_proxy_total` is this nominal convention, not measured forward/backward
calls or GPU kernels. The RC charges attempted updates, including rejected work.
`outer_steps` counts accepted updates; `attempted_steps_total` includes rejection.
On successful runs the counts coincide. Schur-CG iterations are not full-block
GMRES iterations. `paper_formula_applicable` identifies full-block GMRES only;
caching, warm starts, restarts, batching and initialization can still differ.

## Linear solvers

Use `solver.solve_with_stats(A,b,x0)` for real Krylov counts and residuals.
`solve()` retains native JAX return semantics and does NOT return iteration counts.
Counted solves require real nonempty 1D floating arrays. CG max_iter caps inner
steps; GMRES max_iter caps restart cycles. Actual Arnoldi steps, not restart times
cycles, are reported. Production code uses no host callbacks; tests instrument
matvecs with debug.callback, so GPU test environments need cuda,cpu backends.

`matvecs_total` includes warm-start and true-residual products. Schur form has
`mixed_products_nominal_total = 2*(matvecs_total+attempted_steps_total)`;
full-block form has `2*matvecs_total`. Verified Schur-CG reuse subtracts
`mixed_products_reused_total` to obtain `mixed_products_total`. No residual check
is omitted. Batched masked execution can do more physical work than logical counts.

Custom solve-only providers get true-residual verification but unknown work
counts (-1). Their failures are still counted. Counts-enabled providers must
satisfy SolverStats. Gradients through a linear solve require sufficiently
converged primal/adjoint solves; whole-optimizer meta-gradients are not promised.

## Failure and state semantics

The RC rejects unsafe proposals by default. A rejected attempt stops its chunk;
accepted prefix stays committed. Parameters, RMS, warm start and accepted step
are protected, while attempted work/failure counters advance. `stop` returns
status; `accept_inexact` explicitly allows only finite, non-breakdown inexact
updates. Neither is a license to ignore diagnostics. See
[complete RC contracts](RELEASE_CANDIDATE.md) for pure update and checkpoint APIs.

Historical CPU results are pinned to their measured SHA in
[CPU_RESULTS_2026-10-31.md](CPU_RESULTS_2026-10-31.md). New safety checks add work;
no prior speedup automatically transfers to the RC or to GPU execution.
