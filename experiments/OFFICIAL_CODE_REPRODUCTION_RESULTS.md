# Official-code CPINN reproduction: executed results (2026-11-01)

## Verdict

Full-data, independently evolved Poisson and Burgers trajectories support the correctness of the released ACGD core with a full-block solve in the tested CPU cases. This is not a completed reproduction of the paper's final accuracy. The official 34,001/28,001-update runs were not completed. The optimized Schur/warm-start/spatial-jet path does not maintain strict trajectory parity at the tested tolerance.

This branch changes experiment files only. Workflows checked that `jax_cgd/` and `examples/` are identical to the published RC source. No main merge or PyPI publication was performed by this experiment.

## Frozen inputs and environment

- jax-cgd 2.0.0rc1 numerical source: `df67eb6c6309014b6c2752cfadd59a7483c181b2`.
- Official CPINN: `ef0d6a4504e5a5fd69e3d0583f4c821b33436c2d`; actual `CGDs==0.4.5`, `pyDOE==0.3.8`.
- GitHub Actions CPU: Python 3.12.15, JAX 0.10.2, Flax 0.12.8, Torch 2.10.0+cpu, float64. No GPU or original historical Torch/CUDA runtime claim.
- Poisson: actual official 5,000 interior/200 boundary points; tanh P3x50, ReLU D4x50; original 256x256 evaluation grid.
- Burgers: 10,000 LHS points plus the full 456-point candidate pool, 100 supervised IC/BC points; P8x60, shared-middle-layer D8x60; unique parameter counts 25,861/3,962. Original MAT bounds including t.max=0.99, original primal input transform, raw discriminator inputs, and correctly mapped loss signs.
- lr=0.001, beta=0.99, eps=1e-8, rtol=1e-7, atol=1e-20. Full-block, cold-start, non-restarted GMRES for the historical path.
- Actual sampled arrays and Torch initialization tensors were transferred to Flax and archived, not merely matching integer seeds. After initialization, reference and JAX evolve independently with no parameter resynchronization.
- Original Burgers D initial RNG is incompletely specified. This controlled reference explicitly seeds123 and shares those tensors; it does not recover the authors' unpublished initial state.

## Explicit runtime adaptations

Original class/function AST definitions are loaded without executing top-level plotting/training scripts. Their numerical method bodies and CGDs optimizer are unchanged. Burgers' original default GMRES limit is 29,823, giving a preallocated H of about7.1GB. Its CPU zero storage is backed by a sparse mapped file, retaining dimensions, arithmetic and iteration limit. A control test matches original outputs/history exactly. One extra reference matvec measures the true residual without changing updates. These are not fair performance timing runs.

`OfficialCodeGMRES` is an isolated experimental JAX solver adapter, not a silent replacement of the public solver. It preserves one-pass MGS, Givens, projected strict-< stopping, cold starts, tiny-RHS return-b and the historical zero-based index. Workspace starts128; exhaustion retries from the same uncommitted state with larger storage rather than restarting. All reported historical-path runs had zero workspace retries. `accept_inexact` is explicit for historical compatibility, but observed true-residual failures and rejected updates were both zero. Nonfinite-path differences remain outside these successful trajectories.

## Completed full-data experiments

| Method | Updates | Final relative L2 | Actual Krylov steps | Stop reason |
|---|---:|---:|---:|---|
| Poisson official Torch reference |200|0.029105317427558144|4876|requested paired length|
| Poisson JAX ACGD + historical adapter |200|0.029105317427454622|4876|requested paired length|
| Poisson released native GMRES, cold |200|0.029105317427456794|4876|requested control length|
| Poisson released Schur-CG + warm start + jets |200|0.030331481214157748|3077|requested control length|
| Poisson JAX historical path extended |508|0.001626800226223883|23641|1000s wall budget|
| Burgers official Torch reference |58|0.5138471751909935|802|1000s wall budget|
| Burgers JAX ACGD + historical adapter |58|0.5138471751912103|802|1000s wall budget|

Poisson paired200: maximum parameter difference over recorded checkpoints9.789730831544219e-11; maximum recorded L2-curve discrepancy2.9499597209436956e-12. Both actual Arnoldi totals4876; historical reference index sum4676.

Burgers paired58: maximum recorded parameter difference1.650055092561331e-12; maximum L2-curve discrepancy4.67847982577041e-13. Actual Arnoldi totals802; historical index sum744. PDE MSE1.335936833174986 and BC MSE0.09600707712381386: this is clearly not yet an accurate PDE solution. Its matching reference trajectory, including fluctuations, is the implementation evidence.

Checkpoint maxima refer to step0,1,every10 and the final state, not all unrecorded intermediate states. The extended Poisson run pairs only its first20 updates, then continues JAX alone. Its saved Torch step20 must not be compared directly with JAX step508. JAX L2 at100/200/300/400/500/508 is approximately6.3905e-2/2.9105e-2/9.8138e-3/2.0227e-3/1.8291e-3/1.6268e-3. Official full lengths remain unfinished.

## Independent test of published native solvers

To avoid validating only a newly written compatibility solver, the unchanged public `solvers.GMRES` also ran200 updates on the identical full Poisson fixture, cold start with restart1000/max_iter1 (no restart encountered). Its final P/D maximum differences from official Torch were2.0363960517855162e-11/9.78504499649091e-11; L2 and actual iteration totals agree. Default failure protection remained enabled.

The unchanged Schur-CG/warm/jet variant completed200 updates with no solver failures, but final P/D maximum differences were0.009939853295012499/0.10672375188550211, relative parameter differences0.0065137392953347854/0.03093236819458903. L2 is3.033148% versus reference2.910532%, a relative increase of4.21285%, not4.2 percentage points. This optimized configuration must not be described as strictly reproducing the reference trajectory.

## Same-state step200 diagnostic

All following candidates start cold from one identical saved aligned state. Tightened tolerances are diagnostics, not modifications of the primary official-code runs.

| Solve | k | Reported relative residual | Original full-block relative residual | Increment relative difference from tight block |
|---|---:|---:|---:|---:|
| Native block1e-7 |48|9.1621e-8|9.1621e-8|8.6714e-8|
| Native Schur, generic1e-7 |40|4.2118e-8|1.7692e-7|1.6638e-7|
| Native Schur, jets1e-7 |40|4.1738e-8|1.7679e-7|1.6639e-7|
| Native block1e-12 |57|6.9326e-13|6.9369e-13|0|
| Native Schur, jets1e-12 |59|1.9327e-13|7.4497e-13|9.2381e-13|

Tight Schur/block maximum increment difference1.426636586643326e-14. Jet versus nested-autodiff gx/gy maximum differences8.33e-17/2.65e-17; mixed HVP differences1.25e-15/2.66e-15 on the tested vectors.

The concrete discrepancy is the stopping norm: with Dx=S^2 and Schur residual rS=bS-AS*z, the reconstructed block residual is, up to sign, [S*rS,0]. Therefore ||rS||/||bS|| is not ||S*rS||/||bBlock||. Identical numeric rtol does not certify identical full-block accuracy. This point confirms a real metric mismatch without evidence of a gross Schur algebra or spatial-derivative error at the tested state. It is not a complete long-trajectory ablation proving the sole cause of the4.2% error difference.

For further strict reference reproduction, use the tested full-block cold path. Evaluate the accelerated path separately, with full-block residual/accuracy control rather than assuming the same Schur tolerance gives the same trajectory.

## Verification and provenance

Downloaded all five original CI artifacts and checked their GitHub SHA256 digests. Independently re-evaluated archived weights with NumPy MLP forward and analytic chain-rule spatial derivatives, without JAX/Flax/Torch. Final L2 differs from logged values by about1e-16; PDE/BC MSE also agree. All Poisson data, evaluation arrays and initial weights match elementwise across jobs.

Final RMS maximum differences from Torch: Poisson vP4.11e-14/vD1.05e-15; Burgers vP6.65e-17/vD3.91e-16. The508-step saved reference state is correctly excluded from same-step moment comparison.

Read actual CI logs: unchanged released suite90 passed in114.98s; historical adapter/storage regression suite9 passed in6.41s. Repeated executions are not counted as distinct tests. All four workflow runs completed successfully; bounded-run success is not paper-accuracy success.

| Run | Contents | Artifact IDs |
|---|---|---|
|41286351085|Poisson extended, Burgers paired|13075467990 /13075483022|
|41286730023|Poisson paired200|13076195541|
|41286862640|Native solvers200 each,90 tests|13075702909|
|41288207593|Same-state diagnostics|13075819900|

The early local sandbox run was interrupted by a sandbox reset and is not the acceptance evidence; rerun cloud artifacts underpin this report. Inline review was performed; no independent reviewer agent was available.

## Limits

No completed34,001/28,001 run, no paper-level final accuracy, GPU, multiple seeds, or equal-work Adam comparison. No guarantee for arbitrary games. Numerical package source, public safe defaults, main and PyPI remain unchanged. Experimental scripts and original evidence are retained on `experiments/official-code-reproduction`.
