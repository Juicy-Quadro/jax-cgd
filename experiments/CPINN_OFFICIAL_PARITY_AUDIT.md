# Official CPINN / jax-cgd 2.0.0rc1 numerical audit

## Verdict and scope

No core ACGD sign, RMS, bias-correction or Schur-elimination error was found in the tested cases. The released Flax examples are NOT an exact reproduction of the public CPINN code: there are material differences in Burgers discriminator parameter sharing, input coordinates, mapped loss signs, data bounds, initialization, sampling and solver policy.

This is a source-and-short-trajectory audit, not a proof of all-input correctness or of paper-level convergence. The aligned Flax adapters used in this audit do not silently replace the package examples. Main and PyPI are unchanged.

## Frozen sources / actual execution

- CPINN: `comp-physics/CPINN@ef0d6a4504e5a5fd69e3d0583f4c821b33436c2d`.
- Official requirements pin `CGDs==0.4.5` and `pyDOE==0.3.8`. Downloaded/extracted the actual distributions, rather than assuming current master was the historical implementation.
- Audited RC source: `df67eb6c6309014b6c2752cfadd59a7483c181b2`, from its tested sdist.
- CPU sandbox: Python 3.13.5, JAX 0.9.0.1, Flax 0.12.0, PyTorch 2.10.0+cpu, float64. The historical PyTorch 1.9.1+cu102 runtime was NOT recreated; no GPU results are claimed.
- Loaded the original CPINN class/function AST definitions without executing whole training scripts. Numerical method bodies and the actual CGDs optimizer/GMRES are unchanged. Imported actual Torch tensors into Flax, not merely the same integer seed.
- The pinned gmres_acgd.py and gmres_torch.py equal the downloaded current-master versions after newline normalization. No optimizer-version drift was demonstrated here.
- Source download CI run: 41229275380, successful. Artifact 13052752353 contains the immutable official repository archive and pinned distributions. Numerical experiments below ran in the current chat CPU sandbox, NOT in that source-download job.

## Line-by-line execution-path map

Line numbers refer to frozen source files.

| Check | Official source | RC source | Finding |
|---|---|---|---|
| Player order / objective directions | CGDs/gmres_acgd.py:130-154,188-198 | jax_cgd/_optimizer.py:13-33,48-66 | Official handles two minimization objectives, D then P; JAX is minP/maxD of one f. Correct reordering agrees. |
| Step counter | gmres_acgd.py:121-122 | _optimizer.py:20,67 | Same t+1 convention. No old stuck-at-step-one bug in RC. |
| RMS | gmres_acgd.py:136-141 | _optimizer.py:19 | Same elementwise beta*v+(1-beta)*g^2. |
| Bias correction / epsilon | gmres_acgd.py:143-145 | _optimizer.py:20-22 | Same eta*sqrt(1-beta^t)/(sqrt(v)+eps). |
| First-moment momentum | gmres_acgd.py:130-149 | _optimizer.py:17-22 | Neither implementation has an Adam first-moment accumulator; its absence is not an RC bug. |
| Two base learning rates | gmres_acgd.py:115-116 | _optimizer.py:13,21-22 | Official allows unequal lr_x/lr_y; RC one lr. Both official PDE commands use equal 0.001. |
| Mixed HVPs | gmres_acgd.py:151-164; MvProd:14-53 | utils.py:116-130; games.py | Direct reference HVPs and cached/factored JAX HVPs agree in tests. |
| Full block / Schur update | gmres_acgd.py:14-53,147-168 | _optimizer.py:26-33,48-66 | Same exact block equations after player mapping; finite-tolerance residual criteria differ. |
| Cold / warm start | gmres_acgd.py:156-157 forces None | _optimizer.py uses previous | Protocol difference. |
| Restart / max_iter | gmres_torch.py:82-127 | solvers/gmres.py:8,20-22; _counted.py | Official unrestarted; RC defaults restart20 and max_iter restart cycles. |
| True residual / rejection | gmres_torch.py:112-127 | _counted.py; _optimizer.py:54-71,89-105 | RC adds checks and rejection. Previous reported trials had no rejected solves. |
| Poisson architecture | Poisson/train_1Dis_GACGD.py:70-84 | examples/cpinn_flax.py:128-136 | Standard P3x50/D4x50 agrees. |
| Discriminator initialization | Poisson/networks.py:232-252; Burger/BurgerTrainGACGD.py:268-272 | cpinn_flax.py:28-47,147-149 | Official D uses Linear defaults/random bias; current D Xavier/zero bias. |
| Sampling | Poisson/utils.py:30-49; Burger/BurgerTrainGACGD.py:18-47 | cpinn_flax.py:54-90 | Legacy NumPy+pyDOE versus default_rng+SciPy qmc, different consumption order and actual points. |
| Burgers time bound | Burger/BurgerTrainGACGD.py:20-31 and .mat | cpinn_flax.py:70-90 | Actual reference t.max=0.99 versus generated1.0. |
| Burgers P normalization | Burger/BurgerTrainGACGD.py:102-131 | cpinn_flax.py:39-40 | Reference2t/0.99-1 versus2t-1. |
| Burgers D inputs | Burger/BurgerTrainGACGD.py:274-278 | cpinn_flax.py:140-142 | Reference raw(x,t), current normalized time. |
| Burgers shared layers | Burger/BurgerTrainGACGD.py:268-272 | cpinn_flax.py:41-47 | List multiplication reuses one middle Linear, current layers independent. |
| Burgers loss directions | Burger/BurgerTrainGACGD.py:133-139,340-351 | cpinn_flax.py:157-165; cpinn_counted.py:41-56 | After mapping to minP/maxD, reference PDE channel has opposite sign unless the D output channel is reparameterized. |
| Burgers RNG order | Burger/BurgerTrainGACGD.py:291-297 and69-75 | cpinn_flax.py:147-149 | D constructed before torch seed set inside P; original D starting RNG not fully fixed by the script. |
| Evaluation grid | Poisson/utils.py:64-83 | cpinn_counted.py:116 | Original256x256, ordinary current entrypoint64x64; prior bounded trial explicitly used256x256. |
| Counter | gmres_torch.py:106,127 | krylov_iterations_total | Reference returns zero-based final index, RC actual steps. |

## Core algebra

Let x=P, y=D, H=d_x d_y f, and Dx,Dy be positive diagonal adaptive step sizes. Both implementations, after correct mapping, solve:

```
[I, Dx H; -Dy H^T, I] [dx;dy] = [-Dx gx; Dy gy].
```

Writing S=sqrt(Dx) gives the RC Schur system and reconstruction:

```
(I+S H Dy H^T S) z = S(gx+H Dy gy)
dx=-S z
dy=Dy(gy+H^T dx).
```

There is no sign error merely because the original code uses D/P order and subtracts both players' solutions. Signs must be compared after mapping their two losses.

## Actual executable comparisons

| Experiment | Size and steps | Max absolute parameter difference, JAX block vs reference | JAX Schur/factorized jets vs reference |
|---|---|---:|---:|
| Nonlinear scalar game | 5/4 parameters;20 independent updates |8.33e-17|8.84e-16|
| Poisson reduced data | P3x50,D4x50;32PDE,16BC;5updates |4.15e-13|4.03e-11|
| Poisson full data | P3x50,D4x50;5000PDE,200BC;5updates |8.91e-13|1.44e-10|
| Burgers reduced data | P8x60,D8x60(shared);488PDE,8BC;5updates |7.07e-12|7.53e-10|

Scalar-game reference also matches an independently assembled NumPy dense block solve to4.16e-17 per update; RMS-state differences at most1.30e-18. The scalar test uses tolerance1e-12. PDE tests use1e-10, cold starts and GMRES capacity80/one cycle, with all solves stopping before the restart boundary. These diagnostic settings are not the previous training settings.

For full-data Poisson, initial PDE residual difference is4.44e-16, P-gradient5.55e-17, D-gradient4.51e-17; both mixed-HVP maximum differences are below4e-18. The aligned exact spatial-jet features also match nested autodiff. The Burgers test deliberately reduces sampling: it is NOT a full10000-point reproduction.

## Concrete mismatches and negative controls

1. **Burgers shared D:** reference unique3962 parameters versus current25922 at8x60. A small width6/depth3 test copied the shared tensor into independent layers, preserving initial forward outputs and game value exactly; untying alone changed the first P update by max2.85e-5 (relative update difference0.283%).
2. **Burgers mapped loss:** original C=mean(Dbc*(g-u))+mean(Dpde*r), then step(C,-C) with D first. In minP/maxD form f=-C=mean(Dbc*(u-g))-mean(Dpde*r). Current example uses plus on PDE. Sign-only ablation changed the first P increment by max0.00200473 (relative1.87774) at the small fixture. Flipping BOTH kernel column and bias for the final D PDE output gives exactly matching scalar objectives. Thus this is a convention/initial-state mismatch, not evidence that the current PDE is wrong or that the sign alone explains the full accuracy gap.
3. **Input transforms:** actual reference .mat time max0.99, P time slope2.020202 versus2.0; D reference raw coordinates. On the small fixture, normalizing D alone changes its initial output by max0.06747; using P time endpoint1 instead of0.99 changes P output by max0.01233.
4. **Initialization:** official Poisson D has202 nonzero biases in the audit initialization; current0. Burgers122 versus0. Same seed across Torch and Flax cannot substitute for transferring actual tensors. Burgers reference D is not deterministically seeded before construction; the audit explicitly seeds it123, which is not a claim about the authors' final hidden initial state.
5. **Poisson:** no layer-sharing mismatch and its game sign agrees, but data points, initial tensors, cold/warm start, restarted/full GMRES and stopping budget differ.

These are single-state/short-trajectory controls, not a demonstrated ranking of causes of long-training error.

## Later-checkpoint inner-solve diagnostic

Restored the previous full-size Poisson checkpoint at accepted step620, with its actual current-example data. Compared the next update without retraining:

| CG tolerance | iterations | Schur relative residual | Reconstructed full-block relative residual |
|---|---:|---:|---:|
|1e-7|73|6.27e-8|8.92e-8|
|1e-12|106|4.10e-13|6.59e-13|

Maximum updated-parameter difference8.31e-11, relative increment difference6.98e-8. At this specific saved state there is no evidence of a gross block-equation violation concealed by the Schur residual. This does not bound every training state.

## Upstream GMRES behaviors: do not copy blindly

- A=2I,b=[1,2,3]: one actual Arnoldi iteration, original reported num_iter0.
- A=2I,b=[1e-10,2e-10,3e-10]: original shortcut returns b with zero matvecs and true relative residual1, not b/2.

Keep RC true counters/residual protection. For historical comparisons record a separate reference-index axis. The final author plotting pipeline was not available, so this alone does not establish how the published figure budgets were computed.

## Paper text versus public code

Appendix uses ACGD epsilon1e-6; both public training scripts explicitly pass1e-8, as do current examples. Burgers appendix describes100IC plus100perboundary, while public script samples100 total from456 candidates. Public Poisson script has34001 updates versus paper's48000. These need separately named official-code and paper-text protocols, not silent mixtures.

## What remains

The test harness now provides an aligned Flax reference path, but the released examples have NOT been changed. No new PyPI upload or main merge. No full-paper accuracy, full GPU parity or arbitrary-objective convergence guarantee. Next implementation should add an explicit official-code protocol, export/import frozen data and initial weights, and keep true work counters separate from historical ones; then compare hundreds of steps and full-length training before asserting paper-level reproduction.

Detailed Chinese line-map, exact scripts, six numerical JSON outputs, logs, source archives and SHA256 manifest are delivered in the conversation evidence ZIP. See CPINN_OFFICIAL_PARITY_RESULTS.json for machine-readable summaries.
