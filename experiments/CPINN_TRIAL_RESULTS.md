# Measured CPINN convergence attempt: published 2.0.0rc1

## Verdict

The current implementation trains both examples, but these bounded experiments
have NOT reproduced the paper's final accuracy or ACGD work-efficiency advantage.
This is not evidence that the method can never reach those results. None of
the ACGD runs completed the paper-scale training horizon.

Package/examples remain unchanged from published source
`df67eb6c6309014b6c2752cfadd59a7483c181b2`. Only experiment infrastructure and
reports were added on this branch. Main and PyPI were not changed.

## Remote evidence and protocol

Run: https://github.com/Juicy-Quadro/jax-cgd/actions/runs/41226844497
Executed experiment commit: `2cc0e52bf9182ade9e311faca3761739ba447ace`.
Python3.12.15/JAX0.10.2/Flax0.12.8, CPU, float64. Source-integrity checks and
physical residual/jet tests passed; all four jobs completed. Green CI here
means execution finished, not that paper accuracy was attained.

Poisson:5000 interior+200 boundary points,3x50 tanh primal,4x50 ReLU dual;
separate256x256 evaluation grid. Burgers:10000 LHS+456 candidate PDE points,
100 selected IC/BC points,8x60 primal and independent-layer8x60 dual;
all25600 official reference values evaluated.
ACGD:lr1e-3,beta.99,eps1e-8,rtol1e-7,atol1e-20. CG:max1000 inner steps.
Block GMRES:restart20,max50 cycles. No failure accepted, no tolerance relaxed.
Each job had a600s wall budget checked between10-update chunks, with possible
chunk-boundary overshoot. Times include compile/diagnostics on separate CPU
hosts, not a controlled algorithm-runtime comparison.

| Case | Accepted updates | Final relative L2 | Best logged L2 (step) | Cumulative Krylov | Failed solves | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| Poisson ACGD/Schur-CG |620|0.002865590880|0.002151228687 (600)|16139|0|607.87|
| Poisson ACGD/block GMRES |480|0.005908532914|0.003969844687 (450)|32361|0|647.64|
| Poisson PINN/Adam |19020|0.001512858027|0.000179711925 (17460)|N/A|N/A|600.28|
| Burgers ACGD/Schur-CG |90|0.164522586672|0.164522586672 (90)|2437|0|721.26|

All four summaries say `budget_exhausted`; requested48000/28000 targets were
not completed. Best errors are minima of LOGGED EVALUATIONS every10 updates.
Only FINAL parameters/checkpoints are saved; best-error model weights were
not saved separately. The initial error was1.813105082409 for Poisson and
1.036094569918 for Burgers. Last linear relative residuals:6.961e-8,
9.945e-8 and6.816e-8 for Poisson-Schur,Poisson-block,Burgers respectively.

Burgers boundary MSE fell0.3362285 ->0.00359765, while PDE MSE rose
0.0146601 ->0.5837402. Its90-step L2 decrease is not a high-accuracy PDE solution.

## Work accounting

For the current block-GMRES proxy F=2T+2*sum(k), under a18000 cap the last
logged ACGD state hadT330,F17260,L2=0.01801238; Adam atT=F18000 had
L2=0.00163875. Under10000, ACGD F9278 gave0.04933734; Adam F10000 gave
0.00128645. These short-prefix checks do not show an ACGD advantage.
The budgets are near-matched logged points, not exactly equal. This is not
measured AD work or the identical historical author counter convention.
Schur counts are excluded from this block-budget comparison.

Mean Krylov work per update increased5.1 ->61.2 (first/last10 updates) for
Poisson-Schur, and10.1 ->239.1 for block GMRES. Early microbenchmark cost does
not describe late training; no condition number was measured or inferred as fact.

## Independent local CPU checks

Python3.13.5/JAX0.9.0.1/Flax0.12.0. All13 library Python files were checked
against the published wheel. Full-size Poisson-Schur completed680 updates
in620.62s: finalL2=0.001763939694,bestlogged0.001596080881 at670,zero failures.
Restoring final.npz reproduced its final error. An interrupted0..80 attempt
is retained in raw history, but analysis uses the subsequent0..680 sequence
only; work counters from separate attempts were not combined.

A separate300-update epsilon control:1e-6 gave finalL2=0.02590589, versus
0.02786916 at300 for1e-8. Neither reached high precision. The local raw
`complete_paper_training` label only checks the requested --steps, so its
true value for this300-step control is NOT completion of48000 paper steps.
Audited summaries distinguish these two meanings; original raw files remain.

## Artifact provenance (downloaded and SHA256 verified)

- Poisson-Schur:13052322193;29ce9110eb4ab54e854c66db70f57e6f03ab9b916c919003d0089da4c3dfbabf
- Poisson-block:13051764574;904225b5b790e0168656b3b4d3374ae4f12319452780e5ead732287015cc7882
- Poisson-Adam:13052227328;4d4868ec29b1a290b3bd5d3762013aa3f8d2e3848e93a5f6104b03e6c88f580b
- Burgers:13052172903;67c013744813fc175628cea1ec3c73e0e4d7803e07e58a0025225fb7cc327f7b
- burgers_shock.mat:f4d11e6798b8d29b6829984952b04adcc470d9bf70dd7e56c7985cd8e316c4bf

See CPINN_PROTOCOL_AUDIT.md for primary sources and discrepancies: appendix
versus scripts on epsilon/BC counts; Burgers discriminator shared weights;
initialization, sampling, time scaling, warm starts and GMRES restart.
Paper targets are Poisson1.7e-8 at48000 ACGD updates and roughly a few times
1e-5 for Burgers (Figure4 visual order, not an exact tabulated value).
These targets were not attained. Strict reproduction needs a frozen protocol,
identical exported initial parameters/data, verified one-step agreement and
full training trajectories. No GPU or general convergence claim is made.
