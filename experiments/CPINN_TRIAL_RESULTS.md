# Measured CPINN convergence attempt: published 2.0.0rc1

## Verdict

The current implementation trains the examples, but **these experiments have
not reproduced the paper's final accuracy or its ACGD work-efficiency advantage**.
This is a bounded attempt, not evidence that the method can never reach those
results. No experiment completed the paper-scale ACGD training horizon.

The package and existing examples are unchanged from published RC source
`df67eb6c6309014b6c2752cfadd59a7483c181b2`. This branch adds only experiment
infrastructure and reports; main and PyPI were not changed.

## Remote evidence

Workflow: https://github.com/Juicy-Quadro/jax-cgd/actions/runs/41226844497
Executed experiment commit: `2cc0e52bf9182ade9e311faca3761739ba447ace`.
Python3.12.15/JAX0.10.2/Flax0.12.8, CPU, float64. The workflow checked that
`jax_cgd/` and `examples/` exactly match the frozen RC ancestor. All four jobs
finished and preserved raw history, metadata, parameters/checkpoints and logs.
A green job means execution completed, NOT that paper precision was achieved.

Full existing-example sizes: Poisson5000 interior+200 boundary, primal3x50,
dual4x50; Burgers10000 LHS+456 candidate PDE points,100 selected IC/BC points,
primal8x60 and independent-layer dual8x60. Poisson errors use a separate
256x256 regular grid; Burgers errors use all25600 official reference values.
Both ACGD settings use lr1e-3,beta.99,eps1e-8,rtol1e-7,atol1e-20.
Schur-CG has max1000 inner steps. Block GMRES has restart20,max50 cycles.
No failure was accepted and no tolerance was relaxed.

Each job had a600s budget checked between10-update chunks, so final elapsed
wall time can exceed600s. Time includes compilation and diagnostics and comes
from separate CPU hosts, not a controlled runtime benchmark.

| Case | Accepted updates | Final relative L2 | Best logged L2 (step) | Cumulative Krylov | Failed solves | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| Poisson ACGD/Schur-CG |620|0.002865590880|0.002151228687 (600)|16139|0|607.87|
| Poisson ACGD/block GMRES |480|0.005908532914|0.003969844687 (450)|32361|0|647.64|
| Poisson PINN/Adam |19020|0.001512858027|0.000179711925 (17460)|N/A|N/A|600.28|
| Burgers ACGD/Schur-CG |90|0.164522586672|0.164522586672 (90)|2437|0|721.26|

All four summaries say `budget_exhausted`; requested48000/28000 update targets
were not completed. The best values are minima of saved checkpoints every10
updates, not selected outputs from a completed paper run.

Initial relative L2 was1.813105082409 for all Poisson controls and1.036094569918
for Burgers. Last relative linear residuals were6.961e-8,9.945e-8 and6.816e-8
for Poisson-Schur,Poisson-block,Burgers respectively.

Burgers requires caution: its boundary MSE fell0.3362285 ->0.00359765, but its
PDE MSE rose0.0146601 ->0.5837402. The90-step L2 decrease does not establish
an accurate PDE solution or monotonic residual convergence.

## Work and accuracy, not just outer-step counts

Using the current block-GMRES nominal formula F=2T+2*sum(k), the last logged
ACGD state below a18000 budget was atT330,F17260 with L2=0.01801238; Adam at
T=F18000 had L2=0.00163875. At the10000 cap, ACGD usedF9278 and reached
0.04933734; Adam used10000 and reached0.00128645. These short-prefix checks
DO NOT show an ACGD advantage. Counts are not measured AD passes and are not
identical to the author's historical index conventions; see protocol audit.
Schur-CG counts are deliberately excluded from this nominal block comparison.

In Poisson-Schur, mean Krylov work increased from5.1 per update in the first10
updates to61.2 in the last10. In block GMRES it increased10.1 ->239.1.
Thus the cost of an early5-update microbenchmark does not describe later
training cost. No measured condition number is claimed from counts alone.

## Separate local CPU cross-checks

Python3.13.5/JAX0.9.0.1/Flax0.12.0, same source verified against all13 Python
files in the published wheel. Full-size Poisson-Schur reached680 updates in
620.62s: final L2=0.001763939694, best logged=0.001596080881 at670, zero failed
solves. Restoring final.npz reproduced the final L2 to floating-point precision.
An earlier interrupted80-update attempt was kept in raw history; analysis
uses only the second uninterrupted0..680 sequence, not combined work counts.

A separate300-update epsilon control used the appendix's1e-6 without other
changes: final L2=0.02590589 versus0.02786916 at300 with1e-8. Neither is a
high-accuracy reproduction. The local harness's raw `complete_paper_training`
field merely compared to requested --steps; for this300-step control that
label is misleading. Audited summaries explicitly distinguish completing the
requested control from completing the48000-step paper horizon.

## Raw artifacts (downloaded and SHA256 checked)

- Poisson-Schur:13052322193, `29ce9110eb4ab54e854c66db70f57e6f03ab9b916c919003d0089da4c3dfbabf`
- Poisson-block:13051764574, `904225b5b790e0168656b3b4d3374ae4f12319452780e5ead732287015cc7882`
- Poisson-Adam:13052227328, `4d4868ec29b1a290b3bd5d3762013aa3f8d2e3848e93a5f6104b03e6c88f580b`
- Burgers-Schur:13052172903, `67c013744813fc175628cea1ec3c73e0e4d7803e07e58a0025225fb7cc327f7b`
- Official burgers_shock.mat SHA256: `f4d11e6798b8d29b6829984952b04adcc470d9bf70dd7e56c7985cd8e316c4bf`

## Interpretation and next gate

Paper targets: Poisson1.7e-8 after48000 ACGD updates; Burgers Figure4 reaches
roughly a few times1e-5 (visual order only). These are not attained here.
See CPINN_PROTOCOL_AUDIT.md: public paper and scripts differ on epsilon and
Burgers boundary counts, and the current Flax discriminator differs from
the public Burgers script's shared-layer construction. Initialization, sample
sequences, coordinate scaling, warm starts and GMRES restart also differ.

A strict next reproduction should freeze an explicit paper-vs-author-code
protocol, compare one-step updates from identical exported parameters/data,
and then run complete training with saved error and work histories. One seed,
short CPU prefixes and a successful solver do not certify final accuracy,
arbitrary-task convergence, GPU performance or an ACGD-vs-Adam advantage.
