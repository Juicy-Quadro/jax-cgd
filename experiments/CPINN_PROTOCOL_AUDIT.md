# CPINN protocol audit and bounded RC trials

These experiments test the unchanged `2.0.0rc1` library from source
`df67eb6c6309014b6c2752cfadd59a7483c181b2`. They are not a claim of reproducing
final paper accuracy. No changes to the published package or main are made.

## Three separate acceptance questions

1. Does the implementation execute and satisfy its linear-solve tolerances?
2. Does its independently evaluated PDE solution error decrease in real training?
3. Does a specified protocol reach the paper's final accuracy and matched work budget?

Passing 1 or a short test of 2 does not establish 3.

## Sources

- Paper: https://arxiv.org/abs/2204.11144 (v2; sections 3.1, 3.2, 3.4, appendix A).
- Author Poisson driver: https://github.com/comp-physics/CPINN/blob/main/Poisson/train_1Dis_GACGD.py
- Author Poisson networks: https://github.com/comp-physics/CPINN/blob/main/Poisson/networks.py
- Author Burgers driver: https://github.com/comp-physics/CPINN/blob/main/Burger/BurgerTrainGACGD.py
- Referenced GACGD: https://github.com/devzhk/cgds-package/blob/master/src/CGDs/gmres_acgd.py
- Referenced GMRES: https://github.com/devzhk/cgds-package/blob/master/src/CGDs/gmres_torch.py

The current public author code is evidence of its current implementation,
not proof that exactly that revision produced every figure in the paper.

## Published targets and configuration differences

- Poisson: paper reports relative L2 1.7e-8 at 48000 ACGD updates. The public
  driver loops for 34001 updates. Both use 5000 interior points and 50 per edge,
  a 3x50 tanh primal network and 4x50 ReLU discriminator.
- Burgers: Figure 4 reaches a few times 1e-5 (approximate visual reading, not
  an exact tabulated final value). Both paper-described networks are 8x60.
- Appendix sets ACGD epsilon=1e-6, while both inspected public drivers set
  epsilon=1e-8. Our unchanged example baseline uses 1e-8; a separately named
  local 300-update control uses 1e-6. Adam uses 1e-8 in both accounts.
- Burgers appendix describes 100 IC points plus 100 per boundary (300 total).
  The public driver and our example select 100 total from a 456-point pool,
  and append all 456 candidates to the 10000 LHS PDE points.
- The supplied Burgers reference has x.shape=(256,1), t.shape=(100,1),
  usol.shape=(256,100), x in [-1,1], t in [0,0.99]. The public driver derives
  its sampling/scaling bounds from that grid; our example uses t in [0,1].
- The author's Burgers discriminator builds repeated hidden blocks by Python
  list multiplication, sharing one hidden Linear object. Recreating that
  construction for 8x60 yields 9 visible Linear applications but 3 distinct
  Linear objects and 3962 trainable parameters. Our independent-layer
  discriminator has 25922 parameters. Its coordinate scaling and initialization
  also differ. Do not silently call these byte-identical architectures.
- The author's GACGD sets prev_x0=None each step and uses unrestarted block
  GMRES. Our fast default is warm-started Schur-CG. The block control here uses
  restarted GMRES (restart20, max50 cycles), so it is not the identical solver.
- The public gmres_torch returns zero-based loop index j as num_iter; our
  solver reports the number of completed iterations. The paper's nominal
  formula 2*T+2*sum(k) therefore needs a pinned definition/version before
  reproducing historical work-axis values. Schur and block counts are not
  interchangeable network-pass measurements.

## Trial protocol

Float64, lr=1e-3, beta=.99, epsilon=1e-8, solver rtol=1e-7/atol=1e-20.
Keep full current-example point counts and layer widths; no tiny-network
substitution. Poisson evaluation uses a separate 256x256 regular grid;
Burgers evaluates all 25600 original reference values. Record PDE and BC MSE,
relative L2, accepted/attempted steps, true residual and actual Krylov counts.
Stop on the first rejected solve, retaining the accepted prefix and diagnostics.

Each remote trial has a declared 600-second budget checked at chunk boundaries
(10 updates per chunk), with a 900-second process safety limit. Time includes
compilation and diagnostics. `budget_exhausted` is not full training completion
or proof of an error floor. Independent CPU runners are not a controlled
wall-time comparison between algorithms. Different update counts are not an
equal-work comparison either. Raw logs/checkpoints are retained as artifacts.

The first workflow attempt stopped before training because a shallow checkout
could not resolve the frozen ancestor used in the source-integrity check.
The follow-up uses full history; no numerical tolerance or optimizer was changed.
