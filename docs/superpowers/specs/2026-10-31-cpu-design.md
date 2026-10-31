# CPU bottleneck reduction

User intent: improve the actual CPU optimizer and merge only if evidence is good.
Baseline: counted branch 78d2f7c and pre-counting branch b9320ef. Main currently
77059bb. Isolated workspace and branch perf/cpu-bottlenecks; no main writes until
acceptance.

Approaches: (1) share final Schur residual and reconstruction work without
weakening residual checks; (2) simplify/canonicalize autodiff graphs for exact
CPINN products; (3) exact forward derivative propagation for the example Flax MLP.
First profile; choose changes with measured payoff. Do not silently substitute
lower precision, relax solver tolerance, discard counting, or alter game/rates.

Acceptance proposed by implementer: full test suite and analytic derivative
regressions pass; same-data float64/tol=1e-8 comparisons on Poisson/Burgers;
>=15% geometric-mean improvement vs counted reference, no >5% steady-state
regression in targeted cases, repeated remote measurements. Record pre-counting
comparison too. Flag results too noisy to pass rather than claim success.
Full-scale training accuracy/GPU/general-sum/meta-gradients are out of scope.
