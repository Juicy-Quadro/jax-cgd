# CGD 2019: executed validation of jax-cgd 2.0.0rc1

## Verdict and scope

The unchanged published basic-CGD implementation passes analytical scalar trajectory checks and independent matrix-update checks. On the original deterministic covariance problem sizes and work budget, every tested native CGD case remained finite and reached the budget without a rejected linear solve. This reproduces the robustness trend, NOT exact equality of every long Julia/JAX trajectory. Neither the bimodal GAN distribution experiment nor MNIST was run; adaptive ACGD is not validated by these fixed-step experiments. No main merge or new PyPI release is part of this work.

## Frozen sources and actual execution

- Paper: Schafer and Anandkumar, Competitive Gradient Descent, arXiv:1905.12103v3.
- Author implementation: f-t-s/CGD@766fa08dc29f6e5f3170c781139fe44c42a16267 (Julia), linked by the paper.
- User-provided independent reproduction: julienroyd/competitive_gradient_descent@dda97e7c38a2418fc428dcffb32f83012daf81d1 (PyTorch). Its README explicitly links the original Julia implementation; MNIST is an extension by that repository, not a paper experiment.
- Tested package: exact published jax-cgd==2.0.0rc1 wheel, SHA256 4575d0fa535eff0f92a846f62952acfeaf4f0f92179e0078500af98888047719. Public optimizer source was not edited.
- JAX/Torch calculations: current chat CPU sandbox, Python 3.13.5, JAX 0.9.0.1, PyTorch 2.10.0+cpu, NumPy 2.3.5, float64. No GPU and no Flax network was needed for these scalar/matrix experiments.
- Original Julia covariance update bodies: GitHub Actions run 41290118089, successful; artifact 13076946524. Julia 1.10.12, Manifest-pinned LinearMaps 2.3.0 and IterativeSolvers 0.8.1, matching original dependency tree hashes. Old internal runtime version metadata is inconsistent (IterativeSolvers reports 0.8.0); the locked Manifest and source determine the tested version.
- Source capture: run 41289922221, artifact 13077250334.

The Julia runner loads the original optimizer function bodies unchanged, removing only the two unused historical AD imports and providing analytical derivative closures. These derivatives are independently checked against JAX AD. Thus this is a numerical-update reproduction, not a reconstruction of the entire 2019 Julia/AD runtime. Timings across these different environments/derivative implementations are not a speed benchmark.

## Scalar games (paper Figures 2/3)

Initial x=y=0.5, eta=0.2, alpha in {1,3,6}. The bilinear game f=alpha*x*y uses 50 updates. The convex-concave f=alpha*(x^2-y^2) and concave-convex negative of it use 100 updates. Each case runs through the released Schur-CG and block-GMRES paths, and is compared with an independent analytical recurrence and the unchanged third-party Torch CGD_step. Nine additional notebook-style bilinear settings (eta 0.04,0.2,1, all three alphas, 100 steps) also run on both paths: 36 full trajectories total.

All passed the predeclared normalized-trajectory error threshold 1e-11. Observed max normalized analytical error: 3.2315606445355706e-14; max normalized Torch difference: 2.048687502950853e-14; zero failed solves. Normalization is max absolute state error divided by max(1,max absolute reference state), checked at every step, not relative error in tiny final coordinates.

| alpha | CGD final radius after 50 bilinear updates | GDA final radius |
|---|---:|---:|
|1|0.26524763461079137|1.8850309475282234|
|3|0.0003243487980669366|1541.5503401890667|
|6|1.461299863418507e-10|3421611214.2125397|

For a=eta*alpha, exact CGD transition is [[1,-a],[a,1]]/(1+a^2), and r_n=r_0*(1+a^2)^(-n/2). The separable convex-concave transition factor is 1-2*eta*alpha: alpha6 at eta0.2 has magnitude1.4 and therefore diverges (radius2.8992693875526e14 at100steps). Native CGD correctly reproduces that divergence rather than reporting equilibrium convergence. The reversed quadratic likewise escapes the non-Nash stationary point. These are algorithmic negative controls, not solver failures. Basic CGD is not unconditionally stable with respect to arbitrary own-player curvature and step sizes.

## Full-size deterministic covariance experiments

The authority is the original cvest.jl executable configuration. Sigma=U*U^T; W0=B*B^T with B uniform(-0.5,0.5); V0=U+uniform(-0.5,0.5). Objective min_W max_V f(W,V)=sum(W*(V*V^T-Sigma)). The reported residual is

R=sqrt(||V*V^T-Sigma||_F^2+||sym(W)||_F^2).

Sizes d=20,40,60; eta=0.005,0.025,0.1,0.4; epsilon1e-6; original nominal work budget100000. The original source does not fix a seed. We chose Julia seed123, saved the actual U,W0,V0,Sigma arrays, and loaded those identical arrays in Python. These are controlled new realizations, not recovery of the paper's unpublished random samples.

The original alternates the eliminated player (V first, then W) while preserving both prior update directions. The native driver implements that scheduling through two public BCGD.update/state interfaces; it does not rewrite the package update. Nominal CGD work is 4+2*k per update, as in the author's source; this is not CPINN's 2+2*k and is not measured AD execution. Initial/verification and extra attempted work are recorded separately. An update that would exceed the nominal budget is omitted from the reported accepted endpoint, matching the reference accounting.

All twelve native cases reached the budget boundary without numerical rejection. Last accepted nominal work is <=100000; the next update may exceed it. The reference original CGD also remained finite in all12 cases. The other original update routines reached the budget with finite trajectories in OGDA4/12, SGA1/12, ConOpt1/12. Nonfinite or R>1e100 is a predeclared diagnostic divergence stop, rather than spending the remainder of the budget producing NaNs.

|d|eta|Julia CGD final R|Native published CGD final R|Native accepted updates|
|---|---|---:|---:|---:|
|20|0.005|9.854253598259522|2.4915456128880984|12498|
|20|0.025|0.5832364834289865|0.22397619637923719|12335|
|20|0.1|0.0007126257324103373|0.0014534365391850443|9613|
|20|0.4|4.546332297336348e-11|4.2560864761403385e-11|3898|
|40|0.005|5.220537662488887|6.480667710342626|12462|
|40|0.025|1.7649810680778097|1.8028136572484899|12281|
|40|0.1|0.35215015165698027|0.40849525818371113|8695|
|40|0.4|0.03892055898762276|0.13540930972337836|2755|
|60|0.005|10.865928745876179|10.266850135855282|12392|
|60|0.025|0.6888223182822505|0.8946031055770587|12204|
|60|0.1|0.23467007712387408|0.20727322674775508|6393|
|60|0.4|0.008414276176346182|0.004557954760844777|1812|

At eta0.4, initial R is38.8501,112.3524,216.8144. Native relative covariance fitting errors are respectively2.11256e-13,1.43368e-5,3.65981e-6. These must NOT be confused with the joint residual: in d40, ||sym(W)|| remains0.135313. Thus it is not correct to label every problem fully converged just because V*V^T fits Sigma well.

Three additional native fixed-player (ordinary single-BCGD-instance) eta0.4 runs also reached the budget without failed solves: d20 R6.84379e-11; d40 R0.0785811; d60 R0.00639614. These are supplemental checks, not the original alternating protocol.

## Why finite-tolerance native trajectories are not identical

The author code passes tol=epsilon*||b||/||r0|| to pinned IterativeSolvers0.8.1. That dependency's executable cg_iterator! computes the absolute target as ||b||*tol, despite a docstring referring to r0. The resulting historical target is epsilon*||b||^2/||r0||. Native jax-cgd uses epsilon*||b||. With a good warm start, the historical rule can allow zero iterations even when the native relative criterion is not satisfied.

We verified this with a RED-to-GREEN diagonal-system regression and a separate experimental AuthorToleranceCG adapter. The public package is not weakened. Twelve additional author-policy covariance runs reached the same nominal budget; some trajectories agree closely (e.g. d20,eta0.005 final R9.854253587735995 vs Julia9.854253598259522), but not all full trajectories become identical (d20,eta0.025 adapter0.697295 vs Julia0.583236). Therefore tolerance semantics are a concrete cause of protocol mismatch, not a proven sole explanation of every long-run difference. Float-operation ordering and discrete stopping choices remain possible contributors, not demonstrated unique causes.

The third-party notebook's method labelled OGDA is not the same implementation as the original prior-gradient OGDA; only its actual CGD_step is used as the scalar cross-check. Baseline covariance comparisons use original Julia routines.

## Independent mathematical correctness checks

Twelve regression tests passed: covariance objective gradients/mixed HVPs against analytical expressions; independent dense-block versus spectral solution; released block/Schur updates; historical tolerance behavior.

For the covariance game let C=V*V^T=Q*diag(lambda)*Q^T. Symmetric dx solves dx+2*eta^2*(dx*C+C*dx)=-eta*(gx+eta*Hxy*gy); in the eigenbasis each entry divides by1+2*eta^2*(lambda_i+lambda_j). This gives a non-Krylov reference. Small-dimensional checks also compare it with an explicitly assembled dense full-block solve, avoiding comparison of an implementation only with itself.

Twenty-four paper-size equal-state checks (three d, four eta, initial and saved original-Julia step1000 states) compare the released Schur update at tol1e-11 with this spectral solve. All pass the predeclared absolute next-parameter error threshold1e-8. Maximum observed error4.885343241056717e-10; zero failed solves. This strict diagnostic tolerance is separate from the1e-6 long-run reproduction setting. Final covariance residuals are independently recomputed from saved W,V,Sigma, not merely copied from training logs.

Final verification: scalar36; covariance native12; author-policy12; fixed-native3; spectral state-oracles24; 12 pytest checks passed. Full-size experiments were not truncated by a wall-time budget. Some exploratory aggregate tool calls timed out; final per-case reruns and file validation establish complete final outputs. Scalar archive filenames include step count so50-step paper runs are not overwritten by100-step supplemental runs.

## Limits / deliverables

This validates real zero-sum basic CGD, not arbitrary general-sum games, ACGD hyperparameter behavior, neural-network GAN sample quality, GPU correctness/performance or CPINN final accuracy. The original GAN uses a lagged RMSProp scheme different from current ACGD; directly replacing it would not be a strict reproduction. No GAN or MNIST success is claimed.

The evidence archive provided in the conversation contains the actual executable Python tests, Julia reference runner, frozen numerical fixtures, complete accepted trajectories, saved endpoints/checkpoints, environment locks, hashes and diagnostic logs. Original third-party source archives are not redistributed in that deliverable; immutable fetch instructions are included. The branch contains the original-source capture/reference workflows and this results report; all published package numerical files remain unchanged.
