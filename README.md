# jax-cgd

Competitive gradient descent (BCGD) and adaptive CGD for real differentiable
zero-sum games, with matrix-free JAX linear solves and Flax PINN examples.
Original CGD: https://arxiv.org/abs/1905.12103 ; ACGD: https://arxiv.org/abs/1910.05852 .

## Install current source

```bash
git clone https://github.com/Juicy-Quadro/jax-cgd.git
cd jax-cgd
python -m pip install -e .
# Optional Flax examples and tests:
python -m pip install -r requirements-examples.txt
JAX_ENABLE_X64=1 python -m pytest -q
```

`pip install jax-cgd` installs the published PyPI package; this work does not
publish a new PyPI release. Use the Git source for the counted/optimized version.

## Use and diagnostics

```python
from jax_cgd import ACGD, solvers

# x, y: flat dictionaries of real floating arrays; game(x, y): scalar.
opt = ACGD(x, y, game, lr=1e-3,
           solver=solvers.CG(tol=1e-8, atol=0., max_iter=200))
opt.step_many(20)  # compiled chunk; no trajectory is retained
stats = opt.get_stats()  # JAX device arrays, None before the first update
# Check stats["failed_solves_total"] before trusting the run.
```

For standalone linear systems, use `solver.solve_with_stats(A, b, x0)` for
real iteration counts; `solve()` retains the native JAX return semantics.
See [work accounting and API limits](docs/WORK_ACCOUNTING.md).

## CPU-optimized Flax CPINNs

```bash
python -m examples.cpinn_counted --problem poisson --smoke
python -m examples.cpinn_counted --problem burgers --smoke
# Reference nested-autodiff derivatives:
python -m examples.cpinn_counted --problem poisson --smoke --autodiff
```

Both networks use Flax Linen. The optimized spatial derivative path is exact
chain-rule propagation for the example scalar tanh MLP, not an approximation
and not a universal acceleration for arbitrary user networks. Original
`examples/poisson.py` and `examples/burgers.py` remain reference entrypoints.
See [CPU design and benchmark protocol](docs/CPU_OPTIMIZATION.md).

## Compatibility notes vs original main

- The step counter now persists; ACGD uses per-parameter RMS scaling rather
  than the old global norm. These correctness fixes change training trajectories.
- Default optimizers use SPD Schur-CG at tolerance 1e-6. Scientific runs should
  explicitly set precision/tolerance, as above. Full-block form requires GMRES.
- No promise of whole-optimizer meta-gradients. Linear-solver implicit AD is
  separately tested; inherited optimizer stop-gradient behavior is retained.
- Flat dictionaries are supported; Flax examples flatten/unflatten their
  nested variables explicitly. Failed solves are reported, not skipped.
- GPU performance, peak memory and full-paper accuracy are not established by
  the short CPU acceptance benchmarks.
