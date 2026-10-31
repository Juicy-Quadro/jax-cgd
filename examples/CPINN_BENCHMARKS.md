# ICLR 2023 CPINN benchmarks (Flax/JAX)

Reference: Zeng et al., *Competitive Physics Informed Networks*, ICLR 2023,
https://arxiv.org/abs/2204.11144 ;
https://github.com/comp-physics/CPINN

Both examples use Flax Linen for **both** networks and JAX autodiff
for the PDE residual. The CPINN objective is an unsquared primal-dual
game, with two discriminator outputs for PDE and boundary residuals.

Poisson: `Delta u = -2 sin(x) cos(y)` on [-2,2]^2 with exact
`u = sin(x) cos(y)`; 5,000 LHS interior points and 50 samples on
each of four sides; generator tanh (3 hidden layers, width 50),
discriminator ReLU (4 hidden layers, width 50), 2 outputs.

Burgers: `u_t + u u_x - (0.01/pi) u_xx = 0`, x in [-1,1], t in [0,1],
IC `u(0,x)=-sin(pi x)`, BC `u(t,+/-1)=0`. Uses 10,000 LHS
interior points plus boundary points and N_u=100 selected initial/boundary
points; each network has 8 hidden layers of 60 neurons. Relative error
requires the **original paper's** `burgers_shock.mat` reference.

## Install and run

```bash
pip install -e .
pip install -r requirements-examples.txt
python examples/poisson.py --smoke
python examples/burgers.py --smoke
python examples/poisson.py --mode acgd --solver cg --steps 34001 --output results/poisson.json
python examples/burgers.py --mode acgd --solver cg --steps 28001 --reference-mat burgers_shock.mat
python examples/poisson.py --mode adam --steps 30000
```

For a closer algorithmic match to the authors' GMRES-based ACGD:
`--solver gmres --tol 1e-7`. The optimized JAX-CGD uses an SPD CG
variant by default; this is a **solver difference** from the paper.
The adaptive learning-rate implementation is also not the identical
PyTorch GACGD algorithm, and Flax has different initialization.
Do not treat identical benchmark equations as a quantitative
replication of the paper's final errors.

Use `--smoke` for tiny CPU checks only. Paper-scale runs are
very costly; reported relative L2 error is not attainable in a smoke test.
The Burgers test set is downloadable from:
https://github.com/comp-physics/CPINN/blob/main/Burger/burgers_shock.mat

Run output is JSON lines with PINN loss, elapsed time, and (when a
reference solution exists) relative L2 error. Time includes JIT
compilation and is **not** per-step steady-state timing. Benchmark
CPU/GPU peak memory and warm-up time separately.
