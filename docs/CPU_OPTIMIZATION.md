# CPU optimization: verified residual reuse and exact Flax derivatives

> **2.0.0rc1 update:** These optimization notes describe the pinned pre-RC implementation.
> Current failure handling, PyTree support and checkpoint contracts are in
> [RELEASE_CANDIDATE.md](RELEASE_CANDIDATE.md). Unsafe updates are now rejected by default;
> attempts (including rejected work) are counted separately from accepted updates.

## Scope

The CPU optimization has two separate parts. The library change applies to
built-in Schur-CG updates. The larger example-specific gain comes from the
Flax scalar tanh MLP spatial derivatives. Neither changes the game, training
points, initialization, per-parameter ACGD rates or requested tolerance.

## Library: share final verification and reconstruction

For S=sqrt(eta_x), the Schur operator is A(v)=v+S Hxy eta_y Hyx S v.
The final true-residual check and the max-player reconstruction both require
Hyx S solution. Compute that once and share it. Public solve_with_stats()
still verifies its residual independently; only the private optimizer path
defers it until reconstruction. This saves one mixed product per nonzero-CG
solve, not one Krylov iteration. Logical matvecs and actual mixed-product counts
are reported separately. Custom solver subclasses are not bypassed.

## Flax: exact spatial chain rule

For every Dense layer, multiply value, input gradient and selected Hessian
trace by the kernel (bias affects only value). For tanh, propagate using
f'=1-f^2 and f''=-2*f*(1-f^2). Poisson propagates the x+y Hessian trace;
Burgers propagates xx and both first derivatives, including the time input
rescaling. This avoids repeated nested spatial autodiff through the MLP.
Parameter gradients and mixed products still use JAX AD, with no finite
differences, stopped derivatives or Hessian approximation.

JetMLP inherits the original Flax forward method and Dense_i variable layout.
The reference cpinn_flax.py file is unchanged. The counted CLI selects the
jet path by default; --autodiff uses the reference derivatives. --generic
also disables the factored game. Adam still uses reference autodiff; these
benchmarks do not compare Adam versus ACGD runtime or accuracy.

The spatial jet supports only the example scalar tanh MLP with Nx2 inputs.
It rejects ReLU/multiple outputs/invalid axes. Equivalence is tested for
values, first/second derivatives, parameter gradients, JVPs, mixed products
and 10-step optimizer trajectories at multiple seeds. The example uses
float64; standalone counted solvers additionally have float32 tests.

## Reproduce acceptance benchmarks

```bash
mkdir -p /tmp/cgd-legacy /tmp/cgd-counted
git archive b9320efa1d5ef66e88a2a87f215482671ccfcf82 jax_cgd | tar -x -C /tmp/cgd-legacy
git archive 78d2f7c275ad155cc206f3b0261763e08f4a43bb jax_cgd | tar -x -C /tmp/cgd-counted
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m benchmarks.cpu_optimizer \
  --legacy /tmp/cgd-legacy --counted /tmp/cgd-counted \
  --problem poisson --n-pde 256 --n-bc 16 --width 50 --layers 3 \
  --steps 5 --repeats 11 --output results/cpu-poisson.json
```

The benchmark compares eight configurations, including both immutable
baselines, new generic core, new factored core without jets, and jets with
and without compiled chunks. All use float64, tol=1e-8, atol=0, max_iter=200,
identical samples and initial state restored outside every timer. Warm-up
is separated, variant order randomized, and execution synchronized. Every
sample checks finite parameter agreement and zero failed counted solves.
Raw samples and environment metadata are saved. First-chunk time includes
both compilation and execution, not pure compilation.

The preregistered merge gate is at least 15% geometric-mean speedup versus
the previous counted chunk, no >5% regression in targeted optimized CPINN
cases, repeated independent CPU measurements, and all correctness tests.
Generic-core results and pre-counting comparisons must also be reported;
example-specific speedups are not claimed for arbitrary objectives.

No long-run convergence, GPU speed, GPU memory, preconditioner, lower-precision
or relaxed-tolerance speedup is claimed. The RC now rejects unsafe updates
by default; this additional safety work needs its own performance measurement.
