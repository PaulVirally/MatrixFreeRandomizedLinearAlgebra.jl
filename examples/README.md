# Examples

Runnable scripts that run the package on a matrix-free 2D Gaussian blur operator
and compare the randomized results against dense references. Each script writes
its figures into `../docs/src/assets/`, where the README and the documentation
pick them up.

The blur operator (and a `densify` helper for the reference computations) lives
in [`operators.jl`](operators.jl). Blurring is a convolution, which the FFT
diagonalizes, so the operator is applied with a couple of FFTs and the implied
`n² × n²` matrix is never stored.

## Running

The scripts use their own environment. Instantiate it once:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

then run any of them from the repository root:

```sh
julia --project=examples examples/01_rsvd_blur.jl          # randomized SVD vs dense svd
julia --project=examples examples/02_reigen_hermitian.jl   # Hermitian eigenvalues vs dense eigen
julia --project=examples examples/03_trace.jl              # XTrace vs Hutchinson convergence
julia --project=examples examples/04_performance.jl        # runtime: dense vs matrix-free
```

Each script prints the accuracy it achieves against the dense reference and saves
its PNG(s) to `docs/src/assets/`.
