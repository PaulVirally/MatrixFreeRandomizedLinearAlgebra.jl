# MatrixFreeRandomizedLinearAlgebra.jl

## Overview

`MatrixFreeRandomizedLinearAlgebra.jl` is a Julia package that provides
efficient implementations of randomized algorithms for linear algebra tasks,
such as matrix approximations and singular value decompositions. The package is
designed to work with large-scale matrices in a matrix-free manner, meaning that
it does not require explicit storage of the entire matrix.

```@contents
Pages = ["api.md"]
```

## Installation

You can install `MatrixFreeRandomizedLinearAlgebra.jl` using Julia's package
manager:

```julia
] add MatrixFreeRandomizedLinearAlgebra
```

## Usage Example

Below is a simple example of how to use `MatrixFreeRandomizedLinearAlgebra.jl`
to compute a randomized SVD of a matrix:

```julia
using MatrixFreeRandomizedLinearAlgebra

A = randn(100, 50) # Some matrix we want to approximate
target_rank = 10
U, S, Vt = rsvd(A, target_rank) # Compute the randomized SVD
rel_norm = opnorm(A - U * Diagonal(S) * Vt) / opnorm(A) # Compute relative error
println("Relative error of the approximation: ", rel_norm)
```

### Estimating the trace

Use [`trace`](@ref) to estimate `tr(A)` of a square operator without forming it.
By default it uses XTrace, which is accurate for operators with some spectral
decay. For operators so large that even a small sketch will not fit in memory,
pass `low_mem=true` to fall back to the streaming Hutchinson estimator. The
sample budget can be a fixed count or a target relative error.

```julia
using MatrixFreeRandomizedLinearAlgebra

A = randn(500, 500)

t = trace(A, 50)                       # XTrace with 50 test vectors
t = trace(A; relative_tolerance=1e-2)  # XTrace, refined until the error is small
t = trace(A, 2000; low_mem=true)       # streaming Hutchinson, 2000 matvecs

# Ask for the estimated standard error alongside the value
res = trace(A; relative_tolerance=1e-2, return_error=true)
println("trace ≈ ", res.value, " ± ", res.error)
```

### Warm-starting with a pre-computed basis

Every decomposition routine accepts an optional `seed_Q` keyword: an (approximately)
orthonormal basis that already spans the range of the operator. Passing it
warm-starts the range finder instead of drawing a fresh random sketch, which is
handy when refining a previous solve or sweeping a parameter/time step. The seed
is re-orthonormalized internally and may have fewer columns than the sketch size
(it is padded with random columns), so a rough guess is fine.

```julia
using MatrixFreeRandomizedLinearAlgebra

A = randn(200, 200); A = A + A' # A Hermitian operator
δ = randn(200, 200); δ = 1e-3 * (δ + δ') # A small Hermitian perturbation
k = 10

E1 = reigen_hermitian(A, k)                         # First solve
E2 = reigen_hermitian(A + δ, k; seed_Q = E1.vectors) # Warm start from E1's basis
```
