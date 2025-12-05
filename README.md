[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://paulvirally.github.io/MatrixFreeRandomizedLinearAlgebra.jl/stable)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://paulvirally.github.io/MatrixFreeRandomizedLinearAlgebra.jl/dev)
[![Coverage Status](https://codecov.io/gh/PaulVirally/MatrixFreeRandomizedLinearAlgebra.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PaulVirally/MatrixFreeRandomizedLinearAlgebra.jl)

# MatrixFreeRandomizedLinearAlgebra.jl

`MatrixFreeRandomizedLinearAlgebra.jl` is a Julia package that provides
efficient implementations of randomized algorithms for linear algebra tasks,
such as matrix approximations and singular value decompositions. The package is
designed to work with large-scale matrices in a matrix-free manner, meaning that
it does not require explicit storage of the entire matrix.

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
