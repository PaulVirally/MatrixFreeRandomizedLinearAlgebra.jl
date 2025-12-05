"""
    module MatrixFreeRandomizedLinearAlgebra

Tools for matrix-free randomized linear algebra supporting both CPU and GPU arrays.

This module provides:

    * [`rsvd`](@ref) / [`rsvdvals`](@ref): randomized SVD and singular values
    for general (possibly rectangular) operators, using power iteration and
    oversampling.
    * [`reigen_hermitian`](@ref) / [`reigvals_hermitian`](@ref): randomized
    eigen-decomposition and eigenvalues for Hermitian operators.

All routines are written to work with standard `AbstractMatrix` types (e.g., Matrix, CuMatrix), but also the types from [LinearMaps.jl](https://julialinearalgebra.github.io/LinearMaps.jl/stable/), or more generally any type supporting

  * size query `size(operator)` and
  * multiply `operator * X`
  * adjoint multiply `operator' * X`

You can optionally pass a `sample_vec` prototype to place all temporary work
arrays on a specific device (e.g. GPU vs CPU).
"""
module MatrixFreeRandomizedLinearAlgebra

include("common.jl")

include("rsvd.jl")
export rsvd, rsvdvals

include("reigen.jl")
export reigen_hermitian, reigvals_hermitian

end # module MatrixFreeRandomizedLinearAlgebra
