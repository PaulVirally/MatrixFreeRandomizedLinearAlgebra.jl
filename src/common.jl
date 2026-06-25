using CUDA
using SparseArrays
using LinearAlgebra
using LinearMaps

to_host(A) = Array(A) # bring a (small) reduced block to the CPU for dense work

# Get a concrete, dense, mutable matrix out of `A`, with the same eltype and size,
# living on the same device as `prototype` (a CPU `Vector` or a `CuVector`). The
# caller owns the result and may mutate it in place, so when `A` is already a
# dense matrix on the right device we hand it straight back.

# `A` is already exactly what we want, so there is nothing to do.
materialize_mat(A::Matrix, ::Array) = A
materialize_mat(A::CuMatrix, ::CuArray) = A # assumes a single GPU, which is all this package targets

# The R factor that comes out of `qrthin!` on the GPU is an `UpperTriangular`
# wrapping a `CuMatrix`. The generic `copyto!` below would read the implicit zeros
# one element at a time (scalar indexing, which is slow or disallowed on the GPU),
# so we copy the full parent and zero out the lower triangle with `triu!` instead.
function materialize_mat(A::UpperTriangular{T,<:CuMatrix}, prototype::AbstractArray) where {T}
    B = similar(prototype, T, size(A))
    copyto!(B, parent(A))
    triu!(B)
    return B
end

# Anything else that is already a matrix: an Adjoint, Transpose, Hermitian or
# UpperTriangular on the CPU, a sparse matrix, or a dense matrix sitting on the
# wrong device. We allocate on the prototype's device and copy into it. `copyto!`
# fills in the dense form for all of these.
function materialize_mat(A::AbstractMatrix, prototype::AbstractArray)
    B = similar(prototype, eltype(A), size(A))
    copyto!(B, A)
    return B
end

# A `LinearMap` is an `AbstractMatrix`, but `copyto!` into a dense array is not
# defined for it. Multiplying it by a dense matrix (`A * E`, below) would build a
# *lazy* `CompositeMap` rather than evaluating the product, so we use `mul!` to
# read all of its columns into a dense buffer at once. The identity and the
# output live on the prototype's device, so the product lands there too.
function materialize_mat(A::LinearMap, prototype::AbstractArray)
    T = eltype(A)
    m, n = size(A)
    E = similar(prototype, T, n, n)
    copyto!(E, Diagonal(fill!(similar(prototype, T, n), one(T)))) # dense identity, no scalar indexing
    B = similar(prototype, T, m, n)
    mul!(B, A, E) # eager: evaluates the operator columnwise into B
    return B
end

# A bare matrix-free operator that only knows its `size` and how to multiply. We
# materialize it with a single matrix product against the identity rather than one
# matrix-vector product per column.
materialize_mat(A, prototype::AbstractArray) = _materialize_via_matmul(A, prototype)

# Apply `A` to a dense identity to read off all of its columns at once. The
# identity is built on the prototype's device so the product lands there too.
function _materialize_via_matmul(A, prototype::AbstractArray)
    T = eltype(A)
    n = size(A, 2)
    d = fill!(similar(prototype, T, n), one(T))
    E = similar(prototype, T, n, n)
    copyto!(E, Diagonal(d)) # dense identity, built without scalar indexing
    return materialize_mat(A * E, prototype)
end
