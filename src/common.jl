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
# defined for it. We read off its columns by applying it to the columns of the
# identity, one at a time. We deliberately do this column by column with a *vector*
# right-hand side rather than a single block `mul!(B, A, E)` against a dense
# identity: for a nested `CompositeMap` the block path flattens the composite and,
# for BLAS-3 efficiency, densifies an intermediate sub-chain via
# `convert(AbstractArray, ::LinearMap)`, which allocates a *host* `Matrix` and
# probes sub-maps with *host* vectors. When a sub-map wraps a `CuArray` that probe
# dispatches CPU BLAS on a device array and throws. The vector path
# (`_compositemul!`) instead allocates its intermediates with `similar(x)`, so it
# stays on the prototype's device. The single device column `e` is reused across
# columns to keep the extra allocation at O(n) (no dense `n×n` identity).
function materialize_mat(A::LinearMap, prototype::AbstractArray)
    T = eltype(A)
    m, n = size(A)
    B = similar(prototype, T, m, n)
    e = fill!(similar(prototype, T, n), zero(T)) # reused unit-vector RHS, on the prototype's device
    @views for j in 1:n
        e[j:j] .= one(T)    # e = eⱼ via a 1-element broadcast (a kernel, not scalar indexing)
        mul!(B[:, j], A, e) # vector RHS: device-only composite path, no host densification
        e[j:j] .= zero(T)   # reset for the next column
    end
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
