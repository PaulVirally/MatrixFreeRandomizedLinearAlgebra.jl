module MFRLACUDAExt

# GPU specializations. The generic code already runs on a `CuArray`. These only
# avoid copies, scalar indexing, and CUDA's missing in-place Hermitian eigvals.

using LinearAlgebra
using CUDA
import MatrixFreeRandomizedLinearAlgebra as MFRLA

MFRLA.materialize_mat(A::CuMatrix, ::CuArray) = A # FIXME: assumes a single GPU

# The R factor that comes out of `qrthin!` on the GPU is an `UpperTriangular`
# wrapping a `CuMatrix`. The generic `copyto!` reads the implicit zeros one
# element at a time (scalar indexing, which is slow on GPU), so we copy the
# parent and zero out the lower triangle with `triu!` instead.
function MFRLA.materialize_mat(A::UpperTriangular{T,<:CuMatrix}, prototype::AbstractArray) where {T}
    B = similar(prototype, T, size(A))
    copyto!(B, parent(A))
    triu!(B)
    return B
end

function MFRLA.qthin!(A::CuMatrix)
    m = size(A, 1)
    n = min(size(A)...)
    τ = similar(A, n)
    CUDA.CUSOLVER.geqrf!(A, τ)
    CUDA.CUSOLVER.orgqr!(A, τ) # writes the thin Q into A's first n columns
    m >= size(A, 2) && return A
    Q = similar(A, eltype(A), m, n) # wide: extract the m×m economy Q
    copyto!(Q, @view A[:, 1:n])
    return Q
end

function MFRLA.qrthin!(A::CuMatrix)
    m = size(A, 1)
    n = min(size(A)...)
    τ = similar(A, n)
    CUDA.CUSOLVER.geqrf!(A, τ)
    if m >= size(A, 2)
        R = UpperTriangular(CuArray(@view A[1:n, 1:n]))
        # orgqr! writes the thin Q into A's first n columns, so tall/square
        # inputs need no copy at all.
        CUDA.CUSOLVER.orgqr!(A, τ)
        return A, R
    end
    R = triu(A) # wide: full m×n upper-trapezoidal factor, copied before orgqr!
    CUDA.CUSOLVER.orgqr!(A, τ)
    Q = similar(A, eltype(A), m, n)
    copyto!(Q, @view A[:, 1:n])
    return Q, R
end

# CUDA.jl does not support eigvals! for these matrices.
MFRLA.eigvals_hermitian!(B::Hermitian{<:Any,<:CuMatrix}) = eigen(B).values

end # module MFRLACUDAExt
