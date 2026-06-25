using LinearAlgebra
using CUDA
using Random

# Compute the economy-size QR of A in-place and return only Q
function qthin!(A::AbstractMatrix)
    F = qr!(A)
    n = min(size(A)...)
    A .= F.Q[:, 1:n]
    return A
end
function qthin!(A::CuMatrix)
    n = min(size(A)...)
    τ = similar(A, n)
    CUDA.CUSOLVER.geqrf!(A, τ)
    CUDA.CUSOLVER.orgqr!(A, τ)
    return A
end

# Compute the economy-size QR of A in-place and return Q and R
function qrthin!(A::AbstractMatrix)
    F = qr!(A)
    n = min(size(A)...)
    Q = similar(A, eltype(A), size(A, 1), n)
    R = UpperTriangular(similar(A, eltype(A), n, n))
    Q .= F.Q[:, 1:n]
    R .= F.R[1:n, 1:n]
    return Q, R
end
function qrthin!(A::CuMatrix)
    n = min(size(A)...)
    τ = similar(A, n)
    CUDA.CUSOLVER.geqrf!(A, τ)
    R = UpperTriangular(CuArray(@view A[1:n, 1:n]))
    CUDA.CUSOLVER.orgqr!(A, τ)
    Q = similar(A, eltype(A), size(A, 1), n)
    Q .= A[:, 1:n]
    return Q, R
end

# Build an orthonormal sketch basis from a test matrix Ω: apply the operator,
# bring the product on-device, then thin-QR. With return_R, also return the
# triangular factor (needed by the SVD and XTrace reductions). The product is
# consumed by the in-place QR; callers that must retain it (adaptive XTrace)
# keep their own copy and call qrthin!/materialize_mat directly.
function sketch(operator, Ω, sample_vec; return_R::Bool=false)
    Y = materialize_mat(operator * Ω, sample_vec)
    return return_R ? qrthin!(Y) : qthin!(Y)
end

# Build the (orthonormalized) starting block for a randomized range finder.
# Without a seed, the block is `operator * Ω` for a Gaussian random `Ω`, exactly
# as before. When `seed_Q` is provided, its columns seed the block (a warm
# start); any shortfall up to `num_samples` columns is padded with `operator * Ω`
# so oversampling is preserved. The result lives on `sample_vec`'s device.
function range_finder_start(operator, num_samples::Int, sample_vec::AbstractArray, seed_Q)
    if seed_Q === nothing
        Ω = similar(sample_vec, eltype(operator), size(operator, 2), num_samples)
        randn!(Ω) # Generate Gaussian random matrix
        return sketch(operator, Ω, sample_vec)
    end
    size(seed_Q, 1) == size(operator, 1) || throw(DimensionMismatch("seed_Q has $(size(seed_Q, 1)) rows but the operator range has dimension $(size(operator, 1))"))
    s = size(seed_Q, 2)
    num_cols = max(num_samples, s) # Keep all seed columns even if there are more than requested
    Y = similar(sample_vec, eltype(operator), size(operator, 1), num_cols)
    copyto!(view(Y, :, 1:s), seed_Q) # Seed the leading columns (handles CPU↔GPU placement)
    if num_cols > s
        Ω = similar(sample_vec, eltype(operator), size(operator, 2), num_cols - s)
        randn!(Ω) # Pad with Gaussian random columns to preserve oversampling
        copyto!(view(Y, :, s+1:num_cols), materialize_mat(operator * Ω, sample_vec))
    end
    return qthin!(materialize_mat(Y, sample_vec))
end

# Find an orthonormal Q whose columns approximately span the range of `operator`,
# using randomized subspace iteration. For a self-adjoint (`hermitian=true`)
# operator each power iteration is a single multiply; otherwise it is an
# adjoint pass followed by a forward pass.
function randomized_range_finder(operator, num_samples::Int, num_power_iterations::Int, sample_vec::AbstractArray; hermitian::Bool=false, seed_Q=nothing)
    Q = range_finder_start(operator, num_samples, sample_vec, seed_Q)
    for _ in 1:num_power_iterations # Compute power iterations
        hermitian || (Q = sketch(operator', Q, sample_vec)) # extra adjoint pass for general operators
        Q = sketch(operator, Q, sample_vec)
    end
    return Q
end
