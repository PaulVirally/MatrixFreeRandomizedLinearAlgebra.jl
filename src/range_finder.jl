using LinearAlgebra
using CUDA
using Random

# Compute the economy-size QR of A and return only Q. For tall/square inputs Q
# is built in place and `A` itself is returned; a wide A (more columns than
# rows) has an m×m economy Q that cannot fit A's shape, so it comes back as a
# new array on the same device.
function qthin!(A::AbstractMatrix)
    F = qr!(A)
    m, k = size(A)
    n = min(m, k)
    Q = k == n ? A : similar(A, eltype(A), m, n)
    Q .= F.Q[:, 1:n]
    return Q
end
# BLAS-eltype host matrices: build the thin Q in place with raw LAPACK calls
# (geqrf! then orgqr!), instead of materializing `F.Q[:, 1:n]` into a full-size
# temporary and copying it back. For wide inputs the reflectors all live in the
# first m columns, so the economy Q is generated in a copy of that block.
function qthin!(A::Matrix{T}) where {T<:LinearAlgebra.BlasFloat}
    m, n = size(A)
    A, τ = LinearAlgebra.LAPACK.geqrf!(A)
    m >= n || (A = A[:, 1:m]) # wide: the m×m economy Q cannot fit A's shape
    LinearAlgebra.LAPACK.orgqr!(A, τ)
    return A
end
function qthin!(A::CuMatrix)
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

# Compute the economy-size QR of A in-place and return Q and R. For tall/square
# inputs R is a square UpperTriangular; for wide inputs the full m×k
# upper-trapezoidal factor is returned as a plain dense matrix (it has no
# triangular wrapper), so that Q * R always reconstructs A.
function qrthin!(A::AbstractMatrix)
    F = qr!(A)
    m, k = size(A)
    n = min(m, k)
    Q = similar(A, eltype(A), m, n)
    Q .= F.Q[:, 1:n]
    if k == n # tall/square: R is square upper triangular
        R = UpperTriangular(similar(A, eltype(A), n, n))
        R .= F.R[1:n, 1:n]
        return Q, R
    end
    R = similar(A, eltype(A), n, k) # wide: full upper-trapezoidal factor
    R .= F.R
    return Q, R
end
# BLAS-eltype host matrices, tall/square: copy the n×n R block out of the
# factored A (triu! clears the reflectors stored below the diagonal), then turn
# A itself into the thin Q in place. The caller gets Q === A, so the only new
# allocation is the small R block rather than a second full-size matrix.
function qrthin!(A::Matrix{T}) where {T<:LinearAlgebra.BlasFloat}
    m, n = size(A)
    A, τ = LinearAlgebra.LAPACK.geqrf!(A)
    if m >= n
        R = UpperTriangular(triu!(A[1:n, 1:n]))
        LinearAlgebra.LAPACK.orgqr!(A, τ)
        return A, R
    end
    R = triu!(copy(A)) # wide: full m×n upper-trapezoidal factor
    Q = A[:, 1:m] # the reflectors all live in the first m columns
    LinearAlgebra.LAPACK.orgqr!(Q, τ)
    return Q, R
end
function qrthin!(A::CuMatrix)
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

# Build an orthonormal sketch basis from a test matrix Ω: apply the operator,
# bring the product on-device, then thin-QR. With return_R, also return the
# triangular factor (needed by the SVD and XTrace reductions). The product is
# consumed by the in-place QR; callers that must retain it (adaptive XTrace)
# keep their own copy and call qrthin!/materialize_mat directly.
function sketch(operator, Ω, sample_vec; return_R::Bool=false)
    Y = op_product(operator, Ω, sample_vec)
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
        copyto!(view(Y, :, s+1:num_cols), op_product(operator, Ω, sample_vec))
    end
    return qthin!(materialize_mat(Y, sample_vec))
end

# Find an orthonormal Q whose columns approximately span the range of `operator`,
# using randomized subspace iteration. For a self-adjoint (`hermitian=true`)
# operator each power iteration is a single multiply; otherwise it is an
# adjoint pass followed by a forward pass.
function randomized_range_finder(operator, num_samples::Int, num_power_iterations::Int, sample_vec::AbstractArray; hermitian::Bool=false, seed_Q=nothing)
    Q = range_finder_start(operator, num_samples, sample_vec, seed_Q)
    num_power_iterations == 0 && return Q
    if supports_inplace_mul(operator, sample_vec)
        Y = similar(Q, size(operator, 1), size(Q, 2))
        Z = hermitian ? nothing : similar(Q, size(operator, 2), size(Q, 2))
        for _ in 1:num_power_iterations # Compute power iterations
            if !hermitian # extra adjoint pass for general operators
                op_mul!(Z, operator', Q)
                Z = qthin!(Z)
                op_mul!(Y, operator, Z)
            else
                op_mul!(Y, operator, Q)
            end
            Q, Y = qthin!(Y), Q
        end
    else
        for _ in 1:num_power_iterations # Compute power iterations
            hermitian || (Q = sketch(operator', Q, sample_vec)) # extra adjoint pass for general operators
            Q = sketch(operator, Q, sample_vec)
        end
    end
    return Q
end
