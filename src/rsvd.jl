using LinearAlgebra
using CUDA
using Random

"""
    rsvd(operator, num_components; num_oversamples=num_components,
         num_power_iterations=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4),
         sample_vec=similar(operator, eltype(operator), 0), seed_Q=nothing)

Compute a randomized low-rank singular value decomposition (SVD) of a
matrix-like linear operator.

This implements a standard randomized SVD with Gaussian test vectors,
oversampling, and power iteration. It is intended for large, possibly
matrix-free operators where forming a dense matrix is expensive.

# Arguments
- `operator`: A linear operator acting like an `AbstractMatrix`, supporting
  `size(operator)`, `operator * X`, and `operator' * X`. This may be a dense
  matrix, a sparse matrix, a GPU matrix (`CuMatrix`), or a matrix-free object
  (e.g. from `LinearMaps.jl`).
- `num_components::Int`: Target rank `k` for the approximation.

# Keyword arguments
- `num_oversamples::Int = num_components`:
  Number of oversampling vectors `p`. The effective sketch dimension is
  `k + p`. Larger oversampling improves accuracy at the cost of extra
  multiplies.
- `num_power_iterations::Int`:
  Number of power-iteration refinement steps. If `num_components` is less than
  10% of the smaller dimension of `operator`, the default is 7; otherwise 4.
  Increasing this improves spectral separation for slow-decaying singular
  values but increases cost by additional passes over `operator` and
  `operator'`.
- `sample_vec::AbstractVector`:
  Prototype vector used to control allocation of random test matrices. By
  default, `similar(operator, eltype(operator), 0)` is used so that temporary
  arrays are allocated on the same device/storage as `operator`. You can pass
  a vector living on a different device (e.g. a `CuVector`) to force all
  temporaries onto that device.
- `seed_Q = nothing`:
  Optional warm-start basis for the range finder. If you already have an
  (approximately) orthonormal matrix whose columns span the range of `operator`
  (column space), pass it here to seed the subspace iteration instead of
  starting from pure Gaussian noise. It need not be perfectly orthonormal (it is
  re-orthonormalized internally) and may have fewer columns than `num_components
  + num_oversamples`, in which case it is padded with random columns. With
  `num_power_iterations = 0` the seed is used essentially as-is (up to
  re-orthonormalization).  Seeding is not supported for tall operators
  (`size(operator, 1) > size(operator, 2)`), which are transposed internally.
  For tall operators, the user is expected to transpose the operator themselves
  and seed a basis for its (wide) range, or omit `seed_Q`.

# Returns
A `LinearAlgebra.SVD` object `svd` such that

```julia
svd.U * Diagonal(svd.S) * svd.Vt ≈ operator
```

with `length(svd.S) == num_components` (or fewer is the effective numerical rank
is smaller).
"""
function rsvd(operator, num_components::Int; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4), sample_vec::AbstractArray=similar(operator, eltype(operator), 0), seed_Q=nothing)
    if size(operator, 1) > size(operator, 2)
        # For tall matrices, we can use the transpose to reduce work. This puts the
        # range finder in a different space than the operator's column space, so a
        # range seed cannot be threaded through and is not supported here.
        seed_Q === nothing || throw(ArgumentError("seed_Q is not supported for tall operators (size(operator, 1) > size(operator, 2)) because rsvd transposes them internally; transpose the operator yourself and seed a basis for its (wide) range, or omit seed_Q"))
        svd_t = rsvd(operator', num_components; num_oversamples=num_oversamples, num_power_iterations=num_power_iterations, sample_vec=sample_vec)
        return svd_t' # SVD type supports adjoint
    end

    # We need to find an orthonormal matrix Q such that A ≈ Q * Q' * A (where A is the operator)
    num_samples = min(min(size(operator)...) , num_components + num_oversamples)
    Q = randomized_range_finder(operator, num_samples, num_power_iterations, sample_vec; seed_Q=seed_Q)
    return svd_restricted(operator, Q, min(num_components, size(operator)...), sample_vec) # We use Q to compute the restricted SVD
end

"""
    rsvdvals(operator, num_components; num_oversamples=num_components,
             num_power_iterations=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4),
             sample_vec=similar(operator, eltype(operator), 0), seed_Q=nothing)

Compute the leading singualr values of a matrix-like poerator using randomized SVD techniques, without explcitly forming the singular vectors.

This has the same interface and algorithmic structure as [`rsvd`](@ref), but
only returns the approximate singualr values.

# Arguments
- `operator`: A linear operator acting like an `AbstractMatrix`, supporting
  `size(operator)`, `operator * X`, and `operator' * X`. This may be a dense
  matrix, a sparse matrix, a GPU matrix (`CuMatrix`), or a matrix-free object
  (e.g. from `LinearMaps.jl`).
- `num_components::Int`: Target number of singular values `k` to compute.

# Keyword arguments
- `num_oversamples::Int = num_components`:
  Number of oversampling vectors `p`. The effective sketch dimension is
  `k + p`. Larger oversampling improves accuracy at the cost of extra
  multiplies.
- `num_power_iterations::Int`:
    Number of power-iteration refinement steps. If `num_components` is less than
    10% of the smaller dimension of `operator`, the default is 7; otherwise 4.
    Increasing this improves spectral separation for slow-decaying singular
    values but increases cost by additional passes over `operator` and
    `operator'`.
- `sample_vec::AbstractVector`:
  Prototype vector used to control allocation of random test matrices. By
  default, `similar(operator, eltype(operator), 0)` is used so that temporary
  arrays are allocated on the same device/storage as `operator`. You can pass
  a vector living on a different device (e.g. a `CuVector`) to force all
  temporaries onto that device.
- `seed_Q = nothing`:
  Optional warm-start basis for the range finder. If you already have an
  (approximately) orthonormal matrix whose columns span the range of `operator`
  (column space), pass it here to seed the subspace iteration instead of
  starting from pure Gaussian noise. It need not be perfectly orthonormal (it is
  re-orthonormalized internally) and may have fewer columns than `num_components
  + num_oversamples`, in which case it is padded with random columns. With
  `num_power_iterations = 0` the seed is used essentially as-is.  Seeding is not
  supported for tall operators (`size(operator, 1) > size(operator, 2)`). For
  tall operators, the user is expected to transpose the operator themselves

# Returns
A vector of length `num_components` (or fewer if the effective numerical rank
is smaller) containing the leading singular values of `operator`.

This can be significantly cheaper (in memory and computation) to use than
[`rsvd`](@ref) when only singular values are needed.
"""
function rsvdvals(operator, num_components::Int; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4), sample_vec::AbstractArray=similar(operator, eltype(operator), 0), seed_Q=nothing)
    if size(operator, 1) > size(operator, 2)
        # For tall matrices, we can use the transpose to reduce work. This puts the
        # range finder in a different space than the operator's column space, so a
        # range seed cannot be threaded through and is not supported here.
        seed_Q === nothing || throw(ArgumentError("seed_Q is not supported for tall operators (size(operator, 1) > size(operator, 2)) because rsvdvals transposes them internally; transpose the operator yourself and seed a basis for its (wide) range, or omit seed_Q"))
        return rsvdvals(operator', num_components; num_oversamples=num_oversamples, num_power_iterations=num_power_iterations, sample_vec=sample_vec)
    end

    # We need to find an orthonormal matrix Q such that A ≈ Q * Q' * A (where A is the operator)
    num_samples = min(min(size(operator)...) , num_components + num_oversamples)
    Q = randomized_range_finder(operator, num_samples, num_power_iterations, sample_vec; seed_Q=seed_Q)
    return svdvals_restricted(operator, Q, min(num_components, size(operator)...), sample_vec) # We use Q to compute the restricted SVD values
end

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

# Build the (orthonormalized) starting block for a randomized range finder.
# Without a seed, the block is `operator * Ω` for a Gaussian random `Ω`, exactly
# as before. When `seed_Q` is provided, its columns seed the block (a warm
# start); any shortfall up to `num_samples` columns is padded with `operator * Ω`
# so oversampling is preserved. The result lives on `sample_vec`'s device.
function range_finder_start(operator, num_samples::Int, sample_vec::AbstractArray, seed_Q)
    if seed_Q === nothing
        Ω = similar(sample_vec, eltype(operator), size(operator, 2), num_samples)
        randn!(Ω) # Generate Gaussian random matrix
        return qthin!(materialize_mat(operator * Ω, sample_vec))
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

function randomized_range_finder(operator, num_samples::Int, num_power_iterations::Int, sample_vec::AbstractArray; seed_Q=nothing)
    Q = range_finder_start(operator, num_samples, sample_vec, seed_Q)
    for i in 1:num_power_iterations # Compute power iterations
        Q = qthin!(materialize_mat(operator' * Q, sample_vec))
        Q = qthin!(materialize_mat(operator * Q, sample_vec))
    end
    return Q
end

function svd_restricted(operator, Q, num_components::Int, sample_vec::AbstractArray)
    # We are given Q such that A ≈ Q * Q' * A (where A is the operator)
    # We want to find A = U * Σ * V'
    # 
    # Let B' = A' * Q ⇒ B = Q' * A, thus
    # Q * B = Q * Q' * A ≈ A = U * Σ * V'
    # Further, let B' = q * r ⇒ B = r' * q'
    # Q * B = Q * r' * q'
    # Let r = Ũ * Σ̃ * Ṽ' ⇒ r' = Ṽ * Σ̃ * Ũ'
    # Q * B = Q * (Ṽ * Σ̃ * Ũ') * q' = (Q * Ṽ) * Σ̃ * (q * Ũ)'
    # Thus:
    # U ≈ Q * Ṽ
    # Σ ≈ Σ̃
    # V' ≈ (q * Ũ)'
    Bdag = operator' * Q # B' = A' * Q
    k = min(num_components, size(Bdag, 2)) # In case num_components > rank(B), we limit to rank(B)
    q, r = qrthin!(materialize_mat(Bdag, sample_vec)) # B' = q * r
    S = svd!(r) # r = Ũ * Σ̃ * Ṽ'
    left_svecs = Q * (S.Vt[1:k, :])' # U = Q * Ṽ
    svals = S.S[1:k] # Σ = Σ̃
    right_svecs = q * S.U[:, 1:k] # V' = (q * Ũ)' ⟹ V = q * Ũ
    return SVD(left_svecs, svals, right_svecs') # SVD takes (U, Σ, V'), not (U, Σ, V)
end

function svdvals_restricted(operator, Q, num_components::Int, sample_vec::AbstractArray)
    Bdag = operator' * Q # B' = A' * Q
    k = min(num_components, size(Bdag, 2)) # In case num_components > rank(B), we limit to rank(B)
    _, r = qrthin!(materialize_mat(Bdag, sample_vec)) # B' = q * r
    Σ = svdvals!(materialize_mat(r, sample_vec)) # r = Ũ * Σ̃ * Ṽ'
    return Σ[1:k] # Σ = Σ̃
end
