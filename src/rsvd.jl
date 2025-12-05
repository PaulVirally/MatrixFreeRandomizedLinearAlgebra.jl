using LinearAlgebra
using CUDA
using Random

"""
    rsvd(operator, num_components; num_oversamples=num_components,
         num_power_iterations=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4),
         sample_vec=similar(operator, eltype(operator), 0))

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

# Returns
A `LinearAlgebra.SVD` object `svd` such that

```julia
svd.U * Diagonal(svd.S) * svd.Vt ≈ operator
```

with `length(svd.S) == num_components` (or fewer is the effective numerical rank
is smaller).
"""
function rsvd(operator, num_components::Int; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4), sample_vec::AbstractArray=similar(operator, eltype(operator), 0))
    # We need to find an orthonormal matrix Q such that A ≈ Q * Q' * A (where A is the operator)
    Q = randomized_range_finder(operator, num_components + num_oversamples, num_power_iterations, sample_vec)
    return svd_restricted(operator, Q, min(num_components, size(operator)...), sample_vec) # We use Q to compute the restricted SVD
end

"""
    rsvdvals(operator, num_components; num_oversamples=num_components,
             num_power_iterations=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4),
             sample_vec=similar(operator, eltype(operator), 0))

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

# Returns
A vector of length `num_components` (or fewer if the effective numerical rank
is smaller) containing the leading singular values of `operator`.

This can be significantly cheaper (in memory and computation) to use than
[`rsvd`](@ref) when only singular values are needed.
"""
function rsvdvals(operator, num_components::Int; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4), sample_vec::AbstractArray=similar(operator, eltype(operator), 0))
    # We need to find an orthonormal matrix Q such that A ≈ Q * Q' * A (where A is the operator)
    Q = randomized_range_finder(operator, num_components + num_oversamples, num_power_iterations, sample_vec)
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

function randomized_range_finder(operator, num_samples::Int, num_power_iterations::Int, sample_vec::AbstractArray)
    Ω = similar(sample_vec, eltype(operator), size(operator, 2), num_samples)
    randn!(Ω) # Generate Gaussian random matrix
    Q = operator * Ω
    Q = qthin!(materialize_mat(Q, sample_vec))
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
