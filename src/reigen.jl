using LinearAlgebra
using CUDA
using Random

"""
    reigen_hermitian(operator, num_components;
                     num_oversamples=num_components,
                     num_power_iterations=(num_components < 0.1 * minimum(size(operator)) ? 14 : 8),
                     sample_vec=similar(operator, eltype(operator), 0))

Compute a randomized eigendecomposition of a Hermitian matrix-like operator.

This routine finds approximate leading eigenvalues and eigenvectors of a
Hermitian operator `operator` using randomized subspace iteration. It first
builds an approximate invariant subspace via a randomized range finder and then
computes the exact eigendecomposition of `operator` restricted to that subspace.

# Arguments
- `operator`: Hermitian linear operator (self-adjoint with respect to the
  standard inner product), supporting `size(operator)` and `operator * X`.
- `num_components::Int`: Number of leading eigenpairs to approximate.

# Keyword arguments
- `num_oversamples::Int = num_components`:
  Oversampling parameter `p`; the sketch dimension is `num_components + p`.
- `num_power_iterations::Int`:
  Number of power iterations used in the Hermitian range finder. Defaults to 14
  for relatively small ranks (when `num_components < 0.1 * min(size(operator))`)
  and 8 otherwise. Larger values improve separation of clustered eigenvalues.
- `sample_vec::AbstractVector`:
  Prototype vector used for allocating random test matrices. Controls whether
  temporaries live on CPU or GPU.

# Returns
An `Eigen` object `E` such that

```julia
operator * E.vectors ≈ E.vectors * Diagonal(E.values)
```

with `length(E.values) == num_components` (or fewer if the effective numerical rank
is smaller). Eigenvalues are sorted in descending order.
"""
function reigen_hermitian(operator, num_components::Int; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 14 : 8), sample_vec::AbstractArray=similar(operator, eltype(operator), 0))
    # We need to find an orthonormal matrix Q such that A ≈ Q * Q' * A (where A is the operator)
    Q = randomized_hermitian_range_finder(operator, num_components + num_oversamples, num_power_iterations, sample_vec)
    return eigen_hermitian_restricted(operator, Q, min(num_components, size(operator)...), sample_vec) # We use Q to compute the restricted spectral decomposition
end

"""
    reigvals_hermitian(operator, num_components;
                      num_oversamples=num_components,
                      num_power_iterations=(num_components < 0.1 * minimum(size(operator)) ? 14 : 8),
                      sample_vec=similar(operator, eltype(operator), 0))

Compute approximate leading eigenvalues of a Hermitian matrix-like operator.

This routine finds approximate leading eigenvalues of a Hermitian operator
`operator` using randomized subspace iteration. It first builds an approximate
invariant subspace via a randomized range finder and then computes the exact
eigenvalues of `operator` restricted to that subspace.

# Arguments
- `operator`: Hermitian linear operator (self-adjoint with respect to the
    standard inner product), supporting `size(operator)` and `operator * X`.
- `num_components::Int`: Number of leading eigenvalues to approximate.

# Keyword arguments
- `num_oversamples::Int = num_components`:
  Oversampling parameter `p`; the sketch dimension is `num_components + p`.
- `num_power_iterations::Int`:
    Number of power iterations used in the Hermitian range finder. Defaults to 14
    for relatively small ranks (when `num_components < 0.1 * min(size(operator))`)
    and 8 otherwise. Larger values improve separation of clustered eigenvalues.
- `sample_vec::AbstractVector`:
    Prototype vector used for allocating random test matrices. Controls whether
    temporaries live on CPU or GPU.

# Returns
A vector of approximate leading eigenvalues `evals` such that

```julia
operator * v ≈ evals[i] * v
```

for the corresponding eigenvector `v` (not returned here), with `length(evals) == num_components`
(or fewer if the effective numerical rank is smaller). Eigenvalues are sorted in descending order.

This can be significantly cheaper than [`reigen_hermitian`](@ref) if only
eigenvalues are needed.
"""
function reigvals_hermitian(operator, num_components::Int; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 14 : 8), sample_vec::AbstractArray=similar(operator, eltype(operator), 0))
    # We need to find an orthonormal matrix Q such that A ≈ Q * Q' * A (where A is the operator)
    Q = randomized_hermitian_range_finder(operator, num_components + num_oversamples, num_power_iterations, sample_vec)
    return eigvals_hermitian_restricted(operator, Q, min(num_components, size(operator)...), sample_vec) # We use Q to compute the restricted spectral values
end

function randomized_hermitian_range_finder(operator, num_samples::Int, num_power_iterations::Int, sample_vec::AbstractArray)
    Ω = similar(sample_vec, eltype(operator), size(operator, 2), num_samples)
    randn!(Ω) # Generate Gaussian random matrix
    Q = operator * Ω
    Q = qthin!(materialize_mat(Q, sample_vec))
    for i in 1:num_power_iterations # Compute power iterations
        Q = qthin!(materialize_mat(operator * Q, sample_vec))
    end
    return Q
end

function eigen_hermitian_restricted(operator, Q, num_components::Int, sample_vec::AbstractArray)
    # We are given Q such that A ≈ Q * Q' * A (where A is the operator)
    # We want to find A = V * Λ * V'
    # 
    # Let B = Q' * A * Q ⟹ A ≈ Q * B * Q'
    # Further, let B = Ṽ * Λ̃ * Ṽ'
    # Then A ≈ Q * B * Q' = Q * Ṽ * Λ̃ * Ṽ' * Q' = (Q * Ṽ) * Λ̃ * (Q * Ṽ)'
    # Thus:
    # Λ = Λ̃
    # V = Q * Ṽ
    B = Q' * (operator * Q) # B = Q' * A * Q
    S = eigen!(Hermitian(materialize_mat(B, sample_vec))) # B = Ṽ * Λ̃ * Ṽ'
    idxs = sortperm(S.values, rev=true) # Sort eigenvalues in descending order
    k = min(num_components, size(B, 1)) # In case num_components > rank(B), we limit to rank(B)
    evals = S.values[idxs][1:k] # Λ = Λ̃
    evecs = Q * S.vectors[:, idxs][:, 1:k] # V = Q * Ṽ
    return Eigen(evals, evecs)
end

function eigvals_hermitian_restricted(operator, Q, num_components::Int, sample_vec::AbstractArray)
    B = Q' * (operator * Q) # B = Q' * A * Q
    evals = eigvals!(Hermitian(materialize_mat(B, sample_vec))) # B = Ṽ * Λ̃ * Ṽ'
    sort!(evals, rev=true) # Sort eigenvalues in descending order
    k = min(num_components, size(B, 1)) # In case num_components > rank(B), we limit to rank(B)
    return evals[1:k] # Λ = Λ̃
end
