using LinearAlgebra
using CUDA
using Random

"""
    reigen_hermitian(operator, num_components;
                     num_oversamples=num_components,
                     num_power_iterations=(num_components < 0.1 * minimum(size(operator)) ? 14 : 8),
                     sample_vec=similar(operator, eltype(operator), 0), seed_Q=nothing,
                     plan=nothing, seed=nothing, factored=false, validate=true)

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
- `seed_Q = nothing`:
  Optional warm-start basis for the range finder. If you already have an
  (approximately) orthonormal matrix whose columns span the invariant subspace
  of `operator`, pass it here to seed the subspace iteration instead of starting
  from pure Gaussian noise. This is useful when refining a previous solve or
  marching a parameter/time step. `seed_Q` does not have to be perfectly
  orthonormal (it is re-orthonormalized internally) and may have fewer columns
  than `num_components + num_oversamples`, in which case it is padded with
  random columns. With `num_power_iterations = 0` the seed is used essentially
  as-is (up to re-orthonormalization). On the panel path this may also be a
  Funicular `PanelMatrix`, so a previous panel solve's vectors can be fed
  straight back in.
- `plan = nothing`:
  A Funicular `ResidencyPlan`. When given (and Funicular.jl is loaded), the
  `n × (num_components + num_oversamples)` sketch is built as a Funicular
  `PanelMatrix` streamed through the device in column panels rather than as a
  dense array, so a sketch larger than device memory is possible. The plan takes
  precedence over `sample_vec`, which is ignored on this path since the plan's
  backend decides where the panels are computed. `operator` must satisfy
  Funicular's operator contract (see `Funicular.check_operator`) and must be
  square.
- `seed = nothing`:
  Panel path only. Seed for Funicular's regenerated Gaussian test matrix.
  `nothing` draws a fresh seed from the global RNG, so repeated calls sketch
  differently, as they do in memory. An integer makes the sketch reproducible,
  and reproducible across changes to the panel width and the plan's budgets too,
  since Funicular generates a ghost column from its index and the seed alone.
- `factored::Bool = false`:
  Panel path only. If `true`, the eigenvectors come back as a
  [`PanelFactored`](@ref) holding the range basis and the small rotation. This
  saves the sweep and the `n × num_components` panel matrix the product would
  need. Call [`materialize`](@ref) or `Matrix` on it to build the vectors.
- `validate::Bool = true`:
  Panel path only. Run `Funicular.check_operator` on `operator` once on entry,
  which costs a handful of probe multiplies and catches an operator whose
  `adjoint` or `mul!` does not do what Funicular assumes.

# Returns
An `Eigen` object `E` such that

```julia
operator * E.vectors ≈ E.vectors * Diagonal(E.values)
```

with `length(E.values) == num_components` (or fewer if the effective numerical rank
is smaller). Eigenvalues are sorted in descending order.

With a `plan`, the result is a [`PanelEigen`](@ref) instead: `values` is a host
`Vector` and `vectors` is a Funicular `PanelMatrix` (or a [`PanelFactored`](@ref)
when `factored=true`).

# References
- N. Halko, P. G. Martinsson, and J. A. Tropp, "Finding Structure with
  Randomness: Probabilistic Algorithms for Constructing Approximate Matrix
  Decompositions", SIAM Review 53(2), 2011 (arXiv:0909.4061).
"""
function reigen_hermitian(operator, num_components::Int; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 14 : 8), sample_vec=nothing, seed_Q=nothing, plan=nothing, seed=nothing, factored::Bool=false, validate::Bool=true)
    if plan !== nothing
        return reigen_hermitian_panel(operator, num_components, plan; num_oversamples=num_oversamples, num_power_iterations=num_power_iterations, seed_Q=seed_Q, seed=resolve_panel_seed(seed), factored=factored, validate=validate)
    end
    seed === nothing || throw(seed_without_plan("reigen_hermitian"))
    factored && throw(factored_without_plan("reigen_hermitian"))
    prototype = resolve_sample_vec(operator, sample_vec)
    # We need to find an orthonormal matrix Q such that A ≈ Q * Q' * A (where A is the operator)
    Q = randomized_range_finder(operator, num_components + num_oversamples, num_power_iterations, prototype; hermitian=true, seed_Q=seed_Q)
    return eigen_hermitian_restricted(operator, Q, min(num_components, size(operator)...), prototype) # We use Q to compute the restricted spectral decomposition
end

"""
    reigvals_hermitian(operator, num_components;
                      num_oversamples=num_components,
                      num_power_iterations=(num_components < 0.1 * minimum(size(operator)) ? 14 : 8),
                      sample_vec=similar(operator, eltype(operator), 0), seed_Q=nothing,
                      plan=nothing, seed=nothing, validate=true)

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
- `seed_Q = nothing`:
    Optional warm-start basis for the range finder. If you already have an
    (approximately) orthonormal matrix whose columns span the invariant subspace
    of `operator`, pass it here to seed the subspace iteration instead of
    starting from pure Gaussian noise. It does not be perfectly orthonormal (it
    is re-orthonormalized internally) and may have fewer columns than
    `num_components + num_oversamples`, in which case it is padded with random
    columns. With `num_power_iterations = 0` the seed is used essentially as-is
    (up to re-orthonormalization). On the panel path this may also be a
    Funicular `PanelMatrix`.
- `plan = nothing`:
    A Funicular `ResidencyPlan`. When given (and Funicular.jl is loaded), the
    sketch is built as a Funicular `PanelMatrix` streamed through the device in
    column panels rather than as a dense array. The plan takes precedence over
    `sample_vec`, which is ignored on this path. `operator` must satisfy
    Funicular's operator contract (see `Funicular.check_operator`) and must be
    square. The eigenvalues still come back as a plain host `Vector`.
- `seed = nothing`:
    Panel path only. Seed for Funicular's regenerated Gaussian test matrix.
    `nothing` draws a fresh seed from the global RNG; an integer makes the
    sketch reproducible, including across changes to the panel width and the
    plan's budgets.
- `validate::Bool = true`:
    Panel path only. Run `Funicular.check_operator` on `operator` once on entry.

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
function reigvals_hermitian(operator, num_components::Int; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 14 : 8), sample_vec=nothing, seed_Q=nothing, plan=nothing, seed=nothing, validate::Bool=true)
    if plan !== nothing
        return reigvals_hermitian_panel(operator, num_components, plan; num_oversamples=num_oversamples, num_power_iterations=num_power_iterations, seed_Q=seed_Q, seed=resolve_panel_seed(seed), validate=validate)
    end
    seed === nothing || throw(seed_without_plan("reigvals_hermitian"))
    prototype = resolve_sample_vec(operator, sample_vec)
    # We need to find an orthonormal matrix Q such that A ≈ Q * Q' * A (where A is the operator)
    Q = randomized_range_finder(operator, num_components + num_oversamples, num_power_iterations, prototype; hermitian=true, seed_Q=seed_Q)
    return eigvals_hermitian_restricted(operator, Q, min(num_components, size(operator)...), prototype) # We use Q to compute the restricted spectral values
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

    # Materialize A * Q first (device-safe for any operator, including lazy
    # LinearMap composites), then reduce with a single gemm.
    B = Q' * op_product(operator, Q, sample_vec) # B = Q' * A * Q
    B = Hermitian(materialize_mat(B, sample_vec))
    if sample_vec isa CuArray
        # CUDA.jl does not support eigen! yet for these matrices
        S = eigen(B) # B = Ṽ * Λ̃ * Ṽ'
    else
        S = eigen!(B) # B = Ṽ * Λ̃ * Ṽ'
    end
    idxs = sortperm(S.values, rev=true) # Sort eigenvalues in descending order
    k = min(num_components, size(B, 1)) # In case num_components > rank(B), we limit to rank(B)
    evals = S.values[idxs][1:k] # Λ = Λ̃
    evecs = Q * S.vectors[:, idxs][:, 1:k] # V = Q * Ṽ
    return Eigen(evals, evecs)
end

function eigvals_hermitian_restricted(operator, Q, num_components::Int, sample_vec::AbstractArray)
    B = Q' * op_product(operator, Q, sample_vec) # B = Q' * A * Q
    B = Hermitian(materialize_mat(B, sample_vec))
    if sample_vec isa CuArray
        # CUDA.jl does not support eigen! yet for these matrices
        evals = eigen(B).values # B = Ṽ * Λ̃ * Ṽ'
    else
        evals = eigvals!(B) # B = Ṽ * Λ̃ * Ṽ'
    end
    sort!(evals, rev=true) # Sort eigenvalues in descending order
    k = min(num_components, size(B, 1)) # In case num_components > rank(B), we limit to rank(B)
    return evals[1:k] # Λ = Λ̃
end
