using LinearAlgebra
using CUDA
using Random

"""
    rsvd(operator, num_components; num_oversamples=num_components,
         num_power_iterations=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4),
         sample_vec=similar(operator, eltype(operator), 0), seed_Q=nothing,
         plan=nothing, seed=nothing, factored=false, validate=true)

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
  and seed a basis for its (wide) range, or omit `seed_Q`. On the panel path
  this may also be a Funicular `PanelMatrix` or a [`PanelFactored`](@ref), so a
  previous panel solve's `U` can be fed straight back in.
- `plan = nothing`:
  A Funicular `ResidencyPlan`. When given (and Funicular.jl is loaded), the
  `m × (num_components + num_oversamples)` sketch and the `n × (num_components +
  num_oversamples)` block it is reduced against are built as Funicular
  `PanelMatrix` objects streamed through the device in column panels rather than
  as dense arrays, so a sketch larger than device memory is possible. The plan
  takes precedence over `sample_vec`, which is ignored on this path since the
  plan's backend decides where the panels are computed. `operator` must satisfy
  Funicular's operator contract (see `Funicular.check_operator`). A tall operator
  is transposed internally here too, and its factors come back exchanged rather
  than recomputed.

  This path orthonormalizes the sketch and `A' * Q` with CholeskyQR2 rather than
  with the Householder QR the in-memory path uses. CholeskyQR2 forms a Gram
  matrix and so squares the condition number before factoring, which it only
  holds up to `κ ≲ eps^(-1/2)`, about `1e8` in double precision. Past that
  Funicular falls back to a shifted CholeskyQR3, which costs a third pass over
  the block and keeps a numerically rank-deficient sketch from failing outright.
  It is only guaranteed to recover the singular values, not the basis: columns
  that were dependent come back as columns of `U` or `V` that are not
  meaningfully orthonormal, and a block too dependent for the shift to rescue
  raises instead. When rank deficiency cannot be ruled out, check
  `opnorm(Matrix(F.U)' * Matrix(F.U) - I)` before relying on the vectors.
  Oversampling less aggressively, or asking for fewer components, keeps the
  sketch out of that regime altogether.
- `seed = nothing`:
  Panel path only. Seed for Funicular's regenerated Gaussian test matrix.
  `nothing` draws a fresh seed from the global RNG, so repeated calls sketch
  differently, as they do in memory. An integer makes the sketch reproducible,
  and reproducible across changes to the panel width and the plan's budgets too,
  since Funicular generates a ghost column from its index and the seed alone.
- `factored::Bool = false`:
  Panel path only. If `true`, both sets of singular vectors come back as
  [`PanelFactored`](@ref) objects holding a basis and the small rotation that
  turns it into the vectors. This saves two sweeps and two `N × num_components`
  panel matrices. Call [`materialize`](@ref) or `Matrix` on either to build it.
- `validate::Bool = true`:
  Panel path only. Run `Funicular.check_operator` on `operator` once on entry,
  which costs a handful of probe multiplies and catches an operator whose
  `adjoint` or `mul!` does not do what Funicular assumes.

# Returns
A `LinearAlgebra.SVD` object `svd` such that

```julia
svd.U * Diagonal(svd.S) * svd.Vt ≈ operator
```

with `length(svd.S) == num_components` (or fewer is the effective numerical rank
is smaller).

With a `plan`, the result is a [`PanelSVD`](@ref) instead: `S` is a host `Vector`
and `U` and `V` are Funicular `PanelMatrix` objects (or [`PanelFactored`](@ref)
products when `factored=true`). Note that a `PanelSVD` holds the tall `V`, not
the wide `Vt`, since only a tall matrix can be cut into column panels.

# References
- N. Halko, P. G. Martinsson, and J. A. Tropp, "Finding Structure with
  Randomness: Probabilistic Algorithms for Constructing Approximate Matrix
  Decompositions", SIAM Review 53(2), 2011 (arXiv:0909.4061).
"""
function rsvd(operator, num_components::Int; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4), sample_vec=nothing, seed_Q=nothing, plan=nothing, seed=nothing, factored::Bool=false, validate::Bool=true)
    if plan !== nothing
        return rsvd_panel(operator, num_components, plan; num_oversamples=num_oversamples, num_power_iterations=num_power_iterations, seed_Q=seed_Q, seed=resolve_panel_seed(seed), factored=factored, validate=validate)
    end
    seed === nothing || throw(seed_without_plan("rsvd"))
    factored && throw(factored_without_plan("rsvd"))
    sample_vec = resolve_sample_vec(operator, sample_vec)
    if size(operator, 1) > size(operator, 2)
        # For tall matrices, we can use the transpose to reduce work. This puts the
        # range finder in a different space than the operator's column space, so a
        # range seed cannot be threaded through and is not supported here.
        seed_Q === nothing || throw(ArgumentError("seed_Q is not supported for tall operators (size(operator, 1) > size(operator, 2)) because rsvd transposes them internally; transpose the operator yourself and seed a basis for its (wide) range, or omit seed_Q"))
        svd_t = rsvd(operator', num_components; num_oversamples=num_oversamples, num_power_iterations=num_power_iterations, sample_vec=sample_vec)
        F = svd_t' # adjoint maps the wide SVD back to the tall operator's SVD
        # adjoint(::SVD) leaves U/Vt as lazy `Adjoint` wrappers; copy them into plain
        # arrays so they match the operator's storage (e.g. a `CuArray`, not an
        # `Adjoint{CuArray}`). `copy` conjugate-transposes correctly for complex too.
        return SVD(copy(F.U), F.S, copy(F.Vt))
    end

    # We need to find an orthonormal matrix Q such that A ≈ Q * Q' * A (where A is the operator)
    num_samples = min(min(size(operator)...) , num_components + num_oversamples)
    Q = randomized_range_finder(operator, num_samples, num_power_iterations, sample_vec; seed_Q=seed_Q)
    return svd_restricted(operator, Q, min(num_components, size(operator)...), sample_vec) # We use Q to compute the restricted SVD
end

"""
    rsvdvals(operator, num_components; num_oversamples=num_components,
             num_power_iterations=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4),
             sample_vec=similar(operator, eltype(operator), 0), seed_Q=nothing,
             plan=nothing, seed=nothing, validate=true)

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
  tall operators, the user is expected to transpose the operator themselves. On
  the panel path this may also be a Funicular `PanelMatrix` or a
  [`PanelFactored`](@ref).
- `plan = nothing`:
  A Funicular `ResidencyPlan`. When given (and Funicular.jl is loaded), the
  sketch and the block it is reduced against are built as Funicular
  `PanelMatrix` objects streamed through the device in column panels rather than
  as dense arrays. The plan takes precedence over `sample_vec`, which is ignored
  on this path. `operator` must satisfy Funicular's operator contract (see
  `Funicular.check_operator`). The singular values still come back as a plain
  host `Vector`. The CholeskyQR2 caveat for a numerically rank-deficient sketch
  applies here as it does to [`rsvd`](@ref); see that docstring.
- `seed = nothing`:
  Panel path only. Seed for Funicular's regenerated Gaussian test matrix.
  `nothing` draws a fresh seed from the global RNG; an integer makes the sketch
  reproducible, including across changes to the panel width and the plan's
  budgets.
- `validate::Bool = true`:
  Panel path only. Run `Funicular.check_operator` on `operator` once on entry.

# Returns
A vector of length `num_components` (or fewer if the effective numerical rank
is smaller) containing the leading singular values of `operator`.

This can be significantly cheaper (in memory and computation) to use than
[`rsvd`](@ref) when only singular values are needed.
"""
function rsvdvals(operator, num_components::Int; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4), sample_vec=nothing, seed_Q=nothing, plan=nothing, seed=nothing, validate::Bool=true)
    if plan !== nothing
        return rsvdvals_panel(operator, num_components, plan; num_oversamples=num_oversamples, num_power_iterations=num_power_iterations, seed_Q=seed_Q, seed=resolve_panel_seed(seed), validate=validate)
    end
    seed === nothing || throw(seed_without_plan("rsvdvals"))
    sample_vec = resolve_sample_vec(operator, sample_vec)
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
    Bdag = op_product(operator', Q, sample_vec) # B' = A' * Q
    k = min(num_components, size(Bdag, 2)) # In case num_components > rank(B), we limit to rank(B)
    q, r = qrthin!(Bdag) # B' = q * r
    S = svd!(r) # r = Ũ * Σ̃ * Ṽ'
    left_svecs = Q * (S.Vt[1:k, :])' # U = Q * Ṽ
    svals = S.S[1:k] # Σ = Σ̃
    right_svecs = q * S.U[:, 1:k] # V' = (q * Ũ)' ⟹ V = q * Ũ
    return SVD(left_svecs, svals, right_svecs') # SVD takes (U, Σ, V'), not (U, Σ, V)
end

function svdvals_restricted(operator, Q, num_components::Int, sample_vec::AbstractArray)
    Bdag = op_product(operator', Q, sample_vec) # B' = A' * Q
    k = min(num_components, size(Bdag, 2)) # In case num_components > rank(B), we limit to rank(B)
    _, r = qrthin!(Bdag) # B' = q * r
    Σ = svdvals!(materialize_mat(r, sample_vec)) # r = Ũ * Σ̃ * Ṽ'
    return Σ[1:k] # Σ = Σ̃
end
