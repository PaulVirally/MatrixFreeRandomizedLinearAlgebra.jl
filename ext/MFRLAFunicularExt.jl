module MFRLAFunicularExt

# The panel path. Everything here runs the same algorithms as the in-memory
# code, with the N × s sketch matrices held as Funicular `PanelMatrix` objects
# streamed through the device in column panels: `panelmul!` for the operator
# applications, `cholqr2!` for the orthonormalizations, and `gram` / `project`
# for the small reduced block, which is finished on the host exactly as the
# dense path finishes it.

using LinearAlgebra
using Random
using Funicular
using Funicular: ResidencyPlan
import MatrixFreeRandomizedLinearAlgebra as MFRLA
using MatrixFreeRandomizedLinearAlgebra: PanelEigen, PanelSVD, PanelFactored

# Panel width discipline. `panelmul!` and `gram` need the matrices they are
# given cut into the same column panels, and a plan that chooses widths for
# itself takes the row count into account, so two matrices built independently
# from one plan can come out with no step in common. The rule is therefore: the
# first matrix of a solve is built from the plan, and every companion comes from
# `similar` (or, for a regenerated test matrix, from `ghost_like`), which
# carries the width across a change of row count and clamps it to the new column
# count.
panel_like(reference, ::Type{T}, dims::Dims{2}) where {T} = similar(reference, T, dims)
ghost_like(reference, ::Type{T}, dims::Dims{2}, seed) where {T} = GhostPanels(T, dims[1], dims[2]; plan=Funicular.plan(reference), seed=seed, w=min(panelwidth(reference), dims[2]))

# Panel analogue of `randomized_range_finder`: the same subspace iteration, with
# a regenerated Gaussian in place of a stored one and CholeskyQR2 in place of the
# thin QR. A self-adjoint operator takes one multiply per iteration; a general
# one takes an adjoint pass into the domain and a forward pass back, each
# orthonormalized, which also keeps the iterates below the condition number
# CholeskyQR2 cannot hold.
function panel_range_finder(operator, num_samples::Int, num_power_iterations::Int, plan::ResidencyPlan, seed; hermitian::Bool=false, seed_Q=nothing)
    T = eltype(operator)
    Y = panel_range_start(operator, num_samples, plan, seed, seed_Q, T)
    num_power_iterations == 0 && return Y
    if hermitian
        # `Z` is allocated once and swapped with `Y` each iteration, as the
        # in-place fast path in range_finder.jl does.
        Z = similar(Y)
        for _ in 1:num_power_iterations
            panelmul!(Z, operator, Y)
            cholqr2!(Z)
            Y, Z = Z, Y
        end
        Funicular.free!(Z) # the swap leaves the previous iterate holding host blocks
        return Y
    end
    # `Z` lives in the domain, so it differs from `Y` in its row count only; the
    # two passes write into different matrices, so no swap is needed.
    Z = panel_like(Y, T, (size(operator, 2), size(Y, 2)))
    for _ in 1:num_power_iterations
        panelmul!(Z, operator', Y)
        cholqr2!(Z)
        panelmul!(Y, operator, Z)
        cholqr2!(Y)
    end
    Funicular.free!(Z)
    return Y
end

function panel_range_start(operator, num_samples::Int, plan::ResidencyPlan, seed, ::Nothing, ::Type{T}) where {T}
    Ω = GhostPanels(T, size(operator, 2), num_samples; plan=plan, seed=seed)
    Y = panel_like(Ω, T, (size(operator, 1), num_samples))
    panelmul!(Y, operator, Ω)
    Funicular.free!(Ω)
    cholqr2!(Y)
    return Y
end

function panel_range_start(operator, num_samples::Int, plan::ResidencyPlan, seed, seed_Q, ::Type{T}) where {T}
    m = size(operator, 1)
    size(seed_Q, 1) == m || throw(DimensionMismatch("seed_Q has $(size(seed_Q, 1)) rows but the operator range has dimension $m"))
    s = size(seed_Q, 2)
    num_cols = max(num_samples, s) # Keep all seed columns even if there are more than requested
    Y = PanelMatrix{T}(undef, m, num_cols; plan=plan) # the width anchor for this solve
    copy_seed_columns!(Y, seed_Q, s)
    if num_cols > s
        # Pad with `operator * Ω` columns to preserve oversampling. The pad is a
        # solve of its own, so it gets its own anchor cut to Y's width.
        pad = num_cols - s
        Ω = ghost_like(Y, T, (size(operator, 2), pad), seed)
        image = panel_like(Ω, T, (m, pad))
        panelmul!(image, operator, Ω)
        copycols!(Y, (s + 1):num_cols, image, 1:pad)
        Funicular.free!(Ω)
        Funicular.free!(image)
    end
    cholqr2!(Y)
    return Y
end

# A host seed lands column by column; a seed that is itself a panel matrix is
# copied panel to panel. `PanelMatrix` is deliberately not an `AbstractMatrix`,
# so the two never collide.
copy_seed_columns!(Y::PanelMatrix, seed_Q::AbstractMatrix, s::Int) = copycols!(Y, 1:s, seed_Q)
copy_seed_columns!(Y::PanelMatrix, seed_Q::PanelMatrix, s::Int) = copycols!(Y, 1:s, seed_Q, 1:s)
# A deferred product has no columns to copy until it is carried out, and the
# starting block is about to be orthonormalized anyway, so the temporary that
# holds it is short lived.
function copy_seed_columns!(Y::PanelMatrix, seed_Q::PanelFactored, s::Int)
    seeded = MFRLA.materialize(seed_Q)
    try
        copycols!(Y, 1:s, seeded, 1:s)
    finally
        Funicular.free!(seeded)
    end
    return Y
end

# The reduced block B = Q' * A * Q, by whichever of the two routes is cheaper.
# `project` never stores A * Q but sweeps the whole basis once per panel, so it
# moves p (p + 1) panels for p = npanels(Q); `panelmul!` into a second basis
# followed by `gram` moves two sweeps' worth instead, at the price of one more
# N × s matrix. Take the two-sweep route only when that matrix plainly fits: the
# host budget has to cover it on top of Q, which is already living there, and
# `gram` traverses rows and so holds every panel of both at once.
function panel_restricted(operator, Q::PanelMatrix)
    bytes = size(Q, 1) * size(Q, 2) * sizeof(eltype(Q))
    2 * bytes <= Funicular.plan(Q).host_budget || return project(Q, operator)
    Z = similar(Q)
    panelmul!(Z, operator, Q)
    B = gram(Q, Z)
    Funicular.free!(Z)
    return B
end

function panel_hermitian_setup(operator, plan::ResidencyPlan, validate::Bool)
    m, n = size(operator, 1), size(operator, 2)
    m == n || throw(DimensionMismatch("the Hermitian panel path requires a square operator, got size ($m, $n)"))
    validate && check_operator(operator; backend=plan.backend)
    return nothing
end

function MFRLA.reigen_hermitian_panel(operator, num_components::Int, plan::ResidencyPlan; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 14 : 8), seed_Q=nothing, seed, factored::Bool=false, validate::Bool=true)
    panel_hermitian_setup(operator, plan, validate)
    # A ≈ Q * B * Q' with B = Q' * A * Q, so the eigenvectors of B rotate Q into
    # the eigenvectors of A: exactly the reduction eigen_hermitian_restricted
    # performs in memory.
    Q = panel_range_finder(operator, num_components + num_oversamples, num_power_iterations, plan, seed; hermitian=true, seed_Q=seed_Q)
    B = Hermitian(panel_restricted(operator, Q))
    S = eigen!(B)
    idxs = sortperm(S.values, rev=true) # Sort eigenvalues in descending order
    k = min(num_components, size(operator, 1), size(B, 1)) # In case num_components > rank(B), we limit to rank(B)
    evals = S.values[idxs][1:k]
    rotation = S.vectors[:, idxs][:, 1:k]
    factored && return PanelEigen(evals, PanelFactored(Q, rotation))
    V = panel_like(Q, eltype(Q), (size(Q, 1), k))
    rightmul!(V, Q, rotation) # V = Q * Ṽ
    Funicular.free!(Q)
    return PanelEigen(evals, V)
end

function MFRLA.reigvals_hermitian_panel(operator, num_components::Int, plan::ResidencyPlan; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 14 : 8), seed_Q=nothing, seed, validate::Bool=true)
    panel_hermitian_setup(operator, plan, validate)
    Q = panel_range_finder(operator, num_components + num_oversamples, num_power_iterations, plan, seed; hermitian=true, seed_Q=seed_Q)
    B = Hermitian(panel_restricted(operator, Q))
    Funicular.free!(Q)
    evals = eigvals!(B)
    sort!(evals, rev=true) # Sort eigenvalues in descending order
    k = min(num_components, size(operator, 1), size(B, 1)) # In case num_components > rank(B), we limit to rank(B)
    return evals[1:k]
end

# XTrace's test matrix is sphere normalized: each column is a standard normal
# draw rescaled to length sqrt(N), which is what makes the estimator's
# variance-reduction formulas exact. A ghost generator is handed one whole
# column of host memory, so the rescaling here is exact per column, while the
# dense path applies it as a broadcast over a stored draw. Both are draws from
# the same distribution, but neither reproduces the other entry by entry.
function sphere_column!(dst::AbstractVector{T}, rng::AbstractRNG, ::Integer) where {T}
    randn!(rng, dst)
    dst .*= sqrt(real(T)(length(dst))) / norm(dst)
    return dst
end

# The tail columns of a wider ghost, as a ghost of its own. `GhostPanels` stores
# `hash(seed)` and fills column `col` of the matrix it is building from
# `Xoshiro(hash(col, that))`, so a generator that ignores the RNG it is passed
# and rebuilds that stream at `col + offset` yields columns `offset+1:offset+k`
# of the wide matrix, exactly. That is what lets the adaptive path widen its
# sketch without re-multiplying the columns it has already computed.
function offset_sphere_column(offset::Int, seed)
    hashed = hash(seed) # the same fold GhostPanels applies to its own seed
    return (dst, _rng, col) -> sphere_column!(dst, Xoshiro(hash(col + offset, hashed)), col + offset)
end

function MFRLA.trace_panel(operator, num_samples, plan::ResidencyPlan; relative_tolerance=nothing, low_mem::Bool=false, return_error::Bool=false, min_samples::Integer=30, max_samples::Integer=min(size(operator, 1), 512), seed, validate::Bool=true)
    low_mem && throw(MFRLA.trace_low_mem_with_plan())
    validate && check_operator(operator; backend=plan.backend)
    # `min_samples` steadies Hutchinson's variance estimate before its stopping
    # rule is first checked, and XTrace has no such rule, so it goes unused here
    # exactly as it does on the dense XTrace path.
    t, err = num_samples === nothing ?
             panel_xtrace_adaptive(operator, relative_tolerance, max_samples, plan, seed) :
             panel_xtrace_fixed(operator, num_samples, plan, seed)
    return return_error ? (value=t, error=err) : t
end

# The four small reduced blocks XTrace is written in terms of, computed from a
# basis `Q` of the sketch and the test matrix that produced it. `gram` traverses
# rows and so wants its two operands cut into the same panels, which is why `Z`
# comes from `panel_like(Q, ...)` and why the caller pins one width across the
# whole solve. `Z` is the only matrix allocated here and it is gone again by the
# time the host-side estimator runs.
function panel_xtrace_blocks(operator, Ω::PanelMatrix, Q::PanelMatrix, R::AbstractMatrix, n::Int, m::Int)
    Z = panel_like(Q, eltype(Q), (n, m))
    panelmul!(Z, operator, Q)
    W = gram(Q, Ω)
    H = gram(Q, Z)
    Tm = gram(Z, Ω)
    Funicular.free!(Z)
    return MFRLA.xtrace_from_blocks(n, m, W, H, Tm, R)
end

function panel_xtrace_fixed(operator, num_samples::Integer, plan::ResidencyPlan, seed)
    T = eltype(operator)
    n = size(operator, 1)
    m = clamp(Int(num_samples), 2, n) # XTrace needs at least 2, and at most n, test vectors
    Ω = GhostPanels(sphere_column!, T, n, m; plan=plan, seed=seed) # the width anchor for this solve
    Y = panel_like(Ω, T, (n, m))
    panelmul!(Y, operator, Ω)
    # Nothing downstream needs the raw image in fixed mode, so it is
    # orthonormalized in place and is Q from there on; the dense path's copy
    # exists only for the adaptive reuse below.
    R = cholqr2!(Y)
    t, err = panel_xtrace_blocks(operator, Ω, Y, R, n, m)
    Funicular.free!(Ω)
    Funicular.free!(Y)
    return t, err
end

function panel_xtrace_adaptive(operator, relative_tolerance::Real, max_samples::Integer, plan::ResidencyPlan, seed)
    T = eltype(operator)
    n = size(operator, 1)
    cap = min(Int(max_samples), n)
    m = clamp(10, 2, n) # initial sketch size
    Ω = GhostPanels(sphere_column!, T, n, m; plan=plan, seed=seed)
    # The plan chooses a width from the row count and the column count together,
    # so every later round would be cut differently as the sketch doubles. Pin
    # the first ghost's width and force it on every matrix of this solve, or the
    # `gram` calls in the reduction have nothing in common to sweep.
    w = panelwidth(Ω)
    Y = panel_like(Ω, T, (n, m))
    panelmul!(Y, operator, Ω)
    t, err = panel_xtrace_round(operator, Ω, Y, n, m)
    while err > relative_tolerance * abs(t) && m < cap
        add = min(m, cap - m) # double the sketch, capped
        # A ghost column is a function of its index and the seed, so the leading
        # m columns of the wider test matrix are the m already used, and their
        # images are the ones already in Y.
        Ω2 = GhostPanels(sphere_column!, T, n, m + add; plan=plan, seed=seed, w=w)
        Y2 = panel_like(Ω2, T, (n, m + add))
        copycols!(Y2, 1:m, Y, 1:m)
        Ωnew = GhostPanels(offset_sphere_column(m, seed), T, n, add; plan=plan, seed=seed, w=min(w, add))
        image = panel_like(Ωnew, T, (n, add))
        panelmul!(image, operator, Ωnew)
        copycols!(Y2, (m+1):(m+add), image, 1:add)
        Funicular.free!(image)
        Funicular.free!(Ωnew)
        Funicular.free!(Ω)
        Funicular.free!(Y)
        Ω, Y, m = Ω2, Y2, m + add
        t, err = panel_xtrace_round(operator, Ω, Y, n, m)
    end
    Funicular.free!(Ω)
    Funicular.free!(Y)
    return t, err
end

# One adaptive round. `cholqr2!` orthonormalizes in place, and Y has to survive
# for the next round's copy, so the basis is built in a matrix of its own and
# freed again as soon as the estimate is out; the dense path copies for the same
# reason.
function panel_xtrace_round(operator, Ω::PanelMatrix, Y::PanelMatrix, n::Int, m::Int)
    Q = panel_like(Y, eltype(Y), (n, m))
    copyto!(Q, Y)
    R = cholqr2!(Q)
    try
        return panel_xtrace_blocks(operator, Ω, Q, R, n, m)
    finally
        Funicular.free!(Q)
    end
end

# Shared entry work for the two SVD hooks: validate once, and settle the sketch
# dimension. A sketch wider than the smaller dimension of the operator has
# dependent columns, which CholeskyQR2 can only orthonormalize by falling back
# to its shifted third pass, so the width is capped here as on the dense path.
function panel_svd_setup(operator, num_components::Int, num_oversamples::Int, plan::ResidencyPlan, validate::Bool)
    validate && check_operator(operator; backend=plan.backend)
    return min(minimum(size(operator)), num_components + num_oversamples)
end

# `seed_Q` seeds a basis for the range of the operator, and a tall operator is
# solved through its adjoint, whose range is a different space.
function tall_seed_rejected(name::AbstractString)
    return ArgumentError("seed_Q is not supported for tall operators (size(operator, 1) > size(operator, 2)) because $name transposes them internally; transpose the operator yourself and seed a basis for its (wide) range, or omit seed_Q")
end

# The reduction both SVD hooks share, mirroring `svd_restricted`: with B' = A' Q
# orthonormalized as q * R, the SVD of A follows from the SVD of the small R.
# `cholqr2!` leaves q in `Bdag` itself, so this replaces both the dense path's
# `qrthin!` temporary and the separate `q` it returns.
function panel_svd_reduce(operator, Q::PanelMatrix)
    Bdag = panel_like(Q, eltype(Q), (size(operator, 2), size(Q, 2))) # B' = A' * Q
    panelmul!(Bdag, operator', Q)
    R = cholqr2!(Bdag) # B' = q * R, with q now in Bdag
    return Bdag, R
end

# The rank actually delivered: never more than the operator can carry, and never
# more than the sketch resolved.
panel_svd_rank(operator, num_components::Int, R::AbstractMatrix) = min(num_components, size(operator, 1), size(operator, 2), size(R, 2))

function MFRLA.rsvd_panel(operator, num_components::Int, plan::ResidencyPlan; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4), seed_Q=nothing, seed, factored::Bool=false, validate::Bool=true)
    num_samples = panel_svd_setup(operator, num_components, num_oversamples, plan, validate)
    if size(operator, 1) > size(operator, 2)
        seed_Q === nothing || throw(tall_seed_rejected("rsvd"))
        # A' ≈ U₁ Σ V₁' gives A ≈ V₁ Σ U₁', so the tall operator's factors are
        # the wide solve's two bases exchanged. Both are tall panel matrices, so
        # the swap moves no data.
        F = MFRLA.rsvd_panel(operator', num_components, plan; num_oversamples=num_oversamples, num_power_iterations=num_power_iterations, seed=seed, factored=factored, validate=false)
        return PanelSVD(F.V, F.S, F.U)
    end
    Q = panel_range_finder(operator, num_samples, num_power_iterations, plan, seed; seed_Q=seed_Q)
    Bdag, R = panel_svd_reduce(operator, Q)
    S = svd!(R) # R = Ũ * Σ̃ * Ṽ'
    k = panel_svd_rank(operator, num_components, R)
    svals = S.S[1:k]
    rotate_U = (S.Vt[1:k, :])' # U = Q * Ṽ
    rotate_V = S.U[:, 1:k]     # V = q * Ũ
    # The bases are the caller's from here on, whichever form they take.
    factored && return PanelSVD(PanelFactored(Q, rotate_U), svals, PanelFactored(Bdag, rotate_V))
    U = panel_like(Q, eltype(Q), (size(Q, 1), k))
    rightmul!(U, Q, rotate_U)
    # Order matters: freeing Q before V is allocated keeps three N-scale matrices
    # alive at the peak instead of four.
    Funicular.free!(Q)
    V = panel_like(Bdag, eltype(Bdag), (size(Bdag, 1), k))
    rightmul!(V, Bdag, rotate_V)
    Funicular.free!(Bdag)
    return PanelSVD(U, svals, V)
end

function MFRLA.rsvdvals_panel(operator, num_components::Int, plan::ResidencyPlan; num_oversamples::Int=num_components, num_power_iterations::Int=(num_components < 0.1 * minimum(size(operator)) ? 7 : 4), seed_Q=nothing, seed, validate::Bool=true)
    num_samples = panel_svd_setup(operator, num_components, num_oversamples, plan, validate)
    if size(operator, 1) > size(operator, 2)
        seed_Q === nothing || throw(tall_seed_rejected("rsvdvals"))
        return MFRLA.rsvdvals_panel(operator', num_components, plan; num_oversamples=num_oversamples, num_power_iterations=num_power_iterations, seed=seed, validate=false)
    end
    Q = panel_range_finder(operator, num_samples, num_power_iterations, plan, seed; seed_Q=seed_Q)
    Bdag, R = panel_svd_reduce(operator, Q)
    Funicular.free!(Q) # neither basis is returned, so nothing survives the reduction
    Funicular.free!(Bdag)
    k = panel_svd_rank(operator, num_components, R)
    return svdvals!(R)[1:k]
end

function MFRLA.materialize(f::PanelFactored{<:PanelMatrix,<:AbstractMatrix})
    V = panel_like(f.Q, eltype(f.Q), (size(f.Q, 1), size(f.C, 2)))
    rightmul!(V, f.Q, f.C)
    return V
end

# Same host-memory guard as `Matrix(::PanelMatrix)`, applied to the product
# rather than to the (wider) basis it is built from.
function Base.Matrix(f::PanelFactored{<:PanelMatrix,<:AbstractMatrix}; max_bytes::Real=Sys.free_memory() ÷ 2)
    V = MFRLA.materialize(f)
    try
        return Matrix(V; max_bytes=max_bytes)
    finally
        Funicular.free!(V)
    end
end

end # module MFRLAFunicularExt
