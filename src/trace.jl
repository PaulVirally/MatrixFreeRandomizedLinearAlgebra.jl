using LinearAlgebra
using CUDA
using Random

"""
    trace(operator, num_samples; relative_tolerance=nothing, low_mem=false,
          return_error=false, min_samples=30,
          max_samples=(low_mem ? 4096 : min(size(operator, 1), 512)),
          sample_vec=similar(operator, eltype(operator), 0))

Estimate the trace of a square matrix-like linear operator using randomized
(stochastic) trace estimation.

Two estimators are available. By default `trace` uses XTrace, the exchangeable
estimator of Epperly, Tropp and Webber, which combines a low-rank sketch with a
leave-one-out Hutchinson correction and so makes the most of every sample. When
`low_mem=true`, `trace` falls back to the streaming Girard-Hutchinson estimator,
which only ever holds one or two vectors at a time. Hutchinson is less accurate
per matrix-vector product, but is the right choice when the operator is so large
that even a small sketch (a handful of `size(operator, 1)`-length vectors) does
not fit in memory.

The sample budget can be given two ways: a fixed `num_samples`, or a target
`relative_tolerance` that the estimator refines towards adaptively. Provide
exactly one of the two.

# Arguments
- `operator`: A square linear operator acting like an `AbstractMatrix`,
  supporting `size(operator)`, `operator * X`, and (for XTrace) `operator' * X`.
  This may be a dense matrix, a sparse matrix, a GPU matrix (`CuMatrix`), or a
  matrix-free object (e.g., from `LinearMaps.jl`).
- `num_samples::Int` (optional, positional): Fixed sample budget. For Hutchinson
  this is the number of matrix-vector products. For XTrace this is the number of
  random test vectors `m` (which sets the memory cost); XTrace performs `2m`
  matrix-vector products. Omit this and pass `relative_tolerance` instead to run
  adaptively.

# Keyword arguments
- `relative_tolerance::Real = nothing`:
  Target relative error on the trace. When set (and `num_samples` is omitted),
  the estimator adds samples until its error estimate falls below
  `relative_tolerance * abs(trace)`. For Hutchinson the stopping rule is a
  central-limit half-width (drawn after at least `min_samples` samples); for
  XTrace the sketch size is doubled until the jackknife error is small enough.
- `low_mem::Bool = false`:
  If `true`, use the streaming Hutchinson estimator (minimal memory). If
  `false`, use XTrace.
- `return_error::Bool = false`:
  If `true`, return `(value=t, error=e)` where `e` is the estimated standard
  error (the jackknife error for XTrace, the sample standard error for
  Hutchinson). If `false`, return just the scalar estimate `t`.
- `min_samples::Int = 30`:
  Hutchinson only. The number of samples drawn before the adaptive stopping rule
  is first checked, so the variance estimate has settled.
- `max_samples::Int`:
  Cap on the number of samples (Hutchinson) or test vectors (XTrace) used by the
  adaptive modes. Defaults to `4096` for Hutchinson and `min(size(operator, 1),
  512)` for XTrace.
- `sample_vec::AbstractVector`:
  Prototype vector used to control allocation of random test vectors. By
  default, `similar(operator, eltype(operator), 0)` is used so that temporary
  arrays are allocated on the same device/storage as `operator`. You can pass a
  vector living on a different device (e.g. a `CuVector`) to force all
  temporaries onto that device.

# Returns
The estimated trace (a scalar of `eltype(operator)`), or, if
`return_error=true`, a named tuple `(value, error)` with the estimate and its
estimated standard error.

# References
- E. N. Epperly, J. A. Tropp, and R. J. Webber, "XTrace: Making the Most of
  Every Sample in Stochastic Trace Estimation", SIAM J. Matrix Anal. Appl.
  45(1), 2024 (arXiv:2301.07825).
- M. F. Hutchinson, "A stochastic estimator of the trace of the influence matrix
  for Laplacian smoothing splines", Comm. Statist. Simulation Comput. 18(3),
  1989; A. Girard, "A fast Monte-Carlo cross-validation procedure for large
  least squares problems with noisy data", Numer. Math. 56, 1989.
"""
function trace(operator, num_samples::Union{Integer,Nothing}=nothing; relative_tolerance::Union{Real,Nothing}=nothing, low_mem::Bool=false, return_error::Bool=false, min_samples::Integer=30, max_samples::Integer=(low_mem ? 4096 : min(size(operator, 1), 512)), sample_vec::AbstractArray=similar(operator, eltype(operator), 0))
    size(operator, 1) == size(operator, 2) || throw(DimensionMismatch("trace requires a square operator, got size $(size(operator))"))
    # Exactly one of num_samples / relative_tolerance must be given
    if (num_samples === nothing) == (relative_tolerance === nothing)
        throw(ArgumentError("provide exactly one of `num_samples` (positional) or `relative_tolerance` (keyword)"))
    end

    if low_mem
        t, e = num_samples === nothing ?
               hutchinson_trace_adaptive(operator, relative_tolerance, min_samples, max_samples, sample_vec) :
               hutchinson_trace_fixed(operator, num_samples, sample_vec)
    else
        t, e = num_samples === nothing ?
               xtrace_adaptive(operator, relative_tolerance, max_samples, sample_vec) :
               xtrace_fixed(operator, num_samples, sample_vec)
    end
    return return_error ? (value=t, error=e) : t
end

# Fill x with Rademacher entries. Real eltype: ±1 with equal probability.
# Complex eltype: uniform over the four units {1, im, -1, -im}. Both satisfy
# E[x x'] = I, so E[x' A x] = tr(A). The implementations are single fused
# broadcasts over isbits eltypes with no scalar indexing, so the same kernel runs
# on the GPU. The particular forms were chosen after benchmarking several
# alternatives on large vectors.
function rademacher!(x::AbstractArray{T}) where {T<:Real}
    rand!(x) # Reuse x as its own uniform buffer
    x .= ifelse.(x .< T(0.5), -one(T), one(T))
    return x
end
function rademacher!(x::AbstractArray{Complex{T}}, buf::AbstractArray{T}=similar(x, T)) where {T<:Real}
    rand!(buf) # A real uniform draw picks one of the four quadrants per entry
    @. x = Complex(
        ifelse(buf < T(0.25), one(T), ifelse(buf < T(0.5), zero(T), ifelse(buf < T(0.75), -one(T), zero(T)))),
        ifelse(buf < T(0.25), zero(T), ifelse(buf < T(0.5), one(T), ifelse(buf < T(0.75), zero(T), -one(T)))),
    )
    return x
end

# The complex Rademacher needs a real scratch buffer (real one doesn't).
rademacher_buffer(::AbstractArray{<:Real}) = nothing
rademacher_buffer(x::AbstractArray{Complex{T}}) where {T} = similar(x, T)
draw_rademacher!(x::AbstractArray{<:Real}, _) = rademacher!(x)
draw_rademacher!(x::AbstractArray{<:Complex}, buf) = rademacher!(x, buf)

function hutchinson_trace_fixed(operator, num_samples::Integer, sample_vec::AbstractArray)
    num_samples >= 1 || throw(ArgumentError("num_samples must be positive"))
    T = eltype(operator)
    n = size(operator, 2)
    x = similar(sample_vec, T, n)
    buf = rademacher_buffer(x)
    # Welford running mean and sample variance of the (possibly complex) estimates
    μ = zero(T)
    M2 = zero(real(T))
    for k in 1:num_samples
        draw_rademacher!(x, buf)
        q = dot(x, operator * x)
        δ = q - μ
        μ += δ / k
        M2 += real(δ * conj(q - μ))
    end
    var = num_samples > 1 ? M2 / (num_samples - 1) : zero(real(T))
    return μ, sqrt(var / num_samples)
end

function hutchinson_trace_adaptive(operator, relative_tolerance::Real, min_samples::Integer, max_samples::Integer, sample_vec::AbstractArray)
    T = eltype(operator)
    n = size(operator, 2)
    x = similar(sample_vec, T, n)
    buf = rademacher_buffer(x)
    z = 1.96 # 95% central-limit half-width
    μ = zero(T)
    M2 = zero(real(T))
    k = 0
    err = zero(real(T))
    while k < max_samples
        k += 1
        draw_rademacher!(x, buf)
        q = dot(x, operator * x)
        δ = q - μ
        μ += δ / k
        M2 += real(δ * conj(q - μ))
        if k >= min_samples
            err = sqrt((M2 / (k - 1)) / k)
            z * err <= relative_tolerance * abs(μ) && return μ, err
        end
    end
    err = k > 1 ? sqrt((M2 / (k - 1)) / k) : zero(real(T))
    return μ, err
end

# Column-wise utilities for the small reduced (m x m) XTrace blocks.
cnormc(M) = M ./ sqrt.(sum(abs2, M; dims=1)) # normalize each column to unit norm
colnorm(M) = vec(sqrt.(sum(abs2, M; dims=1))) # 2-norm of each column
diag_prod(A, B) = vec(sum(conj.(A) .* B; dims=1)) # diagonal of A' * B

to_host(A) = Array(A) # bring a (small) reduced block to the CPU for dense work

# Draw m sphere-normalized Gaussian test vectors, Ω = sqrt(n) * cnormc(randn).
# XTrace's variance-reduction formulas assume isotropic columns, so this differs
# from the Rademacher vectors used by Hutchinson.
function sphere_test_matrix(operator, m::Integer, sample_vec::AbstractArray)
    T = eltype(operator)
    n = size(operator, 1)
    Ω = similar(sample_vec, T, n, m)
    randn!(Ω)
    return sqrt(real(T)(n)) .* cnormc(Ω)
end

# Core XTrace estimate from a test matrix Ω and its image Y = operator * Ω. This
# is a port of xtrace_helper.m (the "improved" branch) from Epperly, Tropp and
# Webber. randomized_range_finder is intentionally not reused here: XTrace needs
# the R factor and the raw Ω, not just Q, so it calls qrthin! directly.
function xtrace_estimate(operator, Ω, Y, sample_vec::AbstractArray)
    n = size(operator, 1)
    m = size(Ω, 2)
    Q, R = qrthin!(copy(materialize_mat(Y, sample_vec))) # copy so Y survives for adaptive reuse
    Z = materialize_mat(operator * Q, sample_vec)

    # The reduced blocks are m x m and tiny; finish on the host with dense LAPACK,
    # mirroring how reigen.jl handles its restricted problem. The expensive n x m
    # products stay on sample_vec's device.
    W = to_host(Q' * Ω)
    H = to_host(Q' * Z)
    Tm = to_host(Z' * Ω)
    Rh = to_host(R)
    S = cnormc(Matrix(adjoint(inv(UpperTriangular(Rh)))))

    HW = H * W
    dSW = diag_prod(S, W)
    dSHS = diag_prod(S, H * S)
    dTW = diag_prod(Tm, W)
    dWHW = diag_prod(W, HW)
    dSRmHW = diag_prod(S, Rh - HW)
    dTmHRS = diag_prod(Tm - H' * W, S)
    scale = (n - m + 1) ./ (n .- colnorm(W) .^ 2 .+ abs.(dSW .* colnorm(S)) .^ 2)

    ests = tr(H) .- dSHS .+ (.-dTW .+ dWHW .+ conj.(dSW) .* dSRmHW .+ abs.(dSW) .^ 2 .* dSHS .+ dTmHRS .* dSW) .* scale
    t = sum(ests) / m
    err = sqrt(sum(abs2, ests .- t) / (m - 1)) / sqrt(m) # jackknife error
    return t, err
end

function xtrace_fixed(operator, num_samples::Integer, sample_vec::AbstractArray)
    n = size(operator, 1)
    m = clamp(num_samples, 2, n) # XTrace needs at least 2, and at most n, test vectors
    Ω = sphere_test_matrix(operator, m, sample_vec)
    Y = materialize_mat(operator * Ω, sample_vec)
    return xtrace_estimate(operator, Ω, Y, sample_vec)
end

function xtrace_adaptive(operator, relative_tolerance::Real, max_samples::Integer, sample_vec::AbstractArray)
    n = size(operator, 1)
    cap = min(max_samples, n)
    m = clamp(10, 2, n) # initial sketch size
    Ω = sphere_test_matrix(operator, m, sample_vec)
    Y = materialize_mat(operator * Ω, sample_vec)
    t, err = xtrace_estimate(operator, Ω, Y, sample_vec)
    while err > relative_tolerance * abs(t) && m < cap
        add = min(m, cap - m) # double the sketch, capped
        Ωnew = sphere_test_matrix(operator, add, sample_vec)
        Ynew = materialize_mat(operator * Ωnew, sample_vec)
        Ω = hcat(Ω, Ωnew) # reuse the existing operator * Ω products
        Y = hcat(Y, Ynew)
        m += add
        t, err = xtrace_estimate(operator, Ω, Y, sample_vec)
    end
    return t, err
end
