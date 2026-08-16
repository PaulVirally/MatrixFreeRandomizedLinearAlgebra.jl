using LinearAlgebra

# The panel path: when a caller passes a Funicular `ResidencyPlan` through the
# `plan` keyword, the N × s sketch matrices are held as panel matrices streamed
# through the device rather than as dense arrays kept in memory all at once. The
# result containers and the routing hooks live here rather than in the extension
# so that their names are always defined and exportable. The methods that build
# and consume them come with Funicular.jl.

"""
    PanelEigen(values, vectors)

Eigendecomposition returned by [`reigen_hermitian`](@ref) on the panel path,
the counterpart of `LinearAlgebra.Eigen` for a factorization whose eigenvectors
are too tall to hold in memory as a dense array.

`values` is a plain host `Vector` of eigenvalues in descending order. `vectors`
is a Funicular `PanelMatrix` with one column per eigenvalue, or, when
`reigen_hermitian` was called with `factored=true`, a [`PanelFactored`](@ref):
the product that would build it, left unevaluated.
"""
struct PanelEigen{TL,TV}
    values::TL
    vectors::TV
end

"""
    PanelSVD(U, S, V)

Singular value decomposition returned by `rsvd` on the panel path, the
counterpart of `LinearAlgebra.SVD` for a factorization whose singular vectors
are too tall to hold in memory as a dense array.

`S` is a plain host `Vector` of singular values in descending order. `U`
(`m × k`) and `V` (`n × k`) are Funicular `PanelMatrix` objects, or
[`PanelFactored`](@ref) products when `rsvd` was called with `factored=true`.

Unlike `LinearAlgebra.SVD`, this holds the *tall* `V` rather than the wide
`Vt`: a `k × n` matrix is short and wide, and a `PanelMatrix` is cut into
column panels of a tall one, so `Vt` has no panel representation. Reconstruct
with `Matrix(U) * Diagonal(S) * Matrix(V)'` when the dense factors fit, or keep
applying the three factors in turn when they do not.
"""
struct PanelSVD{TU,TS,TV}
    U::TU
    S::TS
    V::TV
end

"""
    PanelFactored(Q, C)

The product `Q * C` of a tall Funicular `PanelMatrix` basis `Q` and a small
dense host factor `C`, left unevaluated.

Forming the product costs a full sweep of `Q` and an `N × size(C, 2)` panel
matrix to land in. When the caller only means to apply the result to something
else, or to feed it back as a `seed_Q`, the two factors are cheaper to hold
than the product. [`materialize`](@ref) performs the multiplication and
`Matrix` collects the result to the host.
"""
struct PanelFactored{TQ,TC}
    Q::TQ
    C::TC
end

Base.size(f::PanelFactored) = (size(f.Q, 1), size(f.C, 2))
function Base.size(f::PanelFactored, d::Integer)
    d >= 1 || throw(ArgumentError("dimension must be positive, got $d"))
    d <= 2 ? size(f)[d] : 1
end
Base.eltype(f::PanelFactored) = promote_type(eltype(f.Q), eltype(f.C))

"""
    materialize(f::PanelFactored)

Evaluate the deferred product `f.Q * f.C` into a new Funicular `PanelMatrix`
built from `f.Q`'s plan, in one sweep over the rows of `f.Q`. Needs Funicular.jl
loaded.
"""
materialize(f::PanelFactored) = throw(funicular_not_loaded("materialize"))

function Base.show(io::IO, F::PanelEigen)
    print(io, "PanelEigen(", length(F.values), " values, ")
    show(io, F.vectors)
    print(io, ")")
end
function Base.show(io::IO, mime::MIME"text/plain", F::PanelEigen)
    summary(io, F)
    println(io)
    println(io, "values:")
    show(io, mime, F.values)
    println(io)
    println(io, "vectors:")
    show(io, mime, F.vectors)
end

function Base.show(io::IO, F::PanelSVD)
    print(io, "PanelSVD(", length(F.S), " values, ")
    show(io, F.U)
    print(io, ", ")
    show(io, F.V)
    print(io, ")")
end
function Base.show(io::IO, mime::MIME"text/plain", F::PanelSVD)
    summary(io, F)
    println(io)
    println(io, "U factor:")
    show(io, mime, F.U)
    println(io)
    println(io, "singular values:")
    show(io, mime, F.S)
    println(io)
    println(io, "V factor:")
    show(io, mime, F.V)
end

function Base.show(io::IO, f::PanelFactored)
    m, n = size(f)
    print(io, "PanelFactored(", m, "×", n, ", ")
    show(io, f.Q)
    print(io, " * ", size(f.C, 1), "×", size(f.C, 2), " ", eltype(f.C), ")")
end
function Base.show(io::IO, mime::MIME"text/plain", f::PanelFactored)
    summary(io, f)
    println(io)
    println(io, "Q factor:")
    show(io, mime, f.Q)
    println(io)
    println(io, "C factor:")
    show(io, mime, f.C)
end

# Routing hooks. Every public entry point forwards to one of these when it is
# given a `plan`; the extension adds the methods that dispatch on Funicular's
# `ResidencyPlan`. These fallbacks are what a caller who never loaded Funicular
# gets, and they are deliberately more general than the extension's methods, so
# that the extension's methods always win.
reigen_hermitian_panel(operator, num_components::Int, plan; kwargs...) = throw(funicular_not_loaded("reigen_hermitian"))
reigvals_hermitian_panel(operator, num_components::Int, plan; kwargs...) = throw(funicular_not_loaded("reigvals_hermitian"))
rsvd_panel(operator, num_components::Int, plan; kwargs...) = throw(funicular_not_loaded("rsvd"))
rsvdvals_panel(operator, num_components::Int, plan; kwargs...) = throw(funicular_not_loaded("rsvdvals"))
trace_panel(operator, num_samples, plan; kwargs...) = throw(funicular_not_loaded("trace"))

function funicular_not_loaded(name::AbstractString)
    return ArgumentError("`$name` was given a `plan`, which routes the computation through Funicular.jl's tiered panel storage, but Funicular.jl is not loaded. Add Funicular.jl to the environment and `using Funicular` before passing `plan`, or drop the keyword to run in memory")
end

# The panel path exists to stream a sketch that is too tall to hold, and the
# Girard-Hutchinson estimator has no sketch: it holds one or two vectors at a
# time, which is already the smallest footprint there is. There is nothing for
# the plan to do here, so the combination is refused instead of being ignored.
function trace_low_mem_with_plan()
    return ArgumentError("`trace` was given both `low_mem=true` and a `plan`, but the two do not fit together: `low_mem` selects the streaming Girard-Hutchinson estimator, which only ever holds one or two vectors at a time, so there is no sketch for Funicular's tiered panel storage to stream. Drop `plan` to run Hutchinson in memory (passing `sample_vec` if those vectors should live on a device), or drop `low_mem` to run XTrace on the panel path")
end

function seed_without_plan(name::AbstractString)
    return ArgumentError("`seed` only applies to the panel path, where it seeds Funicular's regenerated test matrix, but `$name` was called without a `plan`. Pass `plan` as well, or seed the global RNG with `Random.seed!` to make the in-memory path reproducible")
end

function factored_without_plan(name::AbstractString)
    return ArgumentError("`factored=true` only applies to the panel path, where it defers the basis rotations that would build the vectors, but `$name` was called without a `plan`. Pass `plan` as well, or drop `factored`")
end

# Without a seed we draw a fresh sketch on every call, as the in-memory path
# does when it draws the test matrix from the global RNG. An explicit integer
# makes the sketch reproducible instead, and reproducible across panel widths
# and plan budgets too, since Funicular generates a ghost column from its index
# and the seed alone. This is only ever called on the panel path, so the RNG is
# untouched otherwise.
resolve_panel_seed(seed) = seed === nothing ? rand(UInt64) : seed

# The `sample_vec` default is resolved here rather than in the signature so that
# `similar(operator, ...)` is never evaluated on the panel path, where the
# operator only has to satisfy Funicular's contract and need not be an array at
# all.
resolve_sample_vec(operator, sample_vec) = sample_vec === nothing ? similar(operator, eltype(operator), 0) : sample_vec
