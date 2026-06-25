# Performance: matrix-free randomized SVD vs forming the dense matrix.
#
# For each problem size we time two ways of getting the leading singular values:
#   1. materialize the n²×n² operator, then call the dense `svdvals`;
#   2. call rsvdvals directly on the matrix-free operator.
# The dense route is only run for the smaller sizes because it is cubic in N and the
# matrix itself grows quadratically.
#
# Run with: julia --project=examples examples/04_performance.jl

using MatrixFreeRandomizedLinearAlgebra
using LinearAlgebra, Plots, Printf, Random, LaTeXStrings

include(joinpath(@__DIR__, "operators.jl"))

Random.seed!(0)
gr()
const ASSETS = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(ASSETS)

# Take the min over a few runs after a warmup, to dodge one-off noise without
# the full cost of a BenchmarkTools sweep at the largest sizes.
function timed(f; runs = 3)
    f() # warmup
    minimum(@elapsed(f()) for _ in 1:runs)
end

k = 20 # leading singular values to recover
dense_ns = [16, 24, 32, 48, 64] # materialize + dense svd up to here
mf_ns = [16, 24, 32, 48, 64, 96, 128] # matrix-free keeps going

dense_N = Int[];
dense_t = Float64[]
mf_N = Int[];
mf_t = Float64[]

println(" n      N     dense (s)   matrix-free (s)")
for n in mf_ns
    A = blur_operator(n; sigma = n / 20, shift = (1, 0))
    N = n * n
    sv = Float64[]

    tmf = timed(() -> rsvdvals(A, k; num_oversamples = 10, num_power_iterations = 4, sample_vec = sv))
    push!(mf_N, N); push!(mf_t, tmf)

    if n in dense_ns
        td = timed(() -> svdvals(densify(A)))
        push!(dense_N, N); push!(dense_t, td)
        @printf("%3d  %6d   %9.4f   %12.4f\n", n, N, td, tmf)
    else
        @printf("%3d  %6d   %9s   %12.4f\n", n, N, "-", tmf)
    end
end

p = plot(dense_N, dense_t;
         xscale = :log10, yscale = :log10, marker = :square,
         label = "dense (materialize + svd)",
         xlabel = "operator dimension " *L"N (= n^2)", ylabel = "time (s)",
         title = "Leading $k singular values: dense vs matrix-free",
         legend = :topleft)
plot!(p, mf_N, mf_t; label = "matrix-free (rsvdvals)", marker = :circle)
plot!(p, dpi = 300)
savefig(p, joinpath(ASSETS, "performance.png"))

println("Saved performance.png to docs/src/assets/")
