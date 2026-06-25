# Randomized SVD of a matrix-free 2D blur operator.
#
# We never form the n²×n² blur matrix. `rsvd` only ever applies the operator and
# its adjoint. We compare the recovered singular values (and the low-rank
# approximation error) against a dense `svd` reference computed on the
# materialized matrix, which is only affordable here because n is small.
#
# Run with: julia --project=examples examples/01_rsvd_blur.jl

using MatrixFreeRandomizedLinearAlgebra
using LinearAlgebra, Plots, Printf, Random, LaTeXStrings

include(joinpath(@__DIR__, "operators.jl"))

Random.seed!(0)
const ASSETS = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(ASSETS)

# A slightly shifted kernel makes the operator non-symmetric (slightly more
# interesting singular value spectrum than for a symmetric operator).
n = 48
op = blur_operator(n; sigma = n / 10, shift = (1, 0))
N = n * n
sv = Float64[] # CPU prototype for the matrix-free operator
println("Operator: $N × $N matrix-free 2D blur (n = $n)")

# Dense reference (only because N is small enough to materialize here).
dense_op = densify(op)
Fexact = svd(dense_op)

k = 80
F = rsvd(op, k; num_oversamples = 40, num_power_iterations = 8, sample_vec = sv)
@printf("Largest singular value:  randomized %.6e  vs  exact %.6e\n", F.S[1], Fexact.S[1])
@printf("Relative error in the top %d singular values: %.3e\n",
        k, norm(F.S .- Fexact.S[1:k]) / norm(Fexact.S[1:k]))

# Plot 1: singular-value spectrum, randomized vs exact
win = 1:min(140, length(Fexact.S))
p1 = plot(win, Fexact.S[win];
          yscale = :log10, minorticks = true, minorgrid = true,
          label = "Exact (dense SVD)", 
          xlabel = "Index", ylabel = "Singular value",
          title = "Leading singular values of the blur operator",
          legend = :topright)
scatter!(p1, 1:k, F.S; label = "Randomized (rsvd)", ms = 4)
plot!(p1, dpi = 300)
savefig(p1, joinpath(ASSETS, "rsvd_spectrum.png"))

# Plot 2: low-rank approximation error vs target rank
# The best possible rank-r spectral-norm error is σ_{r+1}. We compare the
# randomized approximation against that optimum.
ranks = 1:3:70
rand_err = Float64[]
opt_err = Float64[]
for r in ranks
    Fr = rsvd(op, r; num_oversamples = 20, num_power_iterations = 6, sample_vec = sv)
    approx = Fr.U * Diagonal(Fr.S) * Fr.Vt
    push!(rand_err, opnorm(dense_op - approx) / Fexact.S[1])
    push!(opt_err, Fexact.S[r + 1] / Fexact.S[1])
end
p2 = plot(collect(ranks), opt_err;
          yscale = :log10, minorticks = true, minorgrid = true,
          label = "Optimal " * L"(σ_{r+1})",
          xlabel = "Target rank " * L"r", ylabel = "Relative spectral-norm error",
          title = "Low-rank approximation error", legend = :topright)
plot!(p2, collect(ranks), rand_err;
      label = "Randomized", lw = 2, ls = :dash, marker = :circle)
plot!(p2, dpi = 300)
savefig(p2, joinpath(ASSETS, "rsvd_error.png"))

println("Saved rsvd_spectrum.png and rsvd_error.png to docs/src/assets/")
