# Randomized eigenvalues of a matrix-free Hermitian operator.
#
# With a zero-shift (symmetric) kernel, the blur operator is Hermitian positive
# semidefinite, and its eigenvalues are exactly the Fourier transform of the
# kernel. We recover the leading eigenvalues with `reigvals_hermitian` and check
# them against both a dense `eigvals` reference and the analytic spectrum.
#
# Run with: julia --project=examples examples/02_reigen_hermitian.jl

using MatrixFreeRandomizedLinearAlgebra
using LinearAlgebra, Plots, Printf, Random, LaTeXStrings

include(joinpath(@__DIR__, "operators.jl"))

Random.seed!(0)
const ASSETS = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(ASSETS)

n = 48
op = blur_operator(n; sigma = n / 10) # zero shift => Hermitian PSD
N = n * n
sv = Float64[]
println("Operator: $N × $N matrix-free Hermitian blur (n = $n)")

dense_op = densify(op)
λexact = sort(eigvals(Symmetric(dense_op)); rev = true)

k = 80
λ = reigvals_hermitian(op, k; num_oversamples = 40, num_power_iterations = 8, sample_vec = sv)
@printf("Largest eigenvalue:  randomized %.6e  vs  exact %.6e\n", λ[1], λexact[1])
@printf("Relative error in the top %d eigenvalues: %.3e\n",
        k, norm(λ .- λexact[1:k]) / norm(λexact[1:k]))

# Plot 1: leading eigenvalues, randomized vs exact
win = 1:min(140, length(λexact))
p1 = plot(win, λexact[win];
          yscale = :log10, minorticks = true, minorgrid = true,
          label = "Exact (dense eigen)",
          xlabel = "Index", ylabel = "Eigenvalue",
          title = "Leading eigenvalues of the Hermitian blur operator",
          legend = :topright)
scatter!(p1, 1:k, λ; label = "Randomized ("*L"\mathtt{reigvals\_hermitian}"*")")
plot!(p1, dpi = 300)
savefig(p1, joinpath(ASSETS, "reigen_spectrum.png"))

# Plot 2: how power iteration improves per-eigenvalue accuracy
# Without power iteration the leading eigenvalues are still decent, but accuracy
# degrades toward the truncation rank. A few power iterations fix that, at the
# cost of extra passes over the operator.
ke = 100
floor_err = eps() # keep machine-zero errors off a log axis
p2 = plot(; yscale = :log10, minorticks = true, minorgrid = true,
          xlabel = "Eigenvalue index", ylabel = "Relative error",
          title = "Effect of power iterations on accuracy", legend = :topleft)
for q in (0, 2, 6)
    λq = reigvals_hermitian(op, ke; num_oversamples = 20, num_power_iterations = q, sample_vec = sv)
    relq = max.(abs.(λq .- λexact[1:ke]) ./ abs.(λexact[1:ke]), floor_err)
    plot!(p2, 1:ke, relq; lw = 2, label = L"\mathtt{num\_power\_iterations} = %$q")
end
plot!(p2, dpi = 300)
savefig(p2, joinpath(ASSETS, "reigen_error.png"))

println("Saved reigen_spectrum.png and reigen_error.png to docs/src/assets/")
