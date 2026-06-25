# Stochastic trace estimation of a matrix-free Hermitian operator.
#
# tr(A) is estimated from matrix-vector products alone. We compare the two
# estimators the package offers -- XTrace (default) and the streaming
# Girard-Hutchinson estimator (low_mem=true) -- by their relative error as a
# function of the number of test vectors, averaged over many random trials.
#
# Run with: julia --project=examples examples/03_trace.jl

using MatrixFreeRandomizedLinearAlgebra
using LinearAlgebra, Plots, Printf, Random, Statistics, LaTeXStrings

include(joinpath(@__DIR__, "operators.jl"))

Random.seed!(0)
gr()
const ASSETS = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(ASSETS)

n = 48
A = blur_operator(n; sigma = n / 20) # Hermitian PSD
N = n * n
sv = Float64[]
tr_exact = tr(densify(A))
println("Operator: $N × $N matrix-free Hermitian blur (n = $n)")
@printf("Exact trace: %.6e\n", tr_exact)

# A single adaptive call, asking for the error estimate alongside the value.
res = trace(A; relative_tolerance = 1e-2, return_error = true, sample_vec = sv)
@printf("Adaptive XTrace: %.6e ± %.2e  (actual rel. error %.2e)\n",
        res.value, res.error, abs(res.value - tr_exact) / abs(tr_exact))

# Convergence study: relative error vs number of test vectors, averaged over
# many independent trials so the curves are smooth.
samples = [10, 20, 40, 80, 160, 320]
ntrials = 40
err_x = Float64[]
err_h = Float64[]
for m in samples
    ex = mean(abs(trace(A, m; sample_vec = sv) - tr_exact) for _ in 1:ntrials)
    eh = mean(abs(trace(A, m; low_mem = true, sample_vec = sv) - tr_exact) for _ in 1:ntrials)
    push!(err_x, ex / abs(tr_exact))
    push!(err_h, eh / abs(tr_exact))
    @printf("m = %4d   XTrace %.3e   Hutchinson %.3e\n", m, err_x[end], err_h[end])
end

# Reference 1/sqrt(m) Monte-Carlo rate fit to the Hutchinson curve.
mc = err_h[1] .* sqrt(samples[1]) ./ sqrt.(samples)

p = plot(samples, err_h;
         xscale = :log10, yscale = :log10, marker = :circle,
         label = "Hutchinson (low_mem)",
         xlabel = "Number of test vectors", ylabel = "Mean relative error",
         title = "Trace estimation convergence", legend = :bottomleft)
plot!(p, samples, err_x; label = "XTrace (default)", lw = 2, marker = :diamond, color = :purple)
plot!(p, samples, mc; label = L"\frac{1}{\sqrt{m}}" * " reference", ls = :dot, color = :gray)
plot!(p, dpi = 300)
savefig(p, joinpath(ASSETS, "trace_convergence.png"))

println("Saved trace_convergence.png to docs/src/assets/")
