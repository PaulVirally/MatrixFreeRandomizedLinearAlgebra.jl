# Benchmark suite for MatrixFreeRandomizedLinearAlgebra.jl.
#
# Covers every exported randomized algorithm (rsvd, rsvdvals, reigen_hermitian,
# reigvals_hermitian, trace in all four modes) across representative operator
# types (dense, sparse, matrix-free LinearMap), plus the internal kernels they
# are built on (qthin!, qrthin!, sphere_test_matrix, rademacher!), plus a
# "panel" group measuring Funicular's tiered panel storage against the dense
# path on the same operator.
#
# When CUDA is functional, a mirrored set of GPU benchmarks (groups prefixed
# "gpu-") is added: dense CuMatrix, CUSPARSE sparse, and a CUFFT-based blur
# LinearMap. On machines without a GPU those groups are silently absent.
#
# Because this package targets operators so large that even a few extra
# n-vectors matter, the *memory estimate* column is as important as time. For
# GPU benchmarks BenchmarkTools only sees host allocations, so device memory is
# measured separately (one probe call per benchmark via CUDA.@timed) and shown
# in its own column / saved to a `<label>.gpumem.tsv` sidecar for compare.jl.
#
# Run with:
#   julia --project=benchmark benchmark/benchmarks.jl [label]
#
# Results are printed as a table and saved to benchmark/results/<label>.json
# (BenchmarkTools format) for later comparison with benchmark/compare.jl.

using BenchmarkTools
using CUDA
using Funicular
using LinearAlgebra
using LinearMaps
using FFTW
using Printf
using Random
using SparseArrays
using MatrixFreeRandomizedLinearAlgebra
const MFRLA = MatrixFreeRandomizedLinearAlgebra

BLAS.set_num_threads(Sys.CPU_THREADS ÷ 2)

Random.seed!(0xdeadbeef)

# Dense wide and tall operators (tall exercises the internal transpose path).
const A_wide = randn(2048, 4096)
const A_tall = Matrix(A_wide')

# Sparse square operator, ~10 nnz per row.
const A_sparse = sprandn(50_000, 50_000, 10 / 50_000)

# Sparse Hermitian operator for the eigen routines.
const A_herm_sparse = Symmetric(A_sparse + A_sparse')

# Dense Hermitian operator.
const _G = randn(3072, 3072)
const A_herm_dense = Symmetric(_G + _G')

# Matrix-free 2D periodic Gaussian blur (FFT-based LinearMap), n² × n².
function blur_khat(n::Int; sigma::Real=n / 16, shift::Tuple{Int,Int}=(0, 0))
    c = n ÷ 2 + 1
    g = [exp(-((i - c)^2) / (2 * sigma^2)) for i in 1:n]
    psf = g * g'
    psf ./= sum(psf)
    psf = ifftshift(psf)
    shift == (0, 0) || (psf = circshift(psf, shift))
    return fft(psf)
end

function blur_operator(n::Int; sigma::Real=n / 16, shift::Tuple{Int,Int}=(0, 0))
    khat = blur_khat(n; sigma=sigma, shift=shift)
    N = n * n
    fwd(x) = vec(real(ifft(khat .* fft(reshape(x, n, n)))))
    adj(y) = vec(real(ifft(conj.(khat) .* fft(reshape(y, n, n)))))
    return LinearMap{Float64}(fwd, adj, N, N)
end

const L_blur = blur_operator(128; shift=(1, 0))      # non-symmetric, 16384²
const L_blur_herm = blur_operator(128)               # Hermitian PSD, 16384²

const k = 20 # target rank / components
const p = 10 # oversamples
const q = 4  # power iterations (fixed so runtimes are comparable)

const sv = Float64[] # CPU prototype sample_vec

const SUITE = BenchmarkGroup()

SUITE["kernels"] = BenchmarkGroup()
SUITE["kernels"]["qthin!_tall_8192x64"] =
    @benchmarkable MFRLA.qthin!(A) setup = (A = randn(8192, 64)) evals = 1
SUITE["kernels"]["qrthin!_tall_8192x64"] =
    @benchmarkable MFRLA.qrthin!(A) setup = (A = randn(8192, 64)) evals = 1
SUITE["kernels"]["qthin!_wide_64x8192"] =
    @benchmarkable MFRLA.qthin!(A) setup = (A = randn(64, 8192)) evals = 1
SUITE["kernels"]["qrthin!_wide_64x8192"] =
    @benchmarkable MFRLA.qrthin!(A) setup = (A = randn(64, 8192)) evals = 1
SUITE["kernels"]["sphere_test_matrix_50000x64"] =
    @benchmarkable MFRLA.sphere_test_matrix(A_sparse, 64, sv)
SUITE["kernels"]["rademacher!_real_1M"] =
    @benchmarkable MFRLA.rademacher!(x) setup = (x = Vector{Float64}(undef, 1_000_000))
SUITE["kernels"]["rademacher!_complex_1M"] =
    @benchmarkable MFRLA.rademacher!(x, buf) setup = (x = Vector{ComplexF64}(undef, 1_000_000); buf = Vector{Float64}(undef, 1_000_000))
SUITE["kernels"]["range_finder_sparse_50000_k30_q4"] =
    @benchmarkable MFRLA.randomized_range_finder(A_sparse, k + p, q, sv) setup = (Random.seed!(7))

SUITE["rsvd"] = BenchmarkGroup()
SUITE["rsvd"]["dense_wide_2048x4096"] =
    @benchmarkable rsvd(A_wide, k; num_oversamples=p, num_power_iterations=q, sample_vec=sv) setup = (Random.seed!(7))
SUITE["rsvd"]["dense_tall_4096x2048"] =
    @benchmarkable rsvd(A_tall, k; num_oversamples=p, num_power_iterations=q, sample_vec=sv) setup = (Random.seed!(7))
SUITE["rsvd"]["sparse_50000"] =
    @benchmarkable rsvd(A_sparse, k; num_oversamples=p, num_power_iterations=q, sample_vec=sv) setup = (Random.seed!(7))
SUITE["rsvd"]["blur_map_16384"] =
    @benchmarkable rsvd(L_blur, k; num_oversamples=p, num_power_iterations=q, sample_vec=sv) setup = (Random.seed!(7))

SUITE["rsvdvals"] = BenchmarkGroup()
SUITE["rsvdvals"]["dense_wide_2048x4096"] =
    @benchmarkable rsvdvals(A_wide, k; num_oversamples=p, num_power_iterations=q, sample_vec=sv) setup = (Random.seed!(7))
SUITE["rsvdvals"]["sparse_50000"] =
    @benchmarkable rsvdvals(A_sparse, k; num_oversamples=p, num_power_iterations=q, sample_vec=sv) setup = (Random.seed!(7))

SUITE["reigen"] = BenchmarkGroup()
SUITE["reigen"]["sparse_herm_50000"] =
    @benchmarkable reigen_hermitian(A_herm_sparse, k; num_oversamples=p, num_power_iterations=q, sample_vec=sv) setup = (Random.seed!(7))
SUITE["reigen"]["dense_herm_3072"] =
    @benchmarkable reigen_hermitian(A_herm_dense, k; num_oversamples=p, num_power_iterations=q, sample_vec=sv) setup = (Random.seed!(7))
SUITE["reigen"]["blur_herm_16384"] =
    @benchmarkable reigen_hermitian(L_blur_herm, k; num_oversamples=p, num_power_iterations=q, sample_vec=sv) setup = (Random.seed!(7))

SUITE["reigvals"] = BenchmarkGroup()
SUITE["reigvals"]["sparse_herm_50000"] =
    @benchmarkable reigvals_hermitian(A_herm_sparse, k; num_oversamples=p, num_power_iterations=q, sample_vec=sv) setup = (Random.seed!(7))

SUITE["trace"] = BenchmarkGroup()
SUITE["trace"]["xtrace_fixed_sparse_50000_m64"] =
    @benchmarkable trace(A_sparse, 64; sample_vec=sv) setup = (Random.seed!(7))
SUITE["trace"]["xtrace_fixed_dense_3072_m64"] =
    @benchmarkable trace(A_herm_dense, 64; sample_vec=sv) setup = (Random.seed!(7))
SUITE["trace"]["xtrace_adaptive_dense_3072_rtol1e-2"] =
    @benchmarkable trace(A_herm_dense; relative_tolerance=1e-2, sample_vec=sv) setup = (Random.seed!(7))
SUITE["trace"]["hutchinson_fixed_sparse_50000_128"] =
    @benchmarkable trace(A_sparse, 128; low_mem=true, sample_vec=sv) setup = (Random.seed!(7))
SUITE["trace"]["hutchinson_fixed_dense_3072_128"] =
    @benchmarkable trace(A_herm_dense, 128; low_mem=true, sample_vec=sv) setup = (Random.seed!(7))

# ---------------------------------------------------------------------------
# Panel path (Funicular). What is measured here is the streaming overhead, so
# every solve appears twice on the *same* operator: once through the panel path
# (`plan=`) and once through the plain dense path. The pair of medians is the
# number to watch, and compare.jl tracks both over time, so a change that makes
# the panel path slower relative to the dense one shows up as a widening gap
# rather than as noise on an absolute time.
#
# The budgets are synthetic. A CPUBackend plan has no real device to run out of,
# so a device_budget of 2 MiB plays the part of one: the 2000 × 200 Float64
# sketch is 3.05 MiB, half again as much, which is what forces panels to be
# streamed and evicted instead of all living resident. The panel width is pinned
# at 10 rather than left to the budget arithmetic so that the cut is the same on
# every machine: 20 panels of the sketch, every one of them moved on every sweep.
#
# `validate=false` skips Funicular's `check_operator` probes, which the dense
# path has no counterpart for and which would otherwise sit in the ratio as a
# constant. `seed=7` pins the regenerated test matrix the way `Random.seed!(7)`
# pins the dense path's stored one.
# ---------------------------------------------------------------------------
const panel_n = 2000 # operator dimension
const panel_k = 100  # components
const panel_p = 100  # oversamples, so a 2000 × 200 sketch
const panel_q = 2    # power iterations (fixed, and low, to keep the sweep short)

const A_panel = randn(panel_n, panel_n)
const A_panel_herm = Symmetric(A_panel + A_panel')

panel_plan() = ResidencyPlan(; backend=CPUBackend(), device_budget=2 * 2^20,
    host_budget=64 * 2^20, panel_width=10)

# One plan per benchmark, reused across samples. That only works because the
# vectors a solve returns belong to the caller and nothing else frees them: left
# alone they would accumulate in the plan's host tier until it overflowed
# mid-sweep, so each wrapper frees what it was handed.
const panel_plan_eigen = panel_plan()
const panel_plan_rsvd = panel_plan()

function panel_reigen_solve(plan)
    E = reigen_hermitian(A_panel_herm, panel_k; num_oversamples=panel_p,
        num_power_iterations=panel_q, plan=plan, seed=7, validate=false)
    Funicular.free!(E.vectors)
    return nothing
end

function panel_rsvd_solve(plan)
    F = rsvd(A_panel, panel_k; num_oversamples=panel_p,
        num_power_iterations=panel_q, plan=plan, seed=7, validate=false)
    Funicular.free!(F.U)
    Funicular.free!(F.V)
    return nothing
end

SUITE["panel"] = BenchmarkGroup()
SUITE["panel"]["reigen_herm_2000_k100_dense"] =
    @benchmarkable reigen_hermitian(A_panel_herm, panel_k; num_oversamples=panel_p, num_power_iterations=panel_q, sample_vec=sv) setup = (Random.seed!(7)) evals = 1 samples = 30
SUITE["panel"]["reigen_herm_2000_k100_panel"] =
    @benchmarkable panel_reigen_solve(panel_plan_eigen) evals = 1 samples = 30
SUITE["panel"]["rsvd_2000_k100_dense"] =
    @benchmarkable rsvd(A_panel, panel_k; num_oversamples=panel_p, num_power_iterations=panel_q, sample_vec=sv) setup = (Random.seed!(7)) evals = 1 samples = 30
SUITE["panel"]["rsvd_2000_k100_panel"] =
    @benchmarkable panel_rsvd_solve(panel_plan_rsvd) evals = 1 samples = 30

# ---------------------------------------------------------------------------
# GPU benchmarks (only when CUDA is functional). Groups are prefixed "gpu-" so
# CPU-only and GPU runs can share result files; compare.jl simply skips keys
# that one side lacks.
#
# BenchmarkTools cannot see device allocations, so each GPU benchmark also
# registers a probe thunk in GPU_MEM_PROBES; run_and_report calls it once under
# CUDA.@timed after the timing sweep and reports the device bytes it allocated.
# Every thunk is self-contained (it creates any input it mutates), which lets
# run_and_report reuse it and lets us smoke-test each benchmark once up front —
# an unsupported combination is skipped with a warning instead of killing the
# whole sweep.
# ---------------------------------------------------------------------------
const GPU_MEM_PROBES = Dict{String,Any}()

function add_gpu_benchmark!(group::String, name::String, thunk)
    try
        CUDA.@sync thunk() # smoke test; also serves as a warmup
    catch err
        @warn "Skipping GPU benchmark $group/$name (unsupported on this setup)" exception = (err, catch_backtrace())
        return
    end
    haskey(SUITE, group) || (SUITE[group] = BenchmarkGroup())
    # CUDA.@sync so asynchronous kernels are fully timed.
    SUITE[group][name] = @benchmarkable CUDA.@sync($thunk()) setup = (Random.seed!(7); CUDA.seed!(7)) evals = 1
    GPU_MEM_PROBES["$group/$name"] = thunk
    return
end

if CUDA.functional()
    @info "CUDA functional; adding GPU benchmarks" device = CUDA.name(CUDA.device())

    dsv = CUDA.zeros(Float64, 0) # GPU prototype sample_vec
    dA_wide = CuArray(A_wide)
    dA_tall = CuArray(A_tall)
    dA_sparse = CUDA.CUSPARSE.CuSparseMatrixCSR(A_sparse)
    dA_herm_sparse = CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A_herm_sparse))
    dA_herm_dense = CuArray(Matrix(A_herm_dense)) # plain CuMatrix, numerically Hermitian

    # CUFFT-based blur LinearMap on the device. The incoming column may be a
    # SubArray view, so it is copied into a dense CuArray before the FFT
    # (CUFFT requires dense device arrays).
    function gpu_blur_operator(n::Int; sigma::Real=n / 16, shift::Tuple{Int,Int}=(0, 0))
        khat = CuArray(blur_khat(n; sigma=sigma, shift=shift))
        N = n * n
        fwd(x) = vec(real(ifft(khat .* fft(reshape(CuArray{Float64}(x), n, n)))))
        adj(y) = vec(real(ifft(conj.(khat) .* fft(reshape(CuArray{Float64}(y), n, n)))))
        return LinearMap{Float64}(fwd, adj, N, N)
    end
    dL_blur = gpu_blur_operator(128; shift=(1, 0))
    dL_blur_herm = gpu_blur_operator(128)

    # Kernels. The QR thunks generate their own input on the device, so a few
    # hundred microseconds of randn! are included in their timings.
    add_gpu_benchmark!("gpu-kernels", "qthin!_tall_8192x64", () -> MFRLA.qthin!(CUDA.randn(Float64, 8192, 64)))
    add_gpu_benchmark!("gpu-kernels", "qrthin!_tall_8192x64", () -> MFRLA.qrthin!(CUDA.randn(Float64, 8192, 64)))
    add_gpu_benchmark!("gpu-kernels", "qthin!_wide_64x8192", () -> MFRLA.qthin!(CUDA.randn(Float64, 64, 8192)))
    add_gpu_benchmark!("gpu-kernels", "qrthin!_wide_64x8192", () -> MFRLA.qrthin!(CUDA.randn(Float64, 64, 8192)))
    add_gpu_benchmark!("gpu-kernels", "sphere_test_matrix_50000x64", () -> MFRLA.sphere_test_matrix(dA_sparse, 64, dsv))
    add_gpu_benchmark!("gpu-kernels", "rademacher!_real_1M", () -> MFRLA.rademacher!(CuVector{Float64}(undef, 1_000_000)))
    add_gpu_benchmark!("gpu-kernels", "rademacher!_complex_1M", () -> MFRLA.rademacher!(CuVector{ComplexF64}(undef, 1_000_000)))
    add_gpu_benchmark!("gpu-kernels", "range_finder_sparse_50000_k30_q4", () -> MFRLA.randomized_range_finder(dA_sparse, k + p, q, dsv))

    # rsvd / rsvdvals
    add_gpu_benchmark!("gpu-rsvd", "dense_wide_2048x4096", () -> rsvd(dA_wide, k; num_oversamples=p, num_power_iterations=q, sample_vec=dsv))
    add_gpu_benchmark!("gpu-rsvd", "dense_tall_4096x2048", () -> rsvd(dA_tall, k; num_oversamples=p, num_power_iterations=q, sample_vec=dsv))
    add_gpu_benchmark!("gpu-rsvd", "sparse_50000", () -> rsvd(dA_sparse, k; num_oversamples=p, num_power_iterations=q, sample_vec=dsv))
    add_gpu_benchmark!("gpu-rsvd", "blur_map_16384", () -> rsvd(dL_blur, k; num_oversamples=p, num_power_iterations=q, sample_vec=dsv))
    add_gpu_benchmark!("gpu-rsvdvals", "dense_wide_2048x4096", () -> rsvdvals(dA_wide, k; num_oversamples=p, num_power_iterations=q, sample_vec=dsv))
    add_gpu_benchmark!("gpu-rsvdvals", "sparse_50000", () -> rsvdvals(dA_sparse, k; num_oversamples=p, num_power_iterations=q, sample_vec=dsv))

    # reigen / reigvals
    add_gpu_benchmark!("gpu-reigen", "sparse_herm_50000", () -> reigen_hermitian(dA_herm_sparse, k; num_oversamples=p, num_power_iterations=q, sample_vec=dsv))
    add_gpu_benchmark!("gpu-reigen", "dense_herm_3072", () -> reigen_hermitian(dA_herm_dense, k; num_oversamples=p, num_power_iterations=q, sample_vec=dsv))
    add_gpu_benchmark!("gpu-reigen", "blur_herm_16384", () -> reigen_hermitian(dL_blur_herm, k; num_oversamples=p, num_power_iterations=q, sample_vec=dsv))
    add_gpu_benchmark!("gpu-reigvals", "sparse_herm_50000", () -> reigvals_hermitian(dA_herm_sparse, k; num_oversamples=p, num_power_iterations=q, sample_vec=dsv))

    # trace
    add_gpu_benchmark!("gpu-trace", "xtrace_fixed_sparse_50000_m64", () -> trace(dA_sparse, 64; sample_vec=dsv))
    add_gpu_benchmark!("gpu-trace", "xtrace_fixed_dense_3072_m64", () -> trace(dA_herm_dense, 64; sample_vec=dsv))
    add_gpu_benchmark!("gpu-trace", "xtrace_adaptive_dense_3072_rtol1e-2", () -> trace(dA_herm_dense; relative_tolerance=1e-2, sample_vec=dsv))
    add_gpu_benchmark!("gpu-trace", "hutchinson_fixed_sparse_50000_128", () -> trace(dA_sparse, 128; low_mem=true, sample_vec=dsv))
    add_gpu_benchmark!("gpu-trace", "hutchinson_fixed_dense_3072_128", () -> trace(dA_herm_dense, 128; low_mem=true, sample_vec=dsv))
end

function run_and_report(label::AbstractString)
    println("Tuning + running benchmarks (label = $label) ...")
    results = run(SUITE; verbose=true, seconds=5)

    resdir = joinpath(@__DIR__, "results")
    mkpath(resdir)
    BenchmarkTools.save(joinpath(resdir, "$label.json"), results)

    # Device memory: one probe call per GPU benchmark (everything is warm after
    # the timing sweep). `gpu_bytes` counts allocations the call performed, the
    # analogue of BenchmarkTools' host memory estimate.
    gpumem = Dict{String,Int}()
    for (key, thunk) in GPU_MEM_PROBES
        stats = CUDA.@timed thunk()
        gpumem[key] = stats.gpu_bytes
    end
    if !isempty(gpumem)
        open(joinpath(resdir, "$label.gpumem.tsv"), "w") do io
            for (key, bytes) in sort(collect(gpumem))
                println(io, "$key\t$bytes")
            end
        end
    end

    println()
    @printf("%-45s %12s %12s %10s %12s\n", "benchmark", "median time", "cpu memory", "allocs", "gpu memory")
    println("-"^96)
    for group in sort(collect(keys(results)))
        for name in sort(collect(keys(results[group])))
            t = results[group][name]
            key = "$group/$name"
            @printf("%-45s %12s %12s %10d %12s\n",
                key,
                BenchmarkTools.prettytime(time(median(t))),
                BenchmarkTools.prettymemory(memory(t)),
                allocs(t),
                haskey(gpumem, key) ? BenchmarkTools.prettymemory(gpumem[key]) : "-")
        end
    end
    println("\nSaved to $(joinpath(resdir, "$label.json"))")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    label = isempty(ARGS) ? "baseline" : ARGS[1]
    run_and_report(label)
end
