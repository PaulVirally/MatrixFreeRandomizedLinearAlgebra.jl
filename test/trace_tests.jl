using Random
using LinearAlgebra
using SparseArrays
using LinearMaps
using CUDA

using MatrixFreeRandomizedLinearAlgebra: rademacher!, cnormc, diag_prod, colnorm

# Build an n x n operator with a quickly decaying spectrum, the regime where
# XTrace is accurate. `herm` makes it Hermitian (real eigenvalues).
function decaying_operator(::Type{T}, n; decay=0.6, herm=false) where {T}
    U = Matrix(qr(randn(T, n, n)).Q)
    D = Diagonal(T.(exp.(-decay .* (0:n-1))))
    if herm
        return U * D * U'
    end
    V = Matrix(qr(randn(T, n, n)).Q)
    return U * D * V'
end

@testset "trace tests" begin
    @testset "XTrace dense real" begin
        Random.seed!(0xdeadbeef)
        n, m = 60, 30
        A = decaying_operator(Float64, n)
        tA = tr(A)

        t = trace(A, m)
        @test t isa Float64
        @test isapprox(t, tA; rtol=1e-2)
    end

    @testset "XTrace dense complex Hermitian and general" begin
        Random.seed!(0xdeadbeef)
        n, m = 60, 30

        H = decaying_operator(ComplexF64, n; herm=true)
        tH = tr(H)
        @test isapprox(imag(tH), 0; atol=1e-10) # Hermitian trace is real
        tHapprox = trace(H, m)
        @test tHapprox isa ComplexF64
        @test isapprox(tHapprox, tH; rtol=1e-2)

        G = decaying_operator(ComplexF64, n)
        tG = tr(G)
        @test isapprox(trace(G, m), tG; rtol=1e-2)
    end

    @testset "Hutchinson dense real (low_mem)" begin
        Random.seed!(0xdeadbeef)
        n = 80
        A = decaying_operator(Float64, n; herm=true)
        tA = tr(A)

        t, e = trace(A, 4000; low_mem=true, return_error=true)
        @test t isa Float64
        # within a few standard errors of the truth
        @test abs(t - tA) < 4 * e
        @test isapprox(t, tA; rtol=1e-1)
    end

    @testset "Hutchinson dense complex (low_mem)" begin
        Random.seed!(0xdeadbeef)
        n = 80
        A = decaying_operator(ComplexF64, n; herm=true)
        tA = tr(A)

        t, e = trace(A, 4000; low_mem=true, return_error=true)
        @test t isa ComplexF64
        @test abs(t - tA) < 4 * e
    end

    @testset "adaptive XTrace via relative_tolerance" begin
        Random.seed!(0xdeadbeef)
        n = 80
        A = decaying_operator(Float64, n)
        tA = tr(A)

        reltol = 1e-2
        res = trace(A; relative_tolerance=reltol, return_error=true)
        # the returned error meets the request (it did not bail out at the cap)
        @test res.error <= reltol * abs(res.value)
        # and the estimate is genuinely close
        @test isapprox(res.value, tA; rtol=1e-1)
    end

    @testset "adaptive Hutchinson via relative_tolerance" begin
        Random.seed!(0xdeadbeef)
        n = 60
        A = decaying_operator(Float64, n; herm=true)
        tA = tr(A)

        reltol = 5e-2
        max_samples = 200_000
        res = trace(A; relative_tolerance=reltol, low_mem=true, max_samples=max_samples, return_error=true)
        z = 1.96
        @test z * res.error <= reltol * abs(res.value)
        @test isapprox(res.value, tA; rtol=2e-1)
    end

    @testset "return_error toggle" begin
        Random.seed!(0xdeadbeef)
        n = 50
        A = decaying_operator(Float64, n)

        scalar = trace(A, 20)
        @test scalar isa Number

        nt = trace(A, 20; return_error=true)
        @test nt isa NamedTuple
        @test haskey(nt, :value) && haskey(nt, :error)
        @test nt.error >= 0
    end

    @testset "sparse matrix" begin
        Random.seed!(0xdeadbeef)
        n, m = 100, 40
        A = sprandn(n, n, 0.05) + 5I # shift so the trace is well away from zero
        tA = tr(A)

        sv = similar(Vector{Float64}, 0)
        t_xtrace = trace(A, m; sample_vec=sv)
        @test isapprox(t_xtrace, tA; rtol=2e-1)

        t_hutch = trace(A, 6000; low_mem=true, sample_vec=sv)
        @test isapprox(t_hutch, tA; rtol=5e-2)
    end

    @testset "LinearMaps.jl operator" begin
        Random.seed!(0xdeadbeef)
        n, m = 60, 30
        A = decaying_operator(Float64, n)
        L = LinearMap(A)
        tA = tr(A)

        sv = similar(A, eltype(A), 0)
        @test isapprox(trace(L, m; sample_vec=sv), tA; rtol=1e-1)
        @test isapprox(trace(L, 4000; low_mem=true, sample_vec=sv), tA; rtol=1e-1)
    end

    @testset "rademacher! and reduced helpers (CPU)" begin
        Random.seed!(0xdeadbeef)
        n = 200_000

        xr = Vector{Float64}(undef, n)
        rademacher!(xr)
        @test all(v -> v == 1.0 || v == -1.0, xr)
        @test abs(sum(xr) / n) < 0.05            # mean ≈ 0
        @test sum(abs2, xr) / n ≈ 1.0            # E|x|^2 = 1 exactly

        xc = Vector{ComplexF64}(undef, n)
        rademacher!(xc)
        units = (1.0 + 0im, 0 + 1.0im, -1.0 + 0im, 0 - 1.0im)
        @test all(v -> any(u -> v == u, units), xc)
        @test abs(sum(xc) / n) < 0.05
        @test sum(abs2, xc) / n ≈ 1.0

        # reduced m×m helpers used by XTrace
        M = randn(ComplexF64, 7, 4)
        N = randn(ComplexF64, 7, 4)
        @test diag_prod(M, N) ≈ diag(M' * N)
        @test colnorm(M) ≈ [norm(M[:, j]) for j in 1:size(M, 2)]
        @test all(j -> isapprox(norm(cnormc(M)[:, j]), 1.0), 1:size(M, 2))
    end

    @testset "argument checks" begin
        Random.seed!(0xdeadbeef)
        A = decaying_operator(Float64, 20)
        @test_throws DimensionMismatch trace(randn(3, 4), 5)
        @test_throws ArgumentError trace(A)                                   # no budget
        @test_throws ArgumentError trace(A, 5; relative_tolerance=1e-2)       # both
    end

    @testset "CUDA" begin
        if CUDA.functional()
            CUDA.allowscalar(false) # the device paths must not scalar-index
            Random.seed!(0xdeadbeef)
            n, m = 64, 24

            # real
            A = decaying_operator(Float32, n)
            dA = cu(A)
            tA = tr(A)
            t_x = trace(dA, m)
            @test isapprox(Float64(real(t_x)), real(tA); rtol=5e-2)
            t_h = trace(dA, 4000; low_mem=true)
            @test isapprox(Float64(real(t_h)), real(tA); rtol=1e-1)

            # complex
            B = decaying_operator(ComplexF32, n; herm=true)
            dB = cu(B)
            tB = tr(B)
            @test isapprox(trace(dB, m), ComplexF32(tB); rtol=5e-2)
            @test isapprox(trace(dB, 4000; low_mem=true), ComplexF32(tB); rtol=1e-1)

            # rademacher! on the device
            xr = CUDA.zeros(Float32, 10_000)
            rademacher!(xr)
            @test sum(abs2, xr) / length(xr) ≈ 1.0f0
            xc = CUDA.zeros(ComplexF32, 10_000)
            rademacher!(xc)
            @test sum(abs2, xc) / length(xc) ≈ 1.0f0

            CUDA.allowscalar(true)
        else
            @info "Skipping trace CUDA tests: CUDA not functional on this system"
        end
    end
end # @testset "trace tests"
