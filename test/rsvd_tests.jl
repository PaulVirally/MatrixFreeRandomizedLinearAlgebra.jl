using Random
using LinearAlgebra
using SparseArrays
using LinearMaps
using CUDA

@testset "rsvd tests" begin
    @testset "dense matrix" begin
        Random.seed!(0xdeadbeef)
        m, n, k = 50, 30, 5
        A = randn(m, n)

        svd_full = svd(A)
        svd_approx = rsvd(A, k; num_oversamples=10, num_power_iterations=3)

        @test size(svd_approx.U) == (m, k)
        @test size(svd_approx.Vt) == (k, n)
        @test length(svd_approx.S) == k

        # Compare singular values
        @test isapprox(svd_approx.S, svd_full.S[1:k]; rtol=1e-2, atol=1e-2)

        # Reconstruction error
        A_approx = svd_approx.U * Diagonal(svd_approx.S) * svd_approx.Vt
        @test norm(A - A_approx) < sum(svd_full.S[k+1:end])
    end

    @testset "sparse matrix" begin
        Random.seed!(0xdeadbeef)
        m, n, k = 60, 40, 5
        A_sparse = sprandn(m, n, 0.1)

        svd_full = svd(Matrix(A_sparse))
        svd_approx = rsvd(A_sparse, k; num_oversamples=10, num_power_iterations=3, sample_vec=similar(svd_full.U, eltype(svd_full.U), 0))

        @test size(svd_approx.U) == (m, k)
        @test size(svd_approx.Vt) == (k, n)
        @test length(svd_approx.S) == k

        @test isapprox(svd_approx.S, svd_full.S[1:k]; rtol=1e-1, atol=1e-1)

        A_approx = svd_approx.U * Diagonal(svd_approx.S) * svd_approx.Vt
        @test norm(Matrix(A_sparse) - A_approx) < sum(svd_full.S[k+1:end])
    end

    @testset "LinearMaps.jl operator" begin
        Random.seed!(0xdeadbeef)
        m, n, k = 40, 20, 5
        A = randn(m, n)

        # Build a LinearMap wrapping A
        L = LinearMap(A)
        svd_full = svd(A)

        svd_approx = rsvd(L, k; num_oversamples=10, num_power_iterations=3, sample_vec=similar(A, eltype(A), 0))

        @test size(svd_approx.U) == (m, k)
        @test size(svd_approx.Vt) == (k, n)

        @test isapprox(svd_approx.S, svd_full.S[1:k]; rtol=1e-2, atol=1e-2)
    end

    @testset "CUDA" begin
        if CUDA.functional()
            Random.seed!(0xdeadbeef)
            m, n, k = 64, 32, 8
            A = randn(Float32, m, n)
            dA = cu(A)

            # Default CUDA path
            svd_approx = rsvd(dA, k; num_oversamples=10, num_power_iterations=2)

            @test svd_approx.U isa CuArray
            @test svd_approx.Vt isa CuArray
            @test svd_approx.S isa AbstractVector
            @test length(svd_approx.S) == k

            # Bring back to CPU and compare singular values and reconstruction
            U_cpu = Array(svd_approx.U)
            S_cpu = Array(svd_approx.S)
            Vt_cpu = Array(svd_approx.Vt)

            A_approx = U_cpu * Diagonal(S_cpu) * Vt_cpu

            svd_full = svd(A)
            @test isapprox(S_cpu, svd_full.S[1:k]; rtol=2e-2, atol=2e-2)
            @test norm(A - A_approx) < sum(svd_full.S[k+1:end])

            # Explicit CUDA sample_vec: ensure everything stays on the GPU
            sample_vec = CUDA.zeros(Float32, m)
            svd_approx2 = rsvd(dA, k; num_oversamples=10, num_power_iterations=1, sample_vec=sample_vec)

            @test svd_approx2.U isa CuArray
            @test svd_approx2.Vt isa CuArray
            @test svd_approx2.S isa AbstractVector
            @test length(svd_approx2.S) == k
        else
            @info "Skipping rsvd CUDA tests: CUDA not functional on this system"
        end
    end
end # @testset "rsvd tests"
