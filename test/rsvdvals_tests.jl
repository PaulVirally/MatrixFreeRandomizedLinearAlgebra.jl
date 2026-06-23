using Random
using LinearAlgebra
using SparseArrays
using LinearMaps
using CUDA

@testset "rsvdvals tests" begin
    @testset "dense matrix" begin
        Random.seed!(0xdeadbeef)
        m, n, k = 50, 30, 6
        A = randn(m, n)

        s_full = svdvals(A)
        s_approx = rsvdvals(A, k; num_oversamples=10, num_power_iterations=3)

        @test length(s_approx) == k
        @test isapprox(s_approx, s_full[1:k]; rtol=1e-2, atol=1e-2)
    end

    @testset "sparse matrix" begin
        Random.seed!(0xdeadbeef)
        m, n, k = 70, 40, 7
        A_sparse = sprandn(m, n, 0.05)

        s_full = svdvals(Matrix(A_sparse))
        s_approx = rsvdvals(A_sparse, k; num_oversamples=10, num_power_iterations=3, sample_vec=similar(s_full, eltype(s_full), 0))

        @test length(s_approx) == k
        @test isapprox(s_approx, s_full[1:k]; rtol=2e-1, atol=2e-1)
    end

    @testset "LinearMaps.jl operator" begin
        Random.seed!(0xdeadbeef)
        m, n, k = 30, 25, 5
        A = randn(m, n)
        L = LinearMap(A)

        s_full = svdvals(A)
        s_approx = rsvdvals(L, k; num_oversamples=10, num_power_iterations=2, sample_vec=similar(A, eltype(A), 0))

        @test length(s_approx) == k
        @test isapprox(s_approx, s_full[1:k]; rtol=1e-2, atol=1e-2)
    end

    @testset "CUDA" begin
        if CUDA.functional()
            Random.seed!(0xdeadbeef)
            m, n, k = 64, 32, 10
            A = randn(Float32, m, n)
            dA = cu(A)

            sample_vec = CUDA.zeros(Float32, m)
            s_approx = rsvdvals(dA, k; num_oversamples=10, num_power_iterations=2, sample_vec=sample_vec)

            @test length(s_approx) == k

            # Compare against CPU reference
            s_full = svdvals(A)
            s_approx_cpu = Array(s_approx)
            @test isapprox(s_approx_cpu, s_full[1:k]; rtol=3e-2, atol=3e-2)
        else
            @info "Skipping rsvdvals CUDA tests: CUDA not functional on this system"
        end
    end

    @testset "seed_Q warm start" begin
        Random.seed!(0xdeadbeef)
        m, n, k = 30, 50, 5 # wide so rsvdvals does not transpose internally
        A = randn(m, n)
        s_full = svdvals(A)

        # Warm start from a first solve's left singular vectors
        svd1 = rsvd(A, k; num_oversamples=10, num_power_iterations=3)
        s2 = rsvdvals(A, k; num_oversamples=10, num_power_iterations=3, seed_Q=svd1.U)
        @test length(s2) == k
        @test isapprox(s2, s_full[1:k]; rtol=1e-2, atol=1e-2)

        # Near-exact seed needs no power iterations
        F = svd(A)
        seed = F.U[:, 1:k+5]
        s3 = rsvdvals(A, k; num_oversamples=10, num_power_iterations=0, seed_Q=seed)
        @test isapprox(s3, s_full[1:k]; rtol=1e-2, atol=1e-2)

        # Partial seed is padded with random columns
        seed_partial = F.U[:, 1:3]
        s4 = rsvdvals(A, k; num_oversamples=10, num_power_iterations=2, seed_Q=seed_partial)
        @test length(s4) == k
        @test isapprox(s4, s_full[1:k]; rtol=1e-2, atol=1e-2)

        # Wrong number of rows throws
        @test_throws DimensionMismatch rsvdvals(A, k; seed_Q=randn(m + 1, k))

        # Seeding a tall operator is not supported
        A_tall = randn(n, m)
        @test_throws ArgumentError rsvdvals(A_tall, k; seed_Q=randn(n, k))
    end
end # @testset "rsvdvals tests"
