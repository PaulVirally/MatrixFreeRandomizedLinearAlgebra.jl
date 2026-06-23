using Random
using LinearAlgebra
using SparseArrays
using LinearMaps
using CUDA

@testset "reigvals_hermitian tests" begin
    @testset "dense Hermitian matrix" begin
        Random.seed!(0xdeadbeef)
        n, k = 45, 9
        A = randn(n, n)
        A = (A + A') / 2

        λ_full = eigen(Hermitian(A)).values
        λ_ref = sort(λ_full; rev=true)[1:k]

        λ_approx = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=8, sample_vec=similar(λ_full, eltype(λ_full), 0))

        @test length(λ_approx) == k
        @test isapprox(λ_approx, λ_ref; rtol=2e-2, atol=2e-2)
    end

    @testset "sparse Hermitian matrix" begin
        Random.seed!(0xdeadbeef)
        n, k = 60, 10
        A_sparse = sprandn(n, n, 0.03)
        A_sparse = (A_sparse + A_sparse') / 2

        λ_full = eigen(Hermitian(Matrix(A_sparse))).values
        λ_ref = sort(λ_full; rev=true)[1:k]

        λ_approx = reigvals_hermitian(A_sparse, k; num_oversamples=20, num_power_iterations=10, sample_vec=similar(λ_full, eltype(λ_full), 0))

        @test length(λ_approx) == k
        @test isapprox(λ_approx, λ_ref; rtol=5e-2, atol=5e-2)
    end

    @testset "LinearMaps.jl Hermitian operator" begin
        Random.seed!(0xdeadbeef)
        n, k = 40, 6
        A = randn(n, n)
        A = (A + A') / 2

        L = LinearMap(A)

        λ_full = eigen(Hermitian(A)).values
        λ_ref = sort(λ_full; rev=true)[1:k]

        λ_approx = reigvals_hermitian(L, k; num_oversamples=10, num_power_iterations=8, sample_vec=similar(λ_full, eltype(λ_full), 0))

        @test length(λ_approx) == k
        @test isapprox(λ_approx, λ_ref; rtol=2e-2, atol=2e-2)
    end

    @testset "seed_Q warm start" begin
        Random.seed!(0xdeadbeef)
        n, k = 45, 8
        A = randn(n, n)
        A = (A + A') / 2

        E_full = eigen(Hermitian(A))
        idx = sortperm(E_full.values; rev=true)
        λ_ref = E_full.values[idx][1:k]

        # Warm start from a first solve's eigenvectors
        E1 = reigen_hermitian(A, k; num_oversamples=20, num_power_iterations=8)
        λ2 = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=8, seed_Q=E1.vectors)
        @test length(λ2) == k
        @test isapprox(λ2, λ_ref; rtol=2e-2, atol=2e-2)

        # Near-exact seed needs no power iterations
        seed = E_full.vectors[:, idx[1:k+5]]
        λ3 = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=0, seed_Q=seed)
        @test isapprox(λ3, λ_ref; rtol=2e-2, atol=2e-2)

        # Partial seed is padded with random columns
        seed_partial = E_full.vectors[:, idx[1:3]]
        λ4 = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=6, seed_Q=seed_partial)
        @test length(λ4) == k
        @test isapprox(λ4, λ_ref; rtol=2e-2, atol=2e-2)

        # Wrong number of rows throws
        @test_throws DimensionMismatch reigvals_hermitian(A, k; seed_Q=randn(n + 1, k))
    end
end # @testset "reigvals_hermitian tests"
