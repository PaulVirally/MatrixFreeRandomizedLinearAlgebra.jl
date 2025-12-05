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
end # @testset "reigvals_hermitian tests"
