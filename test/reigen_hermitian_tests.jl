using Random
using LinearAlgebra
using SparseArrays
using LinearMaps
using CUDA

@testset "reigen_hermitian tests" begin
    @testset "dense Hermitian matrix" begin
        Random.seed!(0xdeadbeef)
        n, k = 40, 6
        A = randn(n, n)
        A = (A + A') / 2 # make it symmetric/Hermitian

        E_full = eigen(Hermitian(A))
        E_approx = reigen_hermitian(A, k; num_oversamples=20, num_power_iterations=6, sample_vec=similar(E_full.values, eltype(E_full.values), 0))

        @test length(E_approx.values) == k
        @test size(E_approx.vectors) == (n, k)

        # eigen(Hermitian(A)) returns ascending order; randomized code sorts descending
        λ_ref = sort(E_full.values; rev=true)[1:k]
        @test isapprox(E_approx.values, λ_ref; rtol=1e-2, atol=1e-2)

        # Check approximate orthonormality of eigenvectors
        V = E_approx.vectors
        @test isapprox(V' * V, I(k); rtol=1e-6, atol=1e-6)

        # Check residual norms ||A v_i - λ_i v_i||
        for i in 1:k
            v = V[:, i]
            λ = E_approx.values[i]
            @test norm(A * v - λ * v) / norm(A * v) < 1e-2
        end
    end

    @testset "sparse Hermitian matrix" begin
        Random.seed!(0xdeadbeef)
        n, k = 50, 8
        A_sparse = sprandn(n, n, 0.05)
        A_sparse = (A_sparse + A_sparse') / 2  # Hermitian sparse

        E_full = eigen(Hermitian(Matrix(A_sparse)))
        E_approx = reigen_hermitian(A_sparse, k; num_oversamples=20, num_power_iterations=8, sample_vec=similar(E_full.values, eltype(E_full.values), 0))

        @test length(E_approx.values) == k
        @test size(E_approx.vectors) == (n, k)

        λ_ref = sort(E_full.values; rev=true)[1:k]
        @test isapprox(E_approx.values, λ_ref; rtol=5e-2, atol=5e-2)
    end

    @testset "LinearMaps.jl Hermitian operator" begin
        Random.seed!(0xdeadbeef)
        n, k = 35, 7
        A = randn(n, n)
        A = (A + A') / 2

        # Hermitian LinearMap: define explicit action and adjoint
        L = LinearMap(A) # This preserves A' = A

        E_full = eigen(Hermitian(A))
        E_approx = reigen_hermitian(L, k; num_oversamples=20, num_power_iterations=7, sample_vec=similar(E_full.values, eltype(E_full.values), 0))

        @test length(E_approx.values) == k
        @test size(E_approx.vectors) == (n, k)

        λ_ref = sort(E_full.values; rev=true)[1:k]
        @test isapprox(E_approx.values, λ_ref; rtol=2e-2, atol=2e-2)
    end
end # @testset "reigen_hermitian tests"
