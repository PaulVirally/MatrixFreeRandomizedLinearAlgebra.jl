using Random
using LinearAlgebra
using Funicular

# `panelplan` comes from panel_setup.jl.

# The error the optimal rank-k truncation leaves behind, which no rank-k
# approximation can beat.
optimal_error(A, k) = (F = svd(A); norm(A - F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.V[:, 1:k]'))

@testset "panel rsvd tests" begin
    # Square, wide and tall. The last of these exercises the internal transpose
    # and the factor swap that undoes it.
    @testset "rsvd matches the dense path $(m)×$(n) $T" for (m, n) in ((40, 40), (40, 64), (64, 40)), T in (Float64, ComplexF64)
        Random.seed!(0xdeadbeef)
        k = 5
        A = randn(T, m, n)
        σ = svdvals(A)

        plan = panelplan(; panel_width=3)
        F = rsvd(A, k; num_oversamples=10, num_power_iterations=3, plan=plan, seed=0xB0A7)

        @test F isa PanelSVD
        @test length(F.S) == k
        @test F.S isa Vector{real(T)}
        @test size(F.U) == (m, k)
        @test size(F.V) == (n, k)
        @test isapprox(F.S, σ[1:k]; rtol=1e-2, atol=1e-2)

        U, V = Matrix(F.U), Matrix(F.V)
        @test opnorm(U' * U - I) < 1e-8
        @test opnorm(V' * V - I) < 1e-8
        # A randomized rank-k basis cannot beat the optimal truncation, but with
        # this much oversampling and power iteration it comes very close to it.
        @test norm(A - U * Diagonal(F.S) * V') <= 1.02 * optimal_error(A, k)

        # The values-only entry point runs the same reduction on the same sketch.
        vals = rsvdvals(A, k; num_oversamples=10, num_power_iterations=3, plan=plan, seed=0xB0A7)
        @test vals isa Vector{real(T)}
        @test isapprox(vals, F.S; rtol=1e-10, atol=1e-12)
    end

    @testset "factored defers both products $(m)×$(n)" for (m, n) in ((40, 64), (64, 40))
        Random.seed!(0xdeadbeef)
        k = 5
        A = randn(m, n)

        plan = panelplan(; panel_width=3)
        F = rsvd(A, k; num_oversamples=10, num_power_iterations=3, plan=plan, seed=0xB0A7)
        G = rsvd(A, k; num_oversamples=10, num_power_iterations=3, plan=plan, seed=0xB0A7, factored=true)

        @test G isa PanelSVD
        @test G.U isa PanelFactored
        @test G.V isa PanelFactored
        # The swap that handles a tall operator carries the deferred products
        # across unchanged, so the shapes come out the same either way.
        @test size(G.U) == (m, k)
        @test size(G.V) == (n, k)
        @test G.S ≈ F.S
        @test Matrix(G.U) ≈ Matrix(F.U)
        @test Matrix(G.V) ≈ Matrix(F.V)
        @test Matrix(materialize(G.U)) ≈ Matrix(F.U)
        @test Matrix(materialize(G.V)) ≈ Matrix(F.V)
    end

    @testset "an integer seed is reproducible across panel widths" begin
        Random.seed!(0xdeadbeef)
        m, n, k = 40, 64, 5
        A = randn(m, n)

        σ3 = rsvdvals(A, k; num_oversamples=10, num_power_iterations=3, plan=panelplan(; panel_width=3), seed=1234)
        σ7 = rsvdvals(A, k; num_oversamples=10, num_power_iterations=3, plan=panelplan(; panel_width=7), seed=1234)
        # Funicular generates a ghost column from its index and the seed alone,
        # so only the summation order of the row-block sweeps differs.
        @test isapprox(σ3, σ7; rtol=1e-8, atol=1e-10)

        σauto = rsvdvals(A, k; num_oversamples=10, num_power_iterations=3, plan=panelplan(), seed=1234)
        @test isapprox(σ3, σauto; rtol=1e-8, atol=1e-10)
    end

    @testset "seed = nothing draws a fresh sketch each call" begin
        Random.seed!(0xdeadbeef)
        m, n, k = 40, 64, 5
        A = randn(m, n)
        σ = svdvals(A)

        plan = panelplan(; panel_width=3)
        σ1 = rsvdvals(A, k; num_oversamples=10, num_power_iterations=3, plan=plan)
        σ2 = rsvdvals(A, k; num_oversamples=10, num_power_iterations=3, plan=plan)
        @test isapprox(σ1, σ[1:k]; rtol=1e-2, atol=1e-2)
        @test isapprox(σ2, σ[1:k]; rtol=1e-2, atol=1e-2)
    end

    @testset "seed_Q warm start" begin
        Random.seed!(0xdeadbeef)
        m, n, k = 40, 64, 5 # wide, so rsvd does not transpose internally
        A = randn(m, n)
        σ = svdvals(A)

        plan = panelplan(; panel_width=3)
        F = rsvd(A, k; num_oversamples=10, num_power_iterations=3, plan=plan, seed=0xB0A7)

        # A basis from a previous panel solve, handed straight back in.
        σ_panel = rsvdvals(A, k; num_oversamples=10, num_power_iterations=2, plan=plan, seed=0xB0A7, seed_Q=F.U)
        @test isapprox(σ_panel, σ[1:k]; rtol=1e-2, atol=1e-2)

        # The same basis collected to the host.
        σ_host = rsvdvals(A, k; num_oversamples=10, num_power_iterations=2, plan=plan, seed=0xB0A7, seed_Q=Matrix(F.U))
        @test isapprox(σ_host, σ[1:k]; rtol=1e-2, atol=1e-2)

        # A deferred product, which the seed path has to carry out before it can
        # copy any columns.
        G = rsvd(A, k; num_oversamples=10, num_power_iterations=3, plan=plan, seed=0xB0A7, factored=true)
        σ_factored = rsvdvals(A, k; num_oversamples=10, num_power_iterations=2, plan=plan, seed=0xB0A7, seed_Q=G.U)
        @test isapprox(σ_factored, σ[1:k]; rtol=1e-2, atol=1e-2)

        # A near-exact seed needs no power iterations at all.
        exact = svd(A).U[:, 1:k+5]
        σ_exact = rsvdvals(A, k; num_oversamples=10, num_power_iterations=0, plan=plan, seed=0xB0A7, seed_Q=exact)
        @test isapprox(σ_exact, σ[1:k]; rtol=1e-2, atol=1e-2)

        # Fewer columns than the sketch dimension: the shortfall is padded with
        # operator * Ω columns, so oversampling survives.
        σ_partial = rsvdvals(A, k; num_oversamples=10, num_power_iterations=2, plan=plan, seed=0xB0A7, seed_Q=svd(A).U[:, 1:3])
        @test isapprox(σ_partial, σ[1:k]; rtol=1e-2, atol=1e-2)

        @test_throws DimensionMismatch rsvd(A, k; plan=plan, seed_Q=randn(m + 1, k))
        @test_throws DimensionMismatch rsvdvals(A, k; plan=plan, seed_Q=randn(m + 1, k))

        # A tall operator is solved through its adjoint, whose range is a
        # different space, so a range seed cannot be threaded through.
        A_tall = randn(n, m)
        @test_throws ArgumentError rsvd(A_tall, k; plan=plan, seed_Q=randn(n, k))
        @test_throws ArgumentError rsvdvals(A_tall, k; plan=plan, seed_Q=randn(n, k))
    end

    @testset "a rank-deficient sketch falls back rather than failing" begin
        Random.seed!(0xdeadbeef)
        m, n, r, k = 80, 100, 6, 10
        A = randn(m, r) * randn(r, n) # exact rank r, so a k + p column sketch is dependent

        plan = panelplan(; panel_width=5)
        F = rsvd(A, k; num_oversamples=10, num_power_iterations=2, plan=plan, seed=0xB0A7)
        σ = svdvals(A)

        # The shifted CholeskyQR3 fallback recovers the values that are actually
        # there, and reports the rest at the noise floor, as the dense path does.
        @test isapprox(F.S[1:r], σ[1:r]; rtol=1e-8)
        @test all(<(1e-10 * σ[1]), F.S[r+1:k])
        @test norm(A - Matrix(F.U) * Diagonal(F.S) * Matrix(F.V)') < 1e-10 * norm(A)
    end

    @testset "error paths" begin
        Random.seed!(0xdeadbeef)
        A = randn(30, 40)
        plan = panelplan(; panel_width=3)

        # Panel-only keywords without a plan say so instead of being ignored.
        @test_throws ArgumentError rsvd(A, 3; factored=true)
        @test_throws ArgumentError rsvd(A, 3; seed=1)
        @test_throws ArgumentError rsvdvals(A, 3; seed=1)

        # Validation is skippable, and skipping it still gives a result.
        @test rsvdvals(A, 3; num_oversamples=5, num_power_iterations=2, plan=plan, validate=false) isa Vector
    end
end # @testset "panel rsvd tests"
