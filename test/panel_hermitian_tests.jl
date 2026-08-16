using Random
using LinearAlgebra
using Funicular

hermitian_matrix(::Type{T}, n) where {T} = (A = randn(T, n, n); (A + A') / 2)

# The panel path lives in the extension, and the project-vs-gram heuristic is
# internal to it, so the test that exercises both of its branches reaches in.
const PanelExt = Base.get_extension(MatrixFreeRandomizedLinearAlgebra, :MFRLAFunicularExt)

@testset "panel Hermitian tests" begin
    @testset "reigvals_hermitian matches the dense path $T" for T in (Float64, ComplexF64)
        Random.seed!(0xdeadbeef)
        n, k = 60, 5
        A = hermitian_matrix(T, n)
        λ_ref = sort(eigvals(Hermitian(A)); rev=true)[1:k]

        plan = panelplan(; panel_width=3)
        λ_panel = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=8, plan=plan, seed=0xB0A7)
        λ_dense = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=8)

        @test λ_panel isa Vector{real(T)}
        @test length(λ_panel) == k
        @test isapprox(λ_panel, λ_ref; rtol=1e-2, atol=1e-2)
        @test isapprox(λ_panel, λ_dense; rtol=1e-2, atol=1e-2)
    end

    @testset "reigen_hermitian matches the dense path $T" for T in (Float64, ComplexF64)
        Random.seed!(0xdeadbeef)
        n, k = 60, 5
        A = hermitian_matrix(T, n)
        λ_ref = sort(eigvals(Hermitian(A)); rev=true)[1:k]

        plan = panelplan(; panel_width=3)
        E = reigen_hermitian(A, k; num_oversamples=20, num_power_iterations=8, plan=plan, seed=0xB0A7)

        @test E isa PanelEigen
        @test length(E.values) == k
        @test size(E.vectors) == (n, k)
        @test isapprox(E.values, λ_ref; rtol=1e-2, atol=1e-2)

        V = Matrix(E.vectors)
        @test opnorm(V' * V - I) < 1e-8
        @test norm(A * V - V * Diagonal(E.values)) / norm(A * V) < 1e-2

        # The deferred product is the same basis rotation, just not carried out.
        F = reigen_hermitian(A, k; num_oversamples=20, num_power_iterations=8, plan=plan, seed=0xB0A7, factored=true)
        @test F.vectors isa PanelFactored
        @test size(F.vectors) == (n, k)
        @test F.values ≈ E.values
        @test Matrix(F.vectors) ≈ V
        @test Matrix(materialize(F.vectors)) ≈ V
    end

    @testset "an integer seed is reproducible across panel widths" begin
        Random.seed!(0xdeadbeef)
        n, k = 60, 5
        A = hermitian_matrix(Float64, n)

        λ3 = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=8, plan=panelplan(; panel_width=3), seed=1234)
        λ7 = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=8, plan=panelplan(; panel_width=7), seed=1234)
        # Funicular generates a ghost column from its index and the seed alone,
        # so only the summation order of the row-block sweeps differs.
        @test isapprox(λ3, λ7; rtol=1e-8, atol=1e-10)

        # A different budget cuts the panels differently again, with no explicit
        # width at all.
        λauto = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=8, plan=panelplan(), seed=1234)
        @test isapprox(λ3, λauto; rtol=1e-8, atol=1e-10)
    end

    @testset "seed = nothing draws a fresh sketch each call" begin
        Random.seed!(0xdeadbeef)
        n, k = 60, 5
        A = hermitian_matrix(Float64, n)
        λ_ref = sort(eigvals(Hermitian(A)); rev=true)[1:k]

        λ1 = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=8, plan=panelplan(; panel_width=3))
        λ2 = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=8, plan=panelplan(; panel_width=3))
        @test isapprox(λ1, λ_ref; rtol=1e-2, atol=1e-2)
        @test isapprox(λ2, λ_ref; rtol=1e-2, atol=1e-2)
    end

    @testset "seed_Q warm start" begin
        Random.seed!(0xdeadbeef)
        n, k = 60, 5
        A = hermitian_matrix(Float64, n)
        λ_ref = sort(eigvals(Hermitian(A)); rev=true)[1:k]

        plan = panelplan(; panel_width=3)
        E1 = reigen_hermitian(A, k; num_oversamples=20, num_power_iterations=8, plan=plan, seed=0xB0A7)

        # A basis from a previous panel solve, handed straight back in.
        E2 = reigen_hermitian(A, k; num_oversamples=20, num_power_iterations=0, plan=plan, seed=0xB0A7, seed_Q=E1.vectors)
        @test isapprox(E2.values, λ_ref; rtol=1e-2, atol=1e-2)

        # The same basis collected to the host.
        λ3 = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=0, plan=plan, seed=0xB0A7, seed_Q=Matrix(E1.vectors))
        @test isapprox(λ3, λ_ref; rtol=1e-2, atol=1e-2)

        # Fewer columns than the sketch dimension: the shortfall is padded with
        # operator * Ω columns, so oversampling survives.
        E_full = eigen(Hermitian(A))
        idx = sortperm(E_full.values; rev=true)
        λ4 = reigvals_hermitian(A, k; num_oversamples=20, num_power_iterations=4, plan=plan, seed=0xB0A7, seed_Q=E_full.vectors[:, idx[1:3]])
        @test isapprox(λ4, λ_ref; rtol=1e-2, atol=1e-2)

        # More columns than the sketch dimension: all of them are kept.
        λ5 = reigvals_hermitian(A, k; num_oversamples=2, num_power_iterations=0, plan=plan, seed=0xB0A7, seed_Q=E_full.vectors[:, idx[1:k+10]])
        @test isapprox(λ5, λ_ref; rtol=1e-2, atol=1e-2)

        @test_throws DimensionMismatch reigvals_hermitian(A, k; plan=plan, seed_Q=randn(n + 1, k))
    end

    @testset "the restricted block agrees on both routes" begin
        Random.seed!(0xdeadbeef)
        n, s = 60, 25
        A = hermitian_matrix(Float64, n)
        Qh = Matrix(qr(randn(n, s)).Q)
        reference = Qh' * A * Qh

        # A generous host budget takes the panelmul! + gram route, which needs a
        # second n × s matrix; a budget that cannot hold one falls back to
        # project, which never stores A * Q.
        for host_budget in (64 * 2^20, 20_000)
            plan = panelplan(; host_budget=host_budget, panel_width=3)
            Q = PanelMatrix(Qh; plan=plan)
            @test PanelExt.panel_restricted(A, Q) ≈ reference
            Funicular.free!(Q)
        end
    end

    @testset "error paths" begin
        Random.seed!(0xdeadbeef)
        A = hermitian_matrix(Float64, 30)
        plan = panelplan(; panel_width=3)

        # The Hermitian path needs a square operator.
        @test_throws DimensionMismatch reigen_hermitian(randn(10, 12), 3; plan=plan)
        @test_throws DimensionMismatch reigvals_hermitian(randn(10, 12), 3; plan=plan)

        # Panel-only keywords without a plan say so instead of being ignored.
        @test_throws ArgumentError reigen_hermitian(A, 3; seed=1)
        @test_throws ArgumentError reigen_hermitian(A, 3; factored=true)
        @test_throws ArgumentError reigvals_hermitian(A, 3; seed=1)
        @test_throws ArgumentError rsvd(A, 3; seed=1)
        @test_throws ArgumentError trace(A, 8; seed=1)

        # An operator that fails Funicular's contract is caught on entry, and
        # only when validation is asked for.
        @test reigvals_hermitian(A, 3; num_oversamples=5, num_power_iterations=2, plan=plan, validate=false) isa Vector
    end
end # @testset "panel Hermitian tests"
