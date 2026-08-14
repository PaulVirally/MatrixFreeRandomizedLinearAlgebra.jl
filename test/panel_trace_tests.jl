using Random
using LinearAlgebra
using Funicular

# `panelplan` comes from panel_setup.jl, and `decaying_operator` from
# trace_tests.jl: the panel path is the same estimator on the same family of
# operators, so it is measured against the same yardstick.

# The sphere-normalized column generator and the offset ghost that reproduces a
# wider one's tail columns are internal to the extension. The reuse the adaptive
# path is built on is only correct if the second reproduces the first exactly,
# so the test that pins that down reaches in.
const PanelTraceExt = Base.get_extension(MatrixFreeRandomizedLinearAlgebra, :MFRLAFunicularExt)

@testset "panel trace tests" begin
    @testset "fixed-budget XTrace matches the dense path $T" for T in (Float64, ComplexF64)
        Random.seed!(0xdeadbeef)
        n, m = 60, 30
        A = decaying_operator(T, n)
        tA = tr(A)

        plan = panelplan(; panel_width=3)
        t = trace(A, m; plan=plan, seed=0xB0A7)
        t_dense = trace(A, m)

        @test t isa T
        @test isapprox(t, tA; rtol=1e-2)
        # The two paths draw different (equally valid) sphere-normalized test
        # matrices, so they cannot agree entry by entry; what has to hold is
        # that the panel estimate is no worse than the dense one from the same
        # estimator.
        @test abs(t - tA) <= 10 * abs(t_dense - tA) + 1e-10 * abs(tA)

        nt = trace(A, m; plan=plan, seed=0xB0A7, return_error=true)
        @test nt isa NamedTuple
        @test nt.value == t
        @test nt.error >= 0
    end

    @testset "fixed-budget XTrace on a Hermitian operator" begin
        Random.seed!(0xdeadbeef)
        n, m = 60, 30
        H = decaying_operator(ComplexF64, n; herm=true)
        tH = tr(H)

        t = trace(H, m; plan=panelplan(; panel_width=3), seed=0xB0A7)
        @test isapprox(t, tH; rtol=1e-2)
    end

    @testset "adaptive XTrace via relative_tolerance $T" for T in (Float64, ComplexF64)
        Random.seed!(0xdeadbeef)
        n = 80
        A = decaying_operator(T, n)
        tA = tr(A)

        reltol = 1e-2
        plan = panelplan(; panel_width=3)
        res = trace(A; relative_tolerance=reltol, plan=plan, seed=0xB0A7, return_error=true)

        @test res isa NamedTuple
        @test res.value isa T
        # The request was met rather than abandoned at the cap.
        @test res.error <= reltol * abs(res.value)
        @test isapprox(res.value, tA; rtol=1e-1)

        # Without `return_error` the same run gives back the scalar alone.
        scalar = trace(A; relative_tolerance=reltol, plan=plan, seed=0xB0A7)
        @test scalar == res.value
    end

    @testset "the offset ghost reproduces the wide ghost's tail columns" begin
        n, m, add = 24, 6, 4
        plan = panelplan(; panel_width=3)
        seed = 0xB0A7

        for T in (Float64, ComplexF64)
            wide = GhostPanels(PanelTraceExt.sphere_column!, T, n, m + add; plan=plan, seed=seed, w=3)
            offset = GhostPanels(PanelTraceExt.offset_sphere_column(m, seed), T, n, add; plan=plan, seed=seed, w=3)
            # Bit for bit, not merely in distribution: the adaptive path keeps the
            # images of the leading columns and only multiplies these through the
            # operator, so a mismatch would silently mix two different sketches.
            @test Matrix(offset) == Matrix(wide)[:, (m+1):(m+add)]

            # The generator is XTrace's sphere normalization exactly, since a
            # regenerated column arrives whole.
            cols = Matrix(wide)
            @test all(j -> isapprox(norm(cols[:, j]), sqrt(n)), 1:(m+add))

            # A narrower cut of the same seed is the same matrix.
            narrow = GhostPanels(PanelTraceExt.sphere_column!, T, n, m + add; plan=plan, seed=seed, w=2)
            @test Matrix(narrow) == cols
        end
    end

    @testset "an integer seed is reproducible across panel widths" begin
        Random.seed!(0xdeadbeef)
        n, m = 60, 20
        A = decaying_operator(Float64, n)

        t3 = trace(A, m; plan=panelplan(; panel_width=3), seed=1234)
        t7 = trace(A, m; plan=panelplan(; panel_width=7), seed=1234)
        # Funicular generates a ghost column from its index and the seed alone,
        # so only the summation order of the row-block sweeps differs.
        @test isapprox(t3, t7; rtol=1e-8, atol=1e-10)

        tauto = trace(A, m; plan=panelplan(), seed=1234)
        @test isapprox(t3, tauto; rtol=1e-8, atol=1e-10)

        # The adaptive path widens its sketch as it goes, and the widening has to
        # be width-independent too.
        a3 = trace(A; relative_tolerance=1e-2, plan=panelplan(; panel_width=3), seed=1234)
        a7 = trace(A; relative_tolerance=1e-2, plan=panelplan(; panel_width=7), seed=1234)
        @test isapprox(a3, a7; rtol=1e-8, atol=1e-10)
    end

    @testset "seed = nothing draws a fresh sample each call" begin
        Random.seed!(0xdeadbeef)
        n, m = 60, 30
        A = decaying_operator(Float64, n)
        tA = tr(A)

        plan = panelplan(; panel_width=3)
        t1 = trace(A, m; plan=plan)
        t2 = trace(A, m; plan=plan)
        @test isapprox(t1, tA; rtol=1e-2)
        @test isapprox(t2, tA; rtol=1e-2)
    end

    @testset "max_samples caps the adaptive sketch" begin
        Random.seed!(0xdeadbeef)
        n = 60
        A = decaying_operator(Float64, n)
        tA = tr(A)

        # A tolerance no sketch this small can meet, so the loop runs until the
        # cap stops it. The cap is not a multiple of the starting size either, so
        # the last round grows by less than a doubling.
        res = trace(A; relative_tolerance=1e-14, max_samples=12, plan=panelplan(; panel_width=3), seed=0xB0A7, return_error=true)
        @test isfinite(res.value)
        @test isfinite(res.error)
        @test res.error > 1e-14 * abs(res.value) # it bailed out rather than converging
        @test isapprox(res.value, tA; rtol=2e-1)
    end

    @testset "error paths" begin
        Random.seed!(0xdeadbeef)
        A = decaying_operator(Float64, 30)
        plan = panelplan(; panel_width=3)

        # Hutchinson holds one or two vectors at a time, so there is no sketch for
        # the plan to stream and the combination is refused rather than ignored.
        @test_throws ArgumentError trace(A, 100; low_mem=true, plan=plan)
        @test_throws ArgumentError trace(A; relative_tolerance=1e-2, low_mem=true, plan=plan)

        # The budget rules and the square check apply on the panel path too.
        @test_throws DimensionMismatch trace(randn(3, 4), 5; plan=plan)
        @test_throws ArgumentError trace(A; plan=plan)
        @test_throws ArgumentError trace(A, 5; relative_tolerance=1e-2, plan=plan)

        # `seed` without a plan says so instead of being ignored.
        @test_throws ArgumentError trace(A, 5; seed=1)

        # Validation is skippable, and skipping it still gives a result.
        @test trace(A, 10; plan=plan, seed=0xB0A7, validate=false) isa Float64
    end
end # @testset "panel trace tests"
