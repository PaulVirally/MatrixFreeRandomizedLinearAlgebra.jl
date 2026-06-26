using Test
using LinearAlgebra
using SparseArrays
using LinearMaps
using CUDA
using Random

# materialize_mat is internal, so reach it through the module.
const materialize_mat = MatrixFreeRandomizedLinearAlgebra.materialize_mat

# CPU-resident fake "device" array that reproduces the GPU host-dispatch bug on a
# machine without a GPU. See the file header for what it does and does not mimic.
include("mock_device.jl")

# A bare matrix-free operator: it knows its size and eltype and how to multiply,
# but it is not an AbstractMatrix, so it exercises the opaque-operator fallback.
struct OpaqueOp{T}
    A::AbstractMatrix{T}
end
Base.size(op::OpaqueOp) = size(op.A)
Base.size(op::OpaqueOp, d::Integer) = size(op.A, d)
Base.eltype(::OpaqueOp{T}) where {T} = T
Base.:*(op::OpaqueOp, X::AbstractMatrix) = op.A * X

# ---------------------------------------------------------------------------
# Test backends. Each names a "device", a zero-length prototype on it, a `wrap`
# that moves a host matrix onto it (eltype-preserving), and the dense array type
# `materialize_mat` must return there. The mock runs everywhere; the GPU is gated.
# ---------------------------------------------------------------------------
function backends_for(::Type{T}) where {T}
    bks = Any[
        (name = "cpu",  proto = Vector{T}(undef, 0),     wrap = copy,                       DT = Matrix),
        (name = "mock", proto = mock_proto(T),           wrap = A -> MockDeviceArray(copy(A)), DT = MockDeviceArray),
    ]
    if CUDA.functional()
        push!(bks, (name = "gpu", proto = CuArray{T}(undef, 0), wrap = A -> CuArray(A), DT = CuArray))
    end
    return bks
end

# The full contract in one place: dense, correct device type, eltype, size and
# values, and an independent buffer the caller may overwrite. Only use this for
# operators that always allocate (never the `===` fast paths).
function check_materialize(op, proto, ref; DT, rtol = sqrt(eps(real(eltype(ref)))))
    B = materialize_mat(op, proto)
    @test B isa DT
    @test eltype(B) == eltype(ref)
    @test size(B) == size(ref)
    @test collect(B) ≈ ref rtol = rtol
    # The result must be an independent, writable buffer: clobbering it must not
    # corrupt the operator, so a fresh materialization still matches.
    fill!(B, zero(eltype(B)))
    B2 = materialize_mat(op, proto)
    @test collect(B2) ≈ ref rtol = rtol
    return B
end

# A matrix-free FunctionMap whose action multiplies by a device-resident matrix
# `Md` (mirrors the FFT/Gila operators that triggered the real bug). The adjoint
# action multiplies by `Md'`.
function device_function_map(::Type{T}, Md, m, n) where {T}
    LinearMap{T}((y, x) -> mul!(y, Md, x), (y, x) -> mul!(y, Md', x), m, n)
end

@testset "materialize_mat tests" begin
    # =====================================================================
    # Fast paths: a dense matrix already on the prototype's device is handed
    # straight back (same object), so callers can mutate it cheaply.
    # =====================================================================
    @testset "dense fast path returns the same object" begin
        A = randn(6, 4)
        @test materialize_mat(A, similar(A, eltype(A), 0)) === A
        if CUDA.functional()
            dA = CuArray(randn(Float32, 6, 4))
            @test materialize_mat(dA, similar(dA, eltype(dA), 0)) === dA
        end
    end

    # =====================================================================
    # Already-a-matrix wrappers materialize to an independent dense Matrix on
    # the CPU (Adjoint, Transpose, (Upper)Triangular, Hermitian, Symmetric,
    # Diagonal, sparse). These hit the generic copyto! method.
    # =====================================================================
    @testset "CPU wrappers materialize to an independent dense Matrix" begin
        m, n = 6, 4
        A = randn(m, n)
        S = randn(n, n) # square so the square wrappers are well defined
        proto = similar(A, eltype(A), 0)

        for (label, W) in (
            ("Adjoint", A'),
            ("Transpose", transpose(A)),
            ("UpperTriangular", UpperTriangular(S)),
            ("LowerTriangular", LowerTriangular(S)),
            ("Hermitian", Hermitian(S)),
            ("Symmetric", Symmetric(S)),
            ("Diagonal", Diagonal(randn(n))),
            ("SparseMatrixCSC", sprandn(m, n, 0.4)),
        )
            @testset "$label" begin
                B = materialize_mat(W, proto)
                @test B isa Matrix{Float64}
                @test size(B) == size(W)
                @test eltype(B) == eltype(W)
                @test B == Matrix(W)

                # The result must not alias the source: mutating it leaves W alone.
                Wbefore = copy(W)
                B .= 0
                @test W == Wbefore
            end
        end
    end

    # =====================================================================
    # Operator sweep: every operator shape × backend × eltype, all checked
    # against an independent host reference. The composite shapes are the ones
    # the old block-mul! path got wrong on the GPU.
    # =====================================================================
    @testset "operator sweep" begin
        for T in (Float64, Float32, ComplexF64)
            Random.seed!(0x5eed00d + hash(T) % 1000)
            n = 8
            # Host reference matrices.
            Ahost = randn(T, n, n)
            Bhost = randn(T, n, n)
            Chost = randn(T, n, n)

            for bk in backends_for(T)
                @testset "$T / $(bk.name)" begin
                    wrap, proto, DT = bk.wrap, bk.proto, bk.DT
                    Ad, Bd, Cd = wrap(Ahost), wrap(Bhost), wrap(Chost)

                    # Single dense LinearMap.
                    check_materialize(LinearMap(Ad), proto, Ahost; DT)

                    # Pure matrix-free FunctionMap.
                    check_materialize(device_function_map(T, Ad, n, n), proto, Ahost; DT)

                    # Lazy composition and linear combination.
                    check_materialize(LinearMap(Ad) * LinearMap(Bd), proto, Ahost * Bhost; DT)
                    check_materialize(LinearMap(Ad) + LinearMap(Bd), proto, Ahost + Bhost; DT)

                    # Composite + combination (the failing operator's shape) and a deep chain.
                    check_materialize(LinearMap(Ad) * LinearMap(Bd) + LinearMap(Cd), proto,
                                      Ahost * Bhost + Chost; DT)
                    check_materialize(LinearMap(Ad) * LinearMap(Bd) * LinearMap(Cd), proto,
                                      Ahost * Bhost * Chost; DT)

                    # FunctionMap composite + combination: the shape that actually
                    # densifies a sub-chain to the host on the GPU. This is the
                    # regression for the reported bug.
                    F1 = device_function_map(T, Ad, n, n)
                    F2 = device_function_map(T, Bd, n, n)
                    F3 = device_function_map(T, Cd, n, n)
                    check_materialize(F1 * F2 + F3, proto, Ahost * Bhost + Chost; DT)

                    # Adjoint of an operator must conjugate-transpose (matters for complex).
                    check_materialize(LinearMap(Ad)', proto, collect(Ahost'); DT)
                    check_materialize(device_function_map(T, Ad, n, n)', proto, collect(Ahost'); DT)
                end
            end
        end
    end

    # =====================================================================
    # The actual call pattern from range_finder/rsvd/trace: `operator * Ω` is a
    # LAZY CompositeMap (Ω wrapped, not a dense product). With a device-backed
    # FunctionMap operator this is exactly what crashed at range_finder.jl:70.
    # =====================================================================
    @testset "operator * Ω stays on device (regression)" begin
        for T in (Float64, ComplexF64)
            Random.seed!(0xC0FFEE + hash(T) % 1000)
            n, k = 16, 4
            Ahost, Bhost, Chost = randn(T, n, n), randn(T, n, n), randn(T, n, n)
            refmat = Ahost * Bhost + Chost

            for bk in backends_for(T)
                @testset "$T / $(bk.name)" begin
                    wrap, proto, DT = bk.wrap, bk.proto, bk.DT
                    Ad, Bd, Cd = wrap(Ahost), wrap(Bhost), wrap(Chost)

                    # FunctionMap operator (the crashing shape) and Ω built on-device,
                    # mirroring `Ω = similar(sample_vec, …)` in range_finder_start.
                    op = device_function_map(T, Ad, n, n) * device_function_map(T, Bd, n, n) +
                         device_function_map(T, Cd, n, n)
                    Ωhost = randn(T, n, k)
                    Ω = wrap(Ωhost)
                    check_materialize(op * Ω, proto, refmat * Ωhost; DT)

                    # The adjoint product `operator' * Q` (the rsvd/svd call site).
                    Qhost = randn(T, n, k)
                    Q = wrap(Qhost)
                    check_materialize(op' * Q, proto, refmat' * Qhost; DT)

                    # Single right-hand column (loop-boundary n=1 on the RHS).
                    ωhost = randn(T, n, 1)
                    check_materialize(op * wrap(ωhost), proto, refmat * ωhost; DT)
                end
            end
        end
    end

    # =====================================================================
    # Rectangular operators: materialize_mat must preserve m×n shape, including a
    # single-column operator (the column loop's n=1 boundary).
    # =====================================================================
    @testset "rectangular and single-column operators" begin
        for T in (Float64, ComplexF64)
            Random.seed!(0xABCDEF + hash(T) % 1000)
            for (m, n) in ((10, 6), (6, 10), (5, 1))
                Mhost = randn(T, m, n)
                for bk in backends_for(T)
                    @testset "$T / $(bk.name) / $(m)x$(n)" begin
                        Md = bk.wrap(Mhost)
                        check_materialize(LinearMap(Md), bk.proto, Mhost; DT = bk.DT)
                        check_materialize(device_function_map(T, Md, m, n), bk.proto, Mhost; DT = bk.DT)
                    end
                end
            end
        end
    end

    # =====================================================================
    # Device placement. For a CONCRETE matrix, materialize_mat transfers it onto
    # the prototype's device (via copyto!), so the result follows the prototype.
    # A matrix-free / LinearMap operator has no such freedom: it can only be
    # applied on the device its own data lives on, so the operator and the
    # prototype MUST be co-located — as they always are in this package, where Ω
    # is built with `similar(sample_vec, …)` and the operator shares that device.
    # A mismatched pair is unsupported and must error rather than silently
    # dispatch host BLAS on a device array (the original block-mul! crashed here
    # too). We pin both directions so a CPU-only run catches it.
    # =====================================================================
    @testset "device placement" begin
        n = 6
        Ahost = randn(ComplexF64, n, n)
        Bhost = randn(ComplexF64, n, n)
        ref = Ahost * Bhost

        @testset "concrete matrices transfer to the prototype's device" begin
            onmock = materialize_mat(Ahost, mock_proto(ComplexF64))
            @test onmock isa MockDeviceArray
            @test collect(onmock) == Ahost
            back = materialize_mat(MockDeviceArray(copy(Ahost)), Vector{ComplexF64}(undef, 0))
            @test back isa Matrix
            @test back == Ahost
        end

        @testset "co-located operator and prototype materialize fine" begin
            op_mock = LinearMap(MockDeviceArray(copy(Ahost))) * LinearMap(MockDeviceArray(copy(Bhost)))
            M = materialize_mat(op_mock, mock_proto(ComplexF64))
            @test M isa MockDeviceArray
            @test collect(M) ≈ ref
        end

        @testset "mismatched operator/prototype device is rejected" begin
            dev_op = LinearMap(MockDeviceArray(copy(Ahost))) * LinearMap(MockDeviceArray(copy(Bhost)))
            @test_throws HostDeviceMixError materialize_mat(dev_op, Vector{ComplexF64}(undef, 0))
            host_op = LinearMap(Ahost) * LinearMap(Bhost)
            @test_throws HostDeviceMixError materialize_mat(host_op, mock_proto(ComplexF64))
        end

        if CUDA.functional()
            @testset "CUDA concrete-matrix transfer (both directions)" begin
                dB = materialize_mat(Ahost, CuArray{ComplexF64}(undef, 0))
                @test dB isa CuArray
                @test Array(dB) == Ahost
                hB = materialize_mat(CuArray(Ahost), Vector{ComplexF64}(undef, 0))
                @test hB isa Matrix
                @test hB == Ahost
            end
            @testset "CUDA co-located operator materializes on device" begin
                op_gpu = LinearMap(CuArray(Ahost)) * LinearMap(CuArray(Bhost))
                M = materialize_mat(op_gpu, CuArray{ComplexF64}(undef, 0))
                @test M isa CuArray
                @test Array(M) ≈ ref
            end
            @testset "CUDA mismatched operator/prototype device errors" begin
                op_gpu = LinearMap(CuArray(Ahost)) * LinearMap(CuArray(Bhost))
                @test_throws Exception materialize_mat(op_gpu, Vector{ComplexF64}(undef, 0))
            end
        end
    end

    # =====================================================================
    # Opaque operator fallback: a non-AbstractMatrix that only knows size + `*`.
    # =====================================================================
    @testset "opaque operator is materialized via multiply" begin
        m, n = 6, 4
        A = randn(m, n)
        check_materialize(OpaqueOp(A), Vector{Float64}(undef, 0), A; DT = Matrix)
        check_materialize(OpaqueOp(MockDeviceArray(copy(A))), mock_proto(Float64), A; DT = MockDeviceArray)
        if CUDA.functional()
            check_materialize(OpaqueOp(CuArray(Float32.(A))), CuArray{Float32}(undef, 0),
                              Float32.(A); DT = CuArray)
        end
    end

    # =====================================================================
    # GPU-specific paths that have no CPU/mock analogue.
    # =====================================================================
    @testset "CUDA-only paths" begin
        if CUDA.functional()
            n = 5
            # UpperTriangular backed by a CuMatrix goes through the triu! path
            # (avoids scalar-reading the implicit zeros).
            S = CuArray(randn(Float32, n, n))
            U = materialize_mat(UpperTriangular(S), CuArray{Float32}(undef, 0))
            @test U isa CuMatrix
            @test Array(U) == Matrix(UpperTriangular(Array(S)))

            # Adjoint of a CuMatrix.
            dA = CuArray(randn(Float32, 6, 4))
            D = materialize_mat(dA', CuArray{Float32}(undef, 0))
            @test D isa CuMatrix
            @test Array(D) == Array(dA)'
        else
            @info "Skipping materialize_mat CUDA-only tests: CUDA not functional on this system"
        end
    end
end
