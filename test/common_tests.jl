using Test
using LinearAlgebra
using SparseArrays
using LinearMaps
using CUDA

# materialize_mat is internal, so reach it through the module.
const materialize_mat = MatrixFreeRandomizedLinearAlgebra.materialize_mat

# A bare matrix-free operator: it knows its size and eltype and how to multiply,
# but it is not an AbstractMatrix, so it exercises the opaque-operator fallback.
struct OpaqueOp{T}
    A::Matrix{T}
end
Base.size(op::OpaqueOp) = size(op.A)
Base.size(op::OpaqueOp, d::Integer) = size(op.A, d)
Base.eltype(::OpaqueOp{T}) where {T} = T
Base.:*(op::OpaqueOp, X::AbstractMatrix) = op.A * X

@testset "materialize_mat tests" begin
    @testset "CPU" begin
        m, n = 6, 4
        A = randn(m, n)
        proto = similar(A, eltype(A), 0) # a 0-length Vector{Float64}

        @testset "dense fast path returns the same object" begin
            @test materialize_mat(A, proto) === A
        end

        @testset "wrappers materialize to an independent dense Matrix" begin
            S = randn(n, n) # square so the wrappers below are well defined

            for (label, W, expected) in (
                ("Adjoint", A', collect(A')),
                ("Transpose", transpose(A), collect(transpose(A))),
                ("UpperTriangular", UpperTriangular(S), Matrix(UpperTriangular(S))),
                ("Hermitian", Hermitian(S), Matrix(Hermitian(S))),
                ("SparseMatrixCSC", sprandn(m, n, 0.4), nothing),
            )
                B = materialize_mat(W, proto)
                @test B isa Matrix{Float64}
                @test size(B) == size(W)
                @test eltype(B) == eltype(W)
                @test B == (expected === nothing ? Matrix(W) : expected)

                # The result must not alias the source: mutating it leaves W alone.
                Wbefore = copy(W)
                B .= 0
                @test W == Wbefore
            end
        end

        @testset "LinearMap is materialized via multiply" begin
            B = materialize_mat(LinearMap(A), proto)
            @test B isa Matrix{Float64}
            @test B == A
        end

        @testset "opaque operator is materialized via multiply" begin
            op = OpaqueOp(A)
            B = materialize_mat(op, proto)
            @test B isa Matrix{Float64}
            @test size(B) == size(A)
            @test B == A
        end
    end

    @testset "CUDA" begin
        if CUDA.functional()
            m, n = 6, 4
            A = randn(Float32, m, n)
            dA = cu(A)
            cuproto = similar(dA, eltype(dA), 0)
            cpuproto = similar(A, eltype(A), 0)

            # Fast path on the GPU returns the same object.
            @test materialize_mat(dA, cuproto) === dA

            # Move a CPU matrix onto the GPU.
            B = materialize_mat(A, cuproto)
            @test B isa CuMatrix
            @test Array(B) == A

            # Move a GPU matrix back to the CPU.
            C = materialize_mat(dA, cpuproto)
            @test C isa Matrix
            @test C == Array(dA)

            # UpperTriangular backed by a CuMatrix goes through the triu! path.
            S = cu(randn(Float32, n, n))
            U = materialize_mat(UpperTriangular(S), cuproto)
            @test U isa CuMatrix
            @test Array(U) == Matrix(UpperTriangular(Array(S)))

            # Adjoint of a CuMatrix.
            D = materialize_mat(dA', cuproto)
            @test D isa CuMatrix
            @test Array(D) == Array(dA)'

            # LinearMap and opaque operator on the GPU.
            L = materialize_mat(LinearMap(dA), cuproto)
            @test L isa CuMatrix
            @test Array(L) == A
        else
            @info "Skipping materialize_mat CUDA tests: CUDA not functional on this system"
        end
    end
end
