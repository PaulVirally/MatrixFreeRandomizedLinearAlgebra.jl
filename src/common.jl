using CUDA
using SparseArrays
using LinearAlgebra
using LinearMaps

# Materialize function to ensure we have a concrete matrix type
materialize_mat(A::CUDA.CuMatrix, target_prototype::AbstractArray) = A
materialize_mat(A::Adjoint{T,S}, target_prototype::AbstractArray) where {T,S<:CUDA.CuMatrix} = CUDA.CuMatrix(A)
materialize_mat(A::Transpose{T,S}, target_prototype::AbstractArray) where {T,S<:CUDA.CuMatrix} = CUDA.CuMatrix(A)
materialize_mat(A::StridedMatrix, target_prototype::AbstractArray) = A
materialize_mat(A::Adjoint{T,S}, target_prototype::AbstractArray) where {T,S<:StridedMatrix} = copy(A)
materialize_mat(A::Transpose{T,S}, target_prototype::AbstractArray) where {T,S<:StridedMatrix} = copy(A)
function materialize_mat(A::AbstractSparseMatrix, target_prototype::AbstractArray)
    B = similar(target_prototype, eltype(A), size(A))
    B .= collect(A)
    return B
end
function materialize_mat(A::LinearMap, target_prototype::AbstractArray)
    B = similar(target_prototype, eltype(A), size(A))
    _, n = size(A)
    v = similar(target_prototype, eltype(A), n)
    @inbounds for i in 1:n
        fill!(v, zero(eltype(A)))
        CUDA.@allowscalar v[i] = one(eltype(A))
        B[:, i] .= A * v
    end
    return B
end
function materialize_mat(A, target_prototype::AbstractArray)
    B = similar(target_prototype, eltype(A), size(A))
    try
        copyto!(B, A)
        return B
    catch
    end
    try
        _, n = size(A)
        @inbounds for i in 1:n
            B[:, i] .= A[:, i]
        end
        return B
    catch
    end
    try
        B = similar(target_prototype, eltype(A), size(A))
        _, n = size(A)
        v = similar(target_prototype, eltype(A), n)
        @inbounds for i in 1:n
            fill!(v, zero(eltype(A)))
            CUDA.@allowscalar v[j] = one(eltype(A))
            B[:, i] .= A * v
        end
        return B
    catch
    end
    try
        m, n = size(A)
        @inbounds for j in 1:n
            for i in 1:m
                B[i, j] = A[i, j]
            end
        end
        return B
    catch
    end
    B .= Array(A)
    return B
end
