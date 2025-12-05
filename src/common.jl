using CUDA
using SparseArrays
using LinearAlgebra

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
function materialize_mat(A, target_prototype::AbstractArray)
    B = similar(target_prototype, eltype(A), size(A))
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
