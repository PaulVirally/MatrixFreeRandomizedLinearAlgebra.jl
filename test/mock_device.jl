# CI-ONLY test scaffolding — NOT a real device and NOT part of the package.
#
# `MockDeviceArray` is a CPU-resident array that pretends to live on a separate
# "device". It exists so the GPU-only `materialize_mat` host-dispatch bug (and its
# whole class) is reproducible on machines without a GPU. The single behavior it
# mimics faithfully is the one the bug violates:
#
#     a host array may NOT be BLAS-multiplied against a device array.
#
# A `mul!` whose matrix operand is a `MockDeviceArray` throws unless *every*
# operand is also device-resident; same-device multiplies delegate to host BLAS on
# the wrapped `.data`. This is the exact failure CUDA raises as
# `unsafe_convert(::Ptr, ::CuPtr)` when LinearMaps' block-composite path densifies
# an intermediate to a host `Matrix` and probes a device-backed sub-map with host
# vectors. The mock intentionally does NOT reproduce CUDA's scalar-indexing ban —
# that class is covered by the real GPU tests (CUDA forbids scalar indexing by
# default) — so it stays robust and free of broadcast/view false positives.
#
# Device transfers (`copyto!` across array types) are legitimate and are left to
# the generic fallbacks; only BLAS mixing is treated as the bug.

using LinearAlgebra

# Raised when a host array is BLAS-multiplied against a device array.
struct HostDeviceMixError <: Exception
    msg::String
end
Base.showerror(io::IO, e::HostDeviceMixError) = print(io, "HostDeviceMixError: ", e.msg)

struct MockDeviceArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

# A zero-length prototype vector "on the mock device", mirroring `similar(cuvec, T, 0)`.
mock_proto(::Type{T}) where {T} = MockDeviceArray(Vector{T}(undef, 0))

Base.size(a::MockDeviceArray) = size(a.data)
Base.IndexStyle(::Type{<:MockDeviceArray}) = IndexLinear()
Base.getindex(a::MockDeviceArray, i::Int) = a.data[i]
Base.setindex!(a::MockDeviceArray, v, i::Int) = (a.data[i] = v; a)
# `similar` must keep results "on the device" so LinearMaps' `similar(x)`
# intermediates do not silently fall back to host arrays.
Base.similar(::MockDeviceArray, ::Type{T}, dims::Dims) where {T} =
    MockDeviceArray(Array{T}(undef, dims))

# Keep broadcast outputs on the device too (e.g. the `e[j:j] .= one(T)` in the fix
# and the `y .+= …` inside a LinearCombination).
Base.BroadcastStyle(::Type{<:MockDeviceArray}) = Broadcast.ArrayStyle{MockDeviceArray}()
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MockDeviceArray}}, ::Type{T}) where {T} =
    MockDeviceArray(similar(Array{T}, axes(bc)))

# `ismock`/`asplain` see through the wrappers LinearMaps and views introduce, so
# the mix check and the delegation work on column views and adjoint/transpose maps.
ismock(::MockDeviceArray) = true
ismock(x::SubArray) = ismock(parent(x))
ismock(x::Base.ReshapedArray) = ismock(parent(x))
ismock(x::Adjoint) = ismock(parent(x))
ismock(x::Transpose) = ismock(parent(x))
ismock(::Any) = false

asplain(x::MockDeviceArray) = x.data
asplain(x::SubArray) = view(asplain(parent(x)), x.indices...)
asplain(x::Base.ReshapedArray) = reshape(asplain(parent(x)), size(x))
asplain(x::Adjoint) = adjoint(asplain(parent(x)))
asplain(x::Transpose) = transpose(asplain(parent(x)))
asplain(x) = x

const MockMatrix{T} = Union{MockDeviceArray{T,2},
                            Adjoint{T,<:MockDeviceArray{T,2}},
                            Transpose{T,<:MockDeviceArray{T,2}}}

function _checked_mul!(C, A, B, α, β)
    if !(ismock(C) && ismock(B))
        throw(HostDeviceMixError(
            "BLAS would mix host and device operands: " *
            "C::$(typeof(C)), A::$(typeof(A)), B::$(typeof(B)). " *
            "On a real GPU this is the unsafe_convert(::Ptr, ::CuPtr) crash."))
    end
    mul!(asplain(C), asplain(A), asplain(B), α, β)
    return C
end

LinearAlgebra.mul!(C::AbstractVector, A::MockMatrix, B::AbstractVector, α::Number, β::Number) =
    _checked_mul!(C, A, B, α, β)
LinearAlgebra.mul!(C::AbstractMatrix, A::MockMatrix, B::AbstractMatrix, α::Number, β::Number) =
    _checked_mul!(C, A, B, α, β)
LinearAlgebra.mul!(C::AbstractVector, A::MockMatrix, B::AbstractVector) =
    _checked_mul!(C, A, B, true, false)
LinearAlgebra.mul!(C::AbstractMatrix, A::MockMatrix, B::AbstractMatrix) =
    _checked_mul!(C, A, B, true, false)
