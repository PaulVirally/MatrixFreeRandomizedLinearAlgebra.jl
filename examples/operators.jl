# Shared operators for the examples.
#
# The examples all use a 2D periodic (circular) Gaussian blur acting on n×n
# images. Blurring is a convolution, and convolution is diagonalized by the
# Fourier transform, so we apply the operator with a couple of FFTs and never
# form the n²×n² matrix. That makes it a simple matrix-free operator: cheap to
# apply, but huge to write down as a dense matrix.

using FFTW
using LinearMaps
using LinearAlgebra

"""
    gaussian_psf(n; sigma, shift)

Build an `n×n` Gaussian point-spread function (the blur kernel), normalized to
sum to one and laid out with its peak at index `(1, 1)` so that the circular
convolution is zero-phase. A non-zero `shift` moves the peak off the origin,
which makes the resulting operator non-symmetric (handy for the SVD example).
"""
function gaussian_psf(n::Int; sigma::Real = n / 16, shift::Tuple{Int,Int} = (0, 0))
    c = n ÷ 2 + 1
    g = [exp(-((i - c)^2) / (2 * sigma^2)) for i in 1:n]
    psf = g * g'                 # separable, symmetric, centered
    psf ./= sum(psf)
    psf = ifftshift(psf)         # move the peak to index (1, 1)
    shift == (0, 0) || (psf = circshift(psf, shift))
    return psf
end

"""
    blur_operator(n; sigma, shift) -> LinearMap

A matrix-free 2D periodic Gaussian-blur operator on `n×n` images, returned as a
`LinearMaps.LinearMap` of size `n²×n²`. Forward and adjoint actions are both two
FFTs, so each is `O(n² log n)` regardless of how large the implied matrix is.

With `shift = (0, 0)` the kernel is symmetric and the operator is Hermitian
positive semidefinite (used by the eigenvalue and trace examples). A small
`shift` breaks the symmetry, giving a genuinely non-symmetric operator for the
SVD example.
"""
function blur_operator(n::Int; sigma::Real = n / 16, shift::Tuple{Int,Int} = (0, 0))
    khat = fft(gaussian_psf(n; sigma = sigma, shift = shift))  # transfer function
    N = n * n
    fwd(x) = vec(real(ifft(khat .* fft(reshape(x, n, n)))))
    adj(y) = vec(real(ifft(conj.(khat) .* fft(reshape(y, n, n)))))
    return LinearMap{Float64}(fwd, adj, N, N)
end

"""
    densify(op) -> Matrix

Materialize a matrix-free operator into a dense matrix by applying it to the
columns of the identity. Only used to compute reference answers for small
problems. Doing this at scale is exactly what the package lets you avoid.
"""
function densify(op)
    M, N = size(op)
    A = Matrix{eltype(op)}(undef, M, N)
    e = zeros(eltype(op), N)
    for j in 1:N
        e[j] = one(eltype(op))
        @views A[:, j] .= op * e
        e[j] = zero(eltype(op))
    end
    return A
end
