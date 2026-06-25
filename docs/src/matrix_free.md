# What "matrix-free" means

Usually, when you do numerical linear algebra, you have matrix you can index
into. That is, normally, you have an array of numbers sitting in memory. A
matrix-free operator is the opposite. You never store the entries, you only know
how to apply the operator to a vector. As long as you can compute `y = A * x`
(and, for some algorithms, `A' * x`), the routines in this package never need
`A` written out as a grid of numbers.

This matters when the matrix would be huge but its action is cheap. An `n × n`
matrix needs `O(n²)` storage, and a dense multiply costs `O(n²)` work per vector.
Many operators that come up in practice can be applied much faster than that, and
often can't be stored at all at the sizes people care about.

## The FFT

A discrete Fourier transform is a dense `n × n` matrix (the DFT matrix), so
applying it looks like it should cost `O(n²)`. The fast Fourier transform gets
the same answer in `O(n log n)` without ever forming that matrix. That is the
idea in a nutshell: the operator has a perfectly good matrix representation, but
you would never want to build it, so you apply it with an algorithm instead.

The examples in this documentation are built around this for matrix-free
operations. A 2D Gaussian blur is a convolution, and convolution is diagonalized
by the Fourier transform, so the blur operator is "apply a forward FFT, multiply
by the transfer function, apply an inverse FFT". The matrix it stands for is `n²
× n²`, far too big to store for a reasonable image, but each apply is just a
couple of FFTs.

## Matrix-free operators in Julia

Julia makes this comfortable because functions and `*`/`mul!` are first class.
Some common sources of matrix-free operators:

- [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) wraps a
  function (or composes existing maps) into something that behaves like a matrix
  under `*`, `'`, `size`, and `mul!`. This is what the examples here use.
- [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)
  is a similar abstraction common in the optimization world.
- [AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl) and
  [FFTW.jl](https://github.com/JuliaMath/FFTW.jl): FFT plans are matrix-free
  operators you apply with `*`.
- Plain functions, sparse matrices, and GPU arrays work too, as long as they
  support `size` and the multiplies.

## What this package needs from your operator

The requirements are light. An operator has to support:

- `size(operator)` for its dimensions,
- `operator * X` to multiply by a vector or a tall, skinny matrix,
- `operator' * X` for the adjoint multiply. [`rsvd`](@ref) and the XTrace
  estimator need this; the Hermitian eigen routines only use `operator * X`.

A dense `Matrix`, a `SparseMatrixCSC`, a `CuMatrix`, a `LinearMaps.LinearMap`, or
your own type all qualify. If the temporaries need to live on a particular device
(CPU or GPU), pass a prototype array through the `sample_vec` keyword. For a
`LinearMap` backed by CPU FFTs, for example, the examples pass `sample_vec =
Float64[]`.
