# Algorithms

This page describes, at a high level, what each routine does and where it comes
from. It is meant for intuition. For the details, see the [References](@ref).

## The randomized range finder

Every decomposition here is built on the same idea, from Halko, Martinsson, and
Tropp. Suppose `A` is well approximated by something of rank `k`. If you multiply
`A` by a tall, skinny matrix `Ω` whose columns are random Gaussian vectors, the
result `Y = A * Ω` mixes the columns of `A` together, and with high probability
its columns span most of the part of `A`'s range that matters. Orthonormalizing
`Y` with a thin QR factorization gives a matrix `Q` whose columns are an
approximate basis for the range of `A`. Everything after that happens inside that
small subspace, where the linear algebra is cheap.

Two knobs control how good `Q` is:

- Oversampling (`num_oversamples`). Instead of exactly `k` random columns you draw
  `k + p`. The extra columns make it unlikely that the random sketch misses an
  important direction. A modest `p` (the default uses `p = k`) buys a lot of
  robustness.
- Power iterations (`num_power_iterations`). When the singular values decay
  slowly, the plain sketch is contaminated by the trailing part of the spectrum.
  Replacing `A` with `(A A')^q A` for a small `q` pushes the unwanted directions
  down without changing the leading subspace. Each power iteration costs a couple
  more passes over the operator, so it trades accuracy for work.

The routines pick reasonable defaults for both, and you can override them.

## Randomized SVD: [`rsvd`](@ref), [`rsvdvals`](@ref)

Once you have the range basis `Q`, the rest is small and exact. Project the
operator into the subspace with `B = Q' * A`, take an ordinary SVD of the small
matrix `B`, and lift the left singular vectors back with `Q`. The result is a
rank-`k` SVD `A ≈ U Σ Vᵀ`, returned as a standard `LinearAlgebra.SVD`.
[`rsvdvals`](@ref) does the same but skips the singular vectors when you only need
the values. Tall operators (more rows than columns) are transposed internally so
the sketch stays small.

## Randomized Hermitian eigendecomposition: [`reigen_hermitian`](@ref), [`reigvals_hermitian`](@ref)

For a Hermitian operator the range finder also finds an invariant subspace, and
you can use a cheaper one-sided power iteration since `A` is its own adjoint. Form
the small projected matrix `B = Q' * A * Q`, eigendecompose it, and lift the
eigenvectors back with `Q`. The leading eigenpairs come out sorted in descending
order as a `LinearAlgebra.Eigen`. Eigenvalues need cleaner subspace separation
than singular values, so the defaults here use more power iterations than
[`rsvd`](@ref).

## Stochastic trace estimation: [`trace`](@ref)

The trace of an operator you can only apply is estimated from samples of the form
`zᵀ A z`. There are two estimators:

- Girard-Hutchinson (`low_mem = true`). Average `zᵀ A z` over random sign
  (Rademacher) vectors `z`. It is unbiased, holds only a vector or two at a time,
  and its error shrinks like `1/√m` in the number of samples `m`. This is the
  fallback for operators so large that even a small sketch won't fit.
- XTrace (default). The estimator of Epperly, Tropp, and Webber. It pairs a small
  randomized low-rank sketch (the same range-finder idea) with a leave-one-out
  Hutchinson correction, so it gets a lot more out of each matrix-vector product
  when the operator has any spectral decay. It needs a bit more memory, since it
  keeps the sketch around, but it usually reaches a given accuracy with far fewer
  applies.

Either estimator can run with a fixed sample budget or adaptively until it hits a
target `relative_tolerance`, and either can return an error estimate alongside the
value (`return_error = true`).

## A note on accuracy

These are randomized, approximate algorithms. They do not return the exact
decomposition a dense solver would. They return a high-probability approximation
whose accuracy you trade against cost through oversampling, power iterations, and
the sample budget. What you get in return is that the work scales with the target
rank and the cost of a matrix-vector product, not with the size of the dense
matrix. That is what makes them worth using on large, matrix-free operators where
a dense factorization isn't an option. The [Examples](@ref) page shows both
sides: how close the approximations land next to a dense reference, and how the
runtime pulls ahead as the operator grows.
