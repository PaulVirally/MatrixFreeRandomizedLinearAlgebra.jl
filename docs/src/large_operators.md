# Very large operators

Every routine here builds an `N × (k + p)` sketch, where `k` is `num_components`
and `p` is `num_oversamples`. For a matrix-free operator, nearly all of the
memory goes to that sketch. This page describes what to do when the sketch no
longer fits.

## The problem

Suppose the operator's matvecs have to run on a GPU, because the apply is
something like an FFT or a stencil that the device does far faster than the host.
The operator itself is matrix-free, so it costs almost nothing to store. The
sketch is a different matter: it is a dense array with one row per degree of
freedom, and it has to be held for the whole solve.

For example, a `10⁷`-row problem with `k + p = 1000` gives a `10⁷ × 1000`
`ComplexF64` sketch, which is 160 GB. There is no device with room for it, and
the range finder needs two of them at once. The usual answers are to shrink `k`
until it fits, or to move the whole computation to the host and give up the
device's throughput on the apply.

## The solution

Pass a [Funicular.jl](https://www.paulvirally.com/Funicular.jl/dev/)
`ResidencyPlan` through the `plan` keyword. Every entry point accepts one, and
with Funicular loaded it switches the tall matrices over to Funicular's tiered
panel storage:

- The random test matrices are not stored at all. They are `GhostPanels`, whose
  columns are regenerated from a seed and a column index whenever a sweep needs
  them, so the `Ω` that would have cost 160 GB costs nothing.
- The sketch and its images are `PanelMatrix` objects, cut into column panels. A
  panel at a time is staged onto the device, multiplied, and written back, while
  the plan prefetches the next one. What does not fit in device memory lives in
  pinned host memory, and what does not fit there spills to disk if the plan was
  given a `scratch_dir`.
- The small reduced block at the end of each algorithm (the `(k + p) × (k + p)`
  matrix that gets eigendecomposed or SVD'd) is formed on the host and finished
  exactly as the in-memory path finishes it.

The algorithms themselves are unchanged: only the storage of the large matrices,
and the order in which they are traversed, differs.

```julia
using CUDA, Funicular, MatrixFreeRandomizedLinearAlgebra

# G: a Hermitian matrix-free operator, 10^7 × 10^7, applied on the GPU.
plan = ResidencyPlan(
    backend = Funicular.cuda_backend(),
    device_budget = 0.8 * CUDA.total_memory(),  # leave the driver some room
    host_budget = 192 * 2^30,                   # pinned host memory for cold panels
    workspace_bytes = 3 * 2^30,                 # what G itself needs on the device
    scratch_dir = "/scratch/me/funicular",      # optional disk tier; needs HDF5.jl
)

E = reigen_hermitian(G, 512; num_oversamples=64, plan=plan, seed=1)
```

`E` is a [`PanelEigen`](@ref), not a `LinearAlgebra.Eigen`. Its `values` are an
ordinary host `Vector` of 512 eigenvalues in descending order; its `vectors` are
a `10⁷ × 512` Funicular `PanelMatrix`, which is 80 GB of `ComplexF64` and so is
still too large to collect into a dense array:

```julia
E.values                    # 512-element Vector, descending

Matrix(E.vectors)           # the dense eigenvectors, when they fit in host memory.
                            # This checks first and throws if they do not, instead
                            # of allocating past the memory that is there; raise
                            # max_bytes if the memory really is available.

# A panel at a time on the device instead, which is the practical option for a
# 10^7-row basis. `block` is a device-resident view of panelwidth(E.vectors, j)
# columns, valid only inside the call, and write=false makes the sweep read-only.
foreachpanel(E.vectors; write=false) do j, block
    consume(panelrange(E.vectors, j), block)
end
```

The other entry points behave the same way. `rsvd` returns a [`PanelSVD`](@ref)
holding `U`, `S`, and the *tall* `V` (a `k × n` `Vt` is short and wide, and only
a tall matrix can be cut into column panels), while `rsvdvals`,
`reigvals_hermitian`, and `trace` return plain host values as they always do:

```julia
σ = rsvdvals(A, 512; num_oversamples=64, plan=plan, seed=1)   # Vector{Float64}
t = trace(G, 128; plan=plan, seed=1)                          # a scalar
```

## The operator contract

On the in-memory path an operator needs `size`, `*`, and `'`. On the panel path
Funicular applies it to device columns instead, so it needs:

- `size(G)` and `eltype(G)`,
- `adjoint(G)`, returning something that is genuinely the adjoint and that has
  the transposed size,
- `LinearAlgebra.mul!(y, G, x)` for device vectors `x` and `y`.

Three optional traits control what Funicular assumes:

- `Funicular.workspace_bytes(G)`: device memory `G` needs for itself while it is
  applied, held back from the panel buffer pool. An operator backed by an FFT
  plan has to declare this. A CUFFT plan's work area is a real allocation that
  the plan cannot see, so if it goes unreported, the budget arithmetic gives the
  panel buffers memory that the operator will then take for itself, and the
  device overflows mid-sweep. Report it through `workspace_bytes(G)`, or pass it
  to the plan as the `workspace_bytes` keyword when `G` is a wrapper type you do
  not own (a `LinearMap`, for instance, has nowhere to carry it).
- `Funicular.panel_capable(G)`: whether `mul!(Y, G, X)` accepts a whole panel of
  several columns rather than one vector. Defaults to `false`, which makes
  Funicular loop over columns. An operator that can apply a batch of columns at
  once should set this to `true`.
- `Funicular.ishermitian_op(G)`: whether `G` is its own adjoint. Defaults to
  `false`, which is always safe.

`validate=true` (the default) runs `Funicular.check_operator` once on entry. It
costs a handful of probe multiplies and catches mistakes that would otherwise
show up as a wrong answer (for example, an `adjoint` that is actually a
transpose, or a `mul!` that assumes its input is contiguous). Turn it off once
the operator is known to be correct. See [Funicular's operators page](https://www.paulvirally.com/Funicular.jl/dev/operators/)
for the full contract.

## The cost model

Once the sketch is larger than device memory, the cost of a solve is dominated
by data movement over the bus rather than by flops, so the useful thing to count
is the number of times the data crosses. We call one traversal of an `N × s`
matrix a *sweep*:

- A `panelmul!` (one operator application to the whole sketch) is one sweep.
- A `cholqr2!` (the CholeskyQR2 orthonormalization that replaces the thin QR) is
  four: two passes, each forming a Gram matrix and then applying the inverse
  factor.

So a Hermitian power iteration is one `panelmul!` and one `cholqr2!`, about five
sweeps, and the default `num_power_iterations = 14` is around seventy of them. A
general (non-Hermitian) power iteration does the adjoint pass into the domain and
the forward pass back, each orthonormalized, so it is about ten sweeps.

Peak storage, in `N × s` panel matrices held at once:

| entry point | panel matrices at peak |
| --- | --- |
| `reigen_hermitian` / `reigvals_hermitian` | 2, during the power iteration |
| `rsvd` / `rsvdvals` | 3, while the two bases and one output are alive |
| `trace` (XTrace) | 3, during the reduction |

The test matrix is not in those counts because it is regenerated rather than
stored. For `trace`, `Ω` is a ghost and the three are the image `Y = A Ω`, its
orthonormalization `Q`, and `Z = A Q`.

A few things reduce the cost:

- Oversample less. `num_oversamples` defaults to `num_components`, which doubles
  the sketch: at `k = 512` the default adds another 512 columns of an `N`-row
  matrix, to be moved seventy-odd times. That default was picked for robustness
  at small `k`, not for a sketch this size. Five to ten percent is usually enough
  at scale, so `num_oversamples = 64` against `num_components = 512` costs a
  fraction of the memory and the sweeps.
- Use fewer power iterations. They are essentially the whole cost of the solve.
  If the spectrum decays quickly, `num_power_iterations = 2` or `3` may be as
  good as the default.
- Warm start. `seed_Q` takes a `PanelMatrix` or a [`PanelFactored`](@ref), so the
  vectors from a previous solve go straight back in. When marching a parameter or
  a time step, the previous basis is usually close enough that the iteration
  count can come down a long way.
- Pass `factored=true`. The last step of `rsvd` and `reigen_hermitian` rotates
  the range basis by a small dense matrix to produce the vectors, which costs a
  sweep and a second `N × k` panel matrix per factor. With `factored=true` the
  result holds the basis and the rotation unevaluated as a
  [`PanelFactored`](@ref) instead. If the vectors are only going to be applied to
  something, or fed back as a `seed_Q`, the product never has to be formed;
  [`materialize`](@ref) or `Matrix` forms it when it does.

## Numerical notes

The panel path orthonormalizes with CholeskyQR2 rather than with the Householder
QR the in-memory path uses, because CholeskyQR2 can be computed one row block at
a time. This comes at a cost in conditioning: CholeskyQR2 forms a Gram matrix,
which squares the condition number before it is factored, so it only holds up to about
`κ ≲ eps^(-1/2)`, roughly `1e8` in double precision. Past that, Funicular falls
back to a shifted CholeskyQR3, which costs a third pass and keeps a numerically
rank-deficient sketch from failing outright.

That fallback is only guaranteed to recover the *values*. The orthonormality of
the *basis* degrades: columns that were linearly dependent come back as columns
of `U`, `V`, or `vectors` that are not meaningfully orthonormal, and a block too
dependent for the shift to rescue raises instead. Singular values and eigenvalues
from such a solve are still usable, but the vectors may not be. When rank
deficiency cannot be ruled out, check it:

```julia
opnorm(Matrix(F.U)' * Matrix(F.U) - I)
```

In practice the fix is to keep the sketch well conditioned in the first place:
oversample less aggressively, or ask for fewer components. A sketch wider than the operator's
smaller dimension is guaranteed to be dependent, and is capped for that reason.

Reproducibility works differently here too. The in-memory path draws from the
global RNG, so `Random.seed!` fixes it. The panel path has no stored test matrix
to seed, so it takes an integer `seed` instead, from which each ghost column is
generated using its own column index. That has two useful consequences and one caveat:

- The same `seed` gives the same test matrix regardless of the panel width or the
  plan's budgets, so a run can be reproduced on a machine with a different amount
  of device memory.
- It reproduces only up to rounding. The panel width sets the row-block height of
  the reductions, so changing it changes the order in which sums are accumulated.
  Values agree to within the usual floating-point tolerance rather than bit for
  bit.
- It holds within one Julia version only. Funicular generates columns from `hash`
  and the `Xoshiro` stream, both of which are Julia implementation details. Two
  runs on different versions that have to agree need the test matrix written out
  once with `Funicular.save` and loaded thereafter.

Passing no `seed` draws a fresh one from the global RNG, so repeated calls sketch
differently, as they do in memory.

## When not to use it

Passing a plan is not always the right choice:

- The sketch fits comfortably on the device. A `PanelMatrix` keeps its panels in
  host memory, so every sweep pays an upload and a writeback that a resident
  array does not. The pipeline hides most of that behind the operator applies,
  but it cannot remove it entirely, so a plain `CuArray` sketch is faster
  whenever it fits. Use `sample_vec` to put the temporaries on the device and
  leave `plan` alone.
- You are calling `trace(...; low_mem=true)`. The streaming Girard-Hutchinson
  estimator only ever holds one or two vectors, so there is no sketch to stream
  and a plan has nothing to do; passing both raises an `ArgumentError` rather
  than silently ignoring one of them. If even the panel path cannot afford
  XTrace, use `low_mem` instead of a plan, not alongside one.
- The operator is CPU-only and there is plenty of RAM. The point of the panel
  path is to keep a device fed with work when the data lives elsewhere. With no
  device in the picture and the sketch already in host memory, the panel
  machinery only adds bookkeeping.
- The operator is cheap next to a transfer. If applying `G` to a column takes
  less time than moving that column across the bus, there is no compute to hide
  the transfers behind, and the total runtime is set by the transfers. Funicular's
  [when to use it](https://www.paulvirally.com/Funicular.jl/dev/when-to-use/)
  page has the arithmetic, and a benchmark for measuring the ratio first.
