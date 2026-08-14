# Benchmarks

A [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) suite covering
every exported randomized algorithm (`rsvd`, `rsvdvals`, `reigen_hermitian`,
`reigvals_hermitian`, `trace` in its XTrace/Hutchinson × fixed/adaptive modes)
across dense, sparse, and matrix-free (`LinearMap`) operators, plus the internal
kernels they are built on (`qthin!`, `qrthin!`, `randomized_range_finder`,
`sphere_test_matrix`, `rademacher!`), plus a `panel` group that measures the
Funicular streaming path against the dense one.

Because this package targets operators large enough that a handful of extra
`n`-vectors can saturate RAM/VRAM, the reported *memory estimate* is as important
as the runtime.

Run a labeled sweep (results land in `benchmark/results/<label>.json`):

```bash
julia --startup-file=no --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()' # first time only
julia --startup-file=no --project=benchmark benchmark/benchmarks.jl mylabel
```

## The `panel` group

`SUITE["panel"]` measures what Funicular's tiered panel storage costs. Each
solve is registered twice on the same operator, once with `plan=` and once
through the plain dense path, so the pair of medians is a streaming-overhead
ratio that `compare.jl` tracks over time. The budgets are synthetic: a
`CPUBackend` plan has no device to run out of, so a 2 MiB `device_budget` plays
the part of one against a 2000 × 200 `Float64` sketch of 3.05 MiB, and the panel
width is pinned at 10 so that every machine cuts the sketch into the same 20
panels. The group needs Funicular.jl, which `benchmark/Project.toml` pulls from
its git URL.

Compare two sweeps (ratios < 1 mean the second run is better):

```bash
julia --startup-file=no --project=benchmark benchmark/compare.jl baseline mylabel
```

## GPU benchmarks

On a machine where CUDA is functional, the same sweep automatically adds
mirrored GPU benchmarks (groups prefixed `gpu-`): dense `CuMatrix` operators,
CUSPARSE sparse operators, and a CUFFT-based blur `LinearMap`, run entirely on
the device via `sample_vec = CUDA.zeros(Float64, 0)`. Each GPU benchmark is
smoke-tested once and skipped with a warning if the combination is unsupported,
so a partial failure never kills the sweep. No flags are needed — on machines
without a GPU the `gpu-` groups are simply absent.

BenchmarkTools only measures host allocations, so device memory is measured
separately (one probe call per benchmark via `CUDA.@timed`), shown in the
table's `gpu memory` column, and saved to `benchmark/results/<label>.gpumem.tsv`.
When both labels passed to `compare.jl` have that sidecar, it prints a GPU
device-memory comparison section as well.

Typical workflow for checking a change on a GPU machine:

```bash
git checkout <commit-before-change>
julia --startup-file=no --project=benchmark benchmark/benchmarks.jl before
git checkout <commit-with-change>
julia --startup-file=no --project=benchmark benchmark/benchmarks.jl after
julia --startup-file=no --project=benchmark benchmark/compare.jl before after
```
