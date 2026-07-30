# Compare two saved benchmark runs.
#
#   julia --project=benchmark benchmark/compare.jl baseline optimized
#
# Prints median-time and memory ratios (new / old); < 1 means the new run is
# better. Uses BenchmarkTools' judge() thresholds to flag improvements and
# regressions.

using BenchmarkTools
using Printf

function flatten(group, prefix="")
    out = Dict{String,BenchmarkTools.Trial}()
    for (name, v) in group
        key = isempty(prefix) ? String(name) : "$prefix/$name"
        if v isa BenchmarkTools.Trial
            out[key] = v
        else
            merge!(out, flatten(v, key))
        end
    end
    return out
end

old_label, new_label = ARGS[1], ARGS[2]
resdir = joinpath(@__DIR__, "results")
old = flatten(BenchmarkTools.load(joinpath(resdir, "$old_label.json"))[1])
new = flatten(BenchmarkTools.load(joinpath(resdir, "$new_label.json"))[1])

println("Comparing benchmark results: $old_label vs $new_label. A ratio < 1 means $new_label is better than $old_label.\n")

@printf("%-45s %11s %11s %8s %11s %11s %8s\n",
    "benchmark", "t($old_label)", "t($new_label)", "ratio", "mem($old_label)", "mem($new_label)", "ratio")
println("-"^102)
for key in sort(collect(keys(old)))
    haskey(new, key) || continue
    to, tn = time(median(old[key])), time(median(new[key]))
    mo, mn = memory(old[key]), memory(new[key])
    @printf("%-45s %11s %11s %7.2fx %11s %11s %7.2fx\n",
        key,
        BenchmarkTools.prettytime(to), BenchmarkTools.prettytime(tn), tn / to,
        BenchmarkTools.prettymemory(mo), BenchmarkTools.prettymemory(mn), mo == 0 ? NaN : mn / mo)
end

# GPU device memory, when both runs saved a `<label>.gpumem.tsv` sidecar
# (written by benchmarks.jl on CUDA-functional machines). BenchmarkTools trials
# only record host allocations, so this is where GPU memory differences show up.
read_gpumem(path) = isfile(path) ?
    Dict(String(k) => parse(Int, v) for (k, v) in (split(line, '\t') for line in eachline(path))) :
    nothing
gold = read_gpumem(joinpath(resdir, "$old_label.gpumem.tsv"))
gnew = read_gpumem(joinpath(resdir, "$new_label.gpumem.tsv"))
if gold !== nothing && gnew !== nothing
    println("\nGPU device memory (bytes allocated per call):")
    @printf("%-45s %11s %11s %8s\n", "benchmark", "gpu($old_label)", "gpu($new_label)", "ratio")
    println("-"^78)
    for key in sort(collect(keys(gold)))
        haskey(gnew, key) || continue
        go, gn = gold[key], gnew[key]
        @printf("%-45s %11s %11s %7.2fx\n",
            key,
            BenchmarkTools.prettymemory(go), BenchmarkTools.prettymemory(gn), go == 0 ? NaN : gn / go)
    end
end
