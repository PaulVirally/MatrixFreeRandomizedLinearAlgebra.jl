# API Reference

```@autodocs
Modules = [MatrixFreeRandomizedLinearAlgebra]
Order = [:type, :function]
Filter = t -> !(t in (PanelEigen, PanelSVD, PanelFactored, materialize))
```

## Panel results

These are what the entry points return when they are given a Funicular
`ResidencyPlan` through the `plan` keyword, in place of `LinearAlgebra.Eigen`
and `LinearAlgebra.SVD`. See [Very large operators](@ref) for when that path is
worth taking.

```@docs
PanelEigen
PanelSVD
PanelFactored
materialize
```
