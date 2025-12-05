using Documenter
using MatrixFreeRandomizedLinearAlgebra

makedocs(
    sitename = "MatrixFreeRandomizedLinearAlgebra.jl",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/PaulVirally/MatrixFreeRandomizedLinearAlgebra.jl.git",
)
