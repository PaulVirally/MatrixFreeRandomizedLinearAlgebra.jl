using Documenter
using MatrixFreeRandomizedLinearAlgebra

makedocs(
    sitename = "MatrixFreeRandomizedLinearAlgebra.jl",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "What is matrix-free?" => "matrix_free.md",
        "Algorithms" => "algorithms.md",
        "Very large operators" => "large_operators.md",
        "Examples" => "examples.md",
        "API" => "api.md",
        "References" => "references.md",
    ],
)

deploydocs(
    repo = "github.com/PaulVirally/MatrixFreeRandomizedLinearAlgebra.jl.git",
)
