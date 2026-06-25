using Test
using MatrixFreeRandomizedLinearAlgebra

@testset "MatrixFreeRandomizedLinearAlgebra" begin
    include("aqua_tests.jl")
    include("common_tests.jl")
    include("rsvd_tests.jl")
    include("rsvdvals_tests.jl")
    include("reigen_hermitian_tests.jl")
    include("reigvals_hermitian_tests.jl")
    include("trace_tests.jl")
end
