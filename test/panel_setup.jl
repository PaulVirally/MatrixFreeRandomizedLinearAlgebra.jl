using Funicular

# Shared by every panel test file, so that the plan the tests run against is
# described in one place. Modelled on Funicular's own test/setup.jl: a CPU
# backend and budgets large enough that nothing spills, with the panel width
# pinned so that the small test matrices are still cut into several panels with
# a ragged last one.
panelplan(; device_budget=64 * 2^20, host_budget=64 * 2^20, kwargs...) = ResidencyPlan(; backend=CPUBackend(), device_budget=device_budget, host_budget=host_budget, kwargs...)
