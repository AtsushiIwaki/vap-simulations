using Test
using ITensors
using ITensorMPS
using VAPSimulations

@testset "sweep validation and filename encoding" begin
    @test isnothing(validate_beta_grid(0.0, 0.5, 0.1))
    @test_throws ArgumentError validate_beta_grid(0.05, 0.25, 0.1)
    @test scaled_filename_token(0.0001, 1000, 3) !=
          scaled_filename_token(0.0002, 1000, 3)
end

@testset "partial trace and index replacement" begin
    sites = siteinds("Qubit", 3)
    psi = MPS(sites, ["0", "0", "0"])
    rho = projector(psi)

    reduced = partial_trace_mpo(rho, [1, 3])
    @test length(reduced) == 2
    @test isapprox(real(tr(reduced)), real(tr(rho)); atol=1e-12)
    @test_throws ArgumentError partial_trace_mpo(rho, Int[])
    @test_throws ArgumentError partial_trace_mpo(rho, [0, 1])

    new_sites = siteinds("Qubit", 3)
    replaced = replace_site_indices(rho, new_sites)
    @test all(siteind(replaced, i) == new_sites[i] for i in eachindex(new_sites))
    @test_throws DimensionMismatch replace_site_indices(rho, new_sites[1:2])
end

@testset "ancilla trace" begin
    sites = siteinds("Qubit", 4)
    psi = MPS(sites, ["0", "0", "0", "0"])
    rho = trace_out_ancilla(psi; cutoff=1e-12)
    @test length(rho) == 2
    @test isapprox(real(tr(rho)), 1.0; atol=1e-12)
    @test_throws ArgumentError trace_out_ancilla(
        MPS(siteinds("Qubit", 3), ["0", "0", "0"])
    )
end

@testset "depolarizing channel" begin
    sites = siteinds("Qubit", 2)
    rho = projector(MPS(sites, ["0", "0"]))
    channel = depolarizing_superoperator(0.0)
    unchanged = apply_local_channel_all(rho, channel)
    @test norm(prod(unchanged) - prod(rho)) <= 1e-12
    @test_throws ArgumentError depolarizing_superoperator(-0.1)
    @test_throws ArgumentError depolarizing_superoperator(1.1)
    @test_throws BoundsError apply_local_channel(rho, 3, channel)
end

include(
    joinpath(
        @__DIR__,
        "..",
        "simulations",
        "tfi_overhead_scaling",
        "generate_tfi_overhead_scaling.jl",
    ),
)

@testset "TFI overhead parameter validation" begin
    @test_throws ArgumentError validate_parameters(2, 4, 4, Int[], 0.0, 0.5, 0.1)
    @test_throws ArgumentError validate_parameters(2, 4, 4, [1, 1], 0.0, 0.5, 0.1)
    @test_throws ArgumentError validate_parameters(2, 4, 4, [1], 0.0, 0.5, 0.0)
    @test_throws ArgumentError validate_parameters(2, 4, 4, [1], 0.05, 0.25, 0.1)
    @test isnothing(validate_parameters(2, 4, 4, [1, 2], 0.0, 0.5, 0.1))
    @test imaginary_time_steps(0.2, 0.1) == 2
end

include(
    joinpath(
        @__DIR__,
        "..",
        "simulations",
        "noisy_tfi",
        "generate_noisy_tfi.jl",
    ),
)

@testset "noisy TFI parameter validation" begin
    @test_throws ArgumentError validate_noisy_tfi_parameters(
        2, 4, 4, [1], Float64[], 0.0, 0.5, 0.1
    )
    @test_throws ArgumentError validate_noisy_tfi_parameters(
        2, 4, 4, [1], [-0.1], 0.0, 0.5, 0.1
    )
    @test_throws ArgumentError validate_noisy_tfi_parameters(
        2, 4, 4, [1], [1.1], 0.0, 0.5, 0.1
    )
    @test_throws ArgumentError validate_noisy_tfi_parameters(
        2, 4, 4, [1, 1], [0.0], 0.0, 0.5, 0.1
    )
    @test_throws ArgumentError validate_noisy_tfi_parameters(
        2, 4, 4, [1], [0.0], 0.05, 0.25, 0.1
    )
    @test error_rate_filename_token(0.0001) != error_rate_filename_token(0.0002)
    @test isnothing(
        validate_noisy_tfi_parameters(
            2, 4, 4, [1, 2], [0.0, 0.001], 0.0, 0.5, 0.1
        ),
    )
end

include(
    joinpath(
        @__DIR__,
        "..",
        "simulations",
        "random_field_heisenberg",
        "generate_random_field_heisenberg.jl",
    ),
)

@testset "random-field Heisenberg parameter validation" begin
    @test parse_realization_range("1:3") == 1:3
    @test_throws ArgumentError parse_realization_range("1")
    seed, fields = sample_random_fields(4, 1.0, 1234, 1)
    repeated_seed, repeated_fields = sample_random_fields(4, 1.0, 1234, 1)
    _, other_fields = sample_random_fields(4, 1.0, 1234, 2)
    @test seed == repeated_seed == 1235
    @test fields == repeated_fields
    @test fields != other_fields
    @test_throws ArgumentError validate_random_field_heisenberg_parameters(
        2, 3, 3, Float64[], 1:1, [1], 0.0, 0.5, 0.1
    )
    @test_throws ArgumentError validate_random_field_heisenberg_parameters(
        2, 3, 3, [1.0], 1:0, [1], 0.0, 0.5, 0.1
    )
    @test_throws ArgumentError validate_random_field_heisenberg_parameters(
        2, 3, 3, [1.0], 1:1, [1, 1], 0.0, 0.5, 0.1
    )
    @test_throws ArgumentError validate_random_field_heisenberg_parameters(
        2, 3, 3, [1.0], 1:1, [1], 0.0, 0.5, 0.0
    )
    @test_throws ArgumentError validate_random_field_heisenberg_parameters(
        2, 3, 3, [1.0], 1:1, [1], 0.05, 0.25, 0.1
    )
    @test field_strength_filename_token(0.51) !=
          field_strength_filename_token(0.52)
    @test isnothing(
        validate_random_field_heisenberg_parameters(
            2, 3, 3, [0.5, 1.0], 1:2, [1, 2], 0.0, 0.5, 0.1
        ),
    )
end

include(joinpath(@__DIR__, "test_paper_parameters.jl"))
include(joinpath(@__DIR__, "test_random_field_observable.jl"))
