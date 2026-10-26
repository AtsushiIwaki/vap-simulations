using Test
using Random

module PaperTfiDefaults
    include(joinpath(@__DIR__, "..", "simulations", "tfi_overhead_scaling", "generate_tfi_overhead_scaling.jl"))
end

module PaperDisorderDefaults
    include(joinpath(@__DIR__, "..", "simulations", "random_field_heisenberg", "generate_random_field_heisenberg.jl"))
end

@testset "paper reproduction parameters" begin
    @testset "temperature samples and integration steps" begin
        defaults = PaperTfiDefaults.parse_commandline(String[])
        @test collect(defaults["beta_min"]:defaults["beta_step"]:defaults["beta_max"]) == collect(0.0:0.2:1.0)
        step = get(defaults, "time_step", defaults["beta_step"])
        @test PaperTfiDefaults.imaginary_time_steps(1.0, step) == 10
    end
    @testset "legacy scalar random-field draws" begin
        Random.seed!(1235)
        legacy_fields = [2 * rand() - 1 for _ in 1:4]
        seed, fields = PaperDisorderDefaults.sample_random_fields(4, 1.0, 1234, 1)
        @test seed == 1235
        @test fields == legacy_fields
    end
end
