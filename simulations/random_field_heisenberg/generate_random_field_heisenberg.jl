using ArgParse
using ITensors
using ITensorMPS
using JSON3
using Printf
using Random
using VAPSimulations

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_OUTPUT_DIR =
    joinpath(REPOSITORY_ROOT, "outputs", "random_field_heisenberg", "data")
const ANCILLA_TRACE_CUTOFF = 1e-15
const TDVP_CUTOFF = 1e-6

function parse_commandline(args=ARGS)
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--size_A"
            arg_type = Int
            default = 2
        "--distance_AC"
            arg_type = Int
            default = 3
        "--distance_BB"
            arg_type = Int
            default = 3
        "--Delta"
            arg_type = Float64
            default = 1.0
        "--random_fields"
            arg_type = Float64
            nargs = '+'
            default = [0.5, 1.0]
        "--random_rots"
            default = "1:10"
        "--base_seed"
            arg_type = Int
            default = 1234
        "--number_of_copies"
            arg_type = Int
            nargs = '+'
            default = collect(1:8)
        "--beta_min"
            arg_type = Float64
            default = 0.0
        "--beta_max"
            arg_type = Float64
            default = 0.5
        "--beta_step"
            arg_type = Float64
            default = 0.1
        "--output_dir"
            default = DEFAULT_OUTPUT_DIR
    end
    return parse_args(args, settings)
end

function parse_realization_range(specification::AbstractString)
    bounds = split(specification, ':'; keepempty=false)
    length(bounds) == 2 ||
        throw(ArgumentError("random_rots must have START:STOP form"))
    first_realization, last_realization = try
        parse.(Int, bounds)
    catch
        throw(ArgumentError("random_rots must contain integer bounds"))
    end
    return first_realization:last_realization
end

function validate_random_field_heisenberg_parameters(
    size_A,
    distance_AC,
    distance_BB,
    random_fields,
    realizations,
    copies,
    beta_min,
    beta_max,
    beta_step,
)
    size_A > 0 || throw(ArgumentError("size_A must be positive"))
    iseven(size_A) || throw(ArgumentError("size_A must be even"))
    distance_AC > 0 || throw(ArgumentError("distance_AC must be positive"))
    distance_BB >= 0 || throw(ArgumentError("distance_BB must be nonnegative"))
    isempty(random_fields) && throw(ArgumentError("random_fields must not be empty"))
    all(>=(0), random_fields) ||
        throw(ArgumentError("random field strengths must be nonnegative"))
    length(unique(random_fields)) == length(random_fields) ||
        throw(ArgumentError("random field strengths must be unique"))
    isempty(realizations) && throw(ArgumentError("random_rots must not be empty"))
    all(>(0), realizations) ||
        throw(ArgumentError("realization indices must be positive"))
    isempty(copies) && throw(ArgumentError("number_of_copies must not be empty"))
    all(>(0), copies) || throw(ArgumentError("copy counts must be positive"))
    length(unique(copies)) == length(copies) ||
        throw(ArgumentError("copy counts must be unique"))
    validate_beta_grid(beta_min, beta_max, beta_step)
    return nothing
end

imaginary_time_steps_random_field(beta, beta_step) = round(Int, beta / beta_step)

function field_strength_filename_token(field_strength)
    return scaled_filename_token(field_strength, 10, 2)
end

function sample_random_fields(total_size, field_strength, base_seed, realization)
    realization_seed = base_seed + realization
    # Preserve the scalar default-RNG draws used by the historical generator.
    Random.seed!(realization_seed)
    fields = [field_strength * (2 * rand() - 1) for _ in 1:total_size]
    return realization_seed, fields
end

function main(cli_args=ARGS)
    args = parse_commandline(cli_args)
    size_A = args["size_A"]
    distance_AC = args["distance_AC"]
    distance_BB = args["distance_BB"]
    anisotropy = args["Delta"]
    random_fields = args["random_fields"]
    realizations = parse_realization_range(args["random_rots"])
    base_seed = args["base_seed"]
    copies = args["number_of_copies"]
    beta_min = args["beta_min"]
    beta_max = args["beta_max"]
    beta_step = args["beta_step"]
    output_dir = args["output_dir"]
    validate_random_field_heisenberg_parameters(
        size_A,
        distance_AC,
        distance_BB,
        random_fields,
        realizations,
        copies,
        beta_min,
        beta_max,
        beta_step,
    )

    beta_values = collect(beta_min:beta_step:beta_max)
    max_copies = maximum(copies)
    subregion_size = size_A + 2 * distance_AC
    total_size = max_copies * subregion_size + (max_copies + 1) * distance_BB
    extended_size = 2 * total_size
    println("Total system size: $total_size")

    physical_sites = siteinds("Qubit", total_size)
    extended_sites = siteinds("Qubit", extended_size)
    subregion_sites = siteinds("Qubit", subregion_size)

    subregion_indices = Dict{Int,Vector{Int}}()
    for n_copies in copies
        copy_spacing = fld(total_size - distance_BB, n_copies)
        edge_distance = fld(total_size - n_copies * copy_spacing, 2)
        center_distance = fld(copy_spacing, 2)
        half_subregion = fld(subregion_size, 2)
        first_center = edge_distance + center_distance
        subregion_indices[n_copies] = reduce(
            vcat,
            (
                (first_center + (copy_index - 1) * copy_spacing - half_subregion + 1):
                (first_center + (copy_index - 1) * copy_spacing + half_subregion) for
                copy_index in 1:n_copies
            ),
        )
    end

    local_left = distance_AC + fld(size_A, 2)
    local_right = local_left + 1
    mkpath(output_dir)

    for field_strength in random_fields
        for realization in realizations
            println("Field strength $field_strength, realization $realization")
            realization_seed, sampled_fields = sample_random_fields(
                total_size, field_strength, base_seed, realization
            )

            hamiltonian_terms = OpSum()
            for site in 1:2:(extended_size - 3)
                hamiltonian_terms +=
                    anisotropy, "Sz", site, "Sz", site + 2
                hamiltonian_terms += 0.5, "S+", site, "S-", site + 2
                hamiltonian_terms += 0.5, "S-", site, "S+", site + 2
            end
            for site in 1:2:(extended_size - 1)
                hamiltonian_terms +=
                    sampled_fields[(site + 1) ÷ 2], "Sz", site
            end
            extended_hamiltonian = MPO(hamiltonian_terms, extended_sites)

            energy_values =
                Matrix{Float64}(undef, length(beta_values), max_copies)
            cost_values =
                Matrix{Float64}(undef, length(beta_values), max_copies)

            for (beta_index, beta) in enumerate(beta_values)
                println(@sprintf("  beta = %0.3f", beta))
                nsteps = imaginary_time_steps_random_field(beta, beta_step)
                purification = MPS(extended_sites, "0")
                for site in 1:2:(extended_size - 1)
                    purification =
                        apply(op("H", extended_sites[site]), purification)
                    purification = apply(
                        op("CNOT", extended_sites[site:(site + 1)]),
                        purification,
                    )
                end
                purification = tdvp(
                    -0.5 * extended_hamiltonian,
                    beta_step * nsteps,
                    purification;
                    time_step=beta_step,
                    nsteps=nsteps,
                    normalize=true,
                    nsite=2,
                    outputlevel=0,
                    cutoff=TDVP_CUTOFF,
                )
                rho = trace_out_ancilla(
                    purification; cutoff=ANCILLA_TRACE_CUTOFF
                )
                rho = replace_site_indices(rho, physical_sites)

                for n_copies in copies
                    reduced = partial_trace_mpo(
                        rho, subregion_indices[n_copies]
                    )
                    vap = MPO(subregion_sites)
                    for site in eachindex(subregion_sites)
                        vap[site] = reduced[site]
                    end
                    vap = replace_site_indices(vap, subregion_sites)

                    for copy_index in 2:n_copies
                        next_copy = MPO(subregion_sites)
                        for site in eachindex(subregion_sites)
                            source = if iseven(copy_index)
                                copy_index * subregion_size - site + 1
                            else
                                (copy_index - 1) * subregion_size + site
                            end
                            next_copy[site] = reduced[source]
                        end
                        next_copy = replace_site_indices(
                            next_copy, subregion_sites
                        )
                        vap = apply(vap, next_copy)
                        open_link = if iseven(copy_index)
                            siteinds(vap; tags="Link", plev=0)[end]
                        else
                            siteinds(next_copy; tags="Link", plev=0)[1]
                        end
                        boundary = iseven(copy_index) ? length(vap) : 1
                        vap[boundary] *= delta(open_link, open_link')
                    end

                    vap_trace = tr(vap)
                    cost_values[beta_index, n_copies] = vap_trace^(-2)
                    normalized_vap = vap / vap_trace

                    energy = 0.0
                    for copy_index in 1:n_copies
                        local_terms = OpSum()
                        local_terms +=
                            anisotropy, "Sz", local_left, "Sz", local_right
                        local_terms +=
                            0.5, "S+", local_left, "S-", local_right
                        local_terms +=
                            0.5, "S-", local_left, "S+", local_right
                        global_left = subregion_indices[n_copies][
                            (copy_index - 1) * subregion_size + local_left
                        ]
                        global_right = global_left + 1
                        # The central pair swaps physical sites when the copy is folded.
                        if iseven(copy_index)
                            global_left, global_right = global_right, global_left
                        end
                        local_terms +=
                            0.5 * sampled_fields[global_left], "Sz", local_left
                        local_terms +=
                            0.5 * sampled_fields[global_right], "Sz", local_right
                        local_hamiltonian = MPO(
                            local_terms, subregion_sites
                        )
                        energy += inner(normalized_vap, local_hamiltonian)
                    end
                    energy_values[beta_index, n_copies] = energy / n_copies
                end
            end

            for n_copies in copies
                file_name = @sprintf(
                    "nA%d_dAC%d_dBB%d_D%s_W%s_m%d_n%d_r%02d.json",
                    size_A,
                    distance_AC,
                    distance_BB,
                    scaled_filename_token(anisotropy, 10, 2),
                    field_strength_filename_token(field_strength),
                    max_copies,
                    n_copies,
                    realization,
                )
                data = Dict(
                    "parameters" => Dict(
                        "size_A" => size_A,
                        "distance_AC" => distance_AC,
                        "distance_BB" => distance_BB,
                        "Delta" => anisotropy,
                        "random_field" => field_strength,
                        "realization" => realization,
                        "base_seed" => base_seed,
                        "realization_seed" => realization_seed,
                    "rng" => string(typeof(Random.default_rng())),
                    "julia_version" => string(VERSION),
                        "sampled_fields" => sampled_fields,
                        "number_of_copies" => n_copies,
                        "requested_copy_counts" => collect(copies),
                        "max_copy_count" => max_copies,
                        "total_size" => total_size,
                        "selected_site_indices" =>
                            subregion_indices[n_copies],
                        "energy_observable_convention" => "folded_physical_sites",
                        "beta_min" => beta_min,
                        "beta_max" => beta_max,
                        "beta_step" => beta_step,
                        "ancilla_trace_cutoff" => ANCILLA_TRACE_CUTOFF,
                        "tdvp_cutoff" => TDVP_CUTOFF,
                    ),
                    "beta" => beta_values,
                    "energy_vap" => energy_values[:, n_copies],
                    "cost_vap" => cost_values[:, n_copies],
                )
                path = joinpath(output_dir, file_name)
                open(path, "w") do io
                    JSON3.pretty(io, data)
                end
                println("Saved $path")
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
