using ArgParse
using ITensors
using ITensorMPS
using JSON3
using Printf
using VAPSimulations

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_OUTPUT_DIR = joinpath(REPOSITORY_ROOT, "outputs", "noisy_tfi", "data")
const ANCILLA_TRACE_CUTOFF = 1e-15

function parse_commandline(args=ARGS)
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--size_A"
            arg_type = Int
            default = 2
        "--distance_AC"
            arg_type = Int
            default = 4
        "--distance_BB"
            arg_type = Int
            default = 4
        "--transverse_field"
            arg_type = Float64
            default = 1.0
        "--error_rates"
            arg_type = Float64
            nargs = '+'
            default = [0.0, 0.001, 0.002, 0.005, 0.01]
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

function validate_noisy_tfi_parameters(
    size_A,
    distance_AC,
    distance_BB,
    copies,
    error_rates,
    beta_min,
    beta_max,
    beta_step,
)
    size_A > 0 || throw(ArgumentError("size_A must be positive"))
    iseven(size_A) || throw(ArgumentError("size_A must be even"))
    distance_AC > 0 || throw(ArgumentError("distance_AC must be positive"))
    distance_BB >= 0 || throw(ArgumentError("distance_BB must be nonnegative"))
    isempty(copies) && throw(ArgumentError("number_of_copies must not be empty"))
    all(>(0), copies) || throw(ArgumentError("copy counts must be positive"))
    length(unique(copies)) == length(copies) ||
        throw(ArgumentError("copy counts must be unique"))
    isempty(error_rates) && throw(ArgumentError("error_rates must not be empty"))
    all(p -> 0 <= p <= 1, error_rates) ||
        throw(ArgumentError("error rates must satisfy 0 <= p <= 1"))
    length(unique(error_rates)) == length(error_rates) ||
        throw(ArgumentError("error rates must be unique"))
    validate_beta_grid(beta_min, beta_max, beta_step)
    return nothing
end

imaginary_time_steps_noisy(beta, beta_step) = round(Int, beta / beta_step)

function error_rate_filename_token(error_rate)
    return scaled_filename_token(error_rate, 1000, 3)
end

function main(cli_args=ARGS)
    args = parse_commandline(cli_args)
    size_A = args["size_A"]
    distance_AC = args["distance_AC"]
    distance_BB = args["distance_BB"]
    hx = args["transverse_field"]
    error_rates = args["error_rates"]
    copies = args["number_of_copies"]
    beta_min = args["beta_min"]
    beta_max = args["beta_max"]
    beta_step = args["beta_step"]
    output_dir = args["output_dir"]
    validate_noisy_tfi_parameters(
        size_A,
        distance_AC,
        distance_BB,
        copies,
        error_rates,
        beta_min,
        beta_max,
        beta_step,
    )

    beta_values = collect(beta_min:beta_step:beta_max)
    max_copies = maximum(copies)
    subregion_size = size_A + 2 * distance_AC
    total_size = max_copies * subregion_size + (max_copies + 1) * distance_BB
    println("Total system size: $total_size")

    physical_sites = siteinds("Qubit", total_size)
    extended_sites = siteinds("Qubit", 2 * total_size)
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

    hamiltonian_terms = OpSum()
    for site in 1:2:(2 * total_size - 3)
        hamiltonian_terms -= "Z", site, "Z", site + 2
    end
    for site in 1:2:(2 * total_size - 1)
        hamiltonian_terms -= hx, "X", site
    end
    extended_hamiltonian = MPO(hamiltonian_terms, extended_sites)

    local_terms = OpSum()
    center_left = distance_AC + fld(size_A, 2)
    center_right = center_left + 1
    local_terms -= "Z", center_left, "Z", center_right
    local_terms -= hx / 2, "X", center_left
    local_terms -= hx / 2, "X", center_right
    local_hamiltonian = MPO(local_terms, subregion_sites)

    energy_values = [
        Matrix{Float64}(undef, length(beta_values), max_copies) for _ in error_rates
    ]
    cost_values = [
        Matrix{Float64}(undef, length(beta_values), max_copies) for _ in error_rates
    ]
    channels = depolarizing_superoperator.(error_rates)

    for (beta_index, beta) in enumerate(beta_values)
        println("beta = $beta")
        nsteps = imaginary_time_steps_noisy(beta, beta_step)
        purification = MPS(extended_sites, "0")
        for site in 1:2:(2 * total_size - 1)
            purification = apply(op("H", extended_sites[site]), purification)
            purification = apply(op("CNOT", extended_sites[site:(site + 1)]), purification)
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
        )
        noiseless_rho = trace_out_ancilla(
            purification; cutoff=ANCILLA_TRACE_CUTOFF
        )
        noiseless_rho = replace_site_indices(noiseless_rho, physical_sites)

        for (rate_index, channel) in enumerate(channels)
            noisy_rho = apply_local_channel_all(noiseless_rho, channel)
            for n_copies in copies
                reduced = partial_trace_mpo(
                    noisy_rho, subregion_indices[n_copies]
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
                    next_copy = replace_site_indices(next_copy, subregion_sites)
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
                cost_values[rate_index][beta_index, n_copies] = vap_trace^(-2)
                normalized_vap = vap / vap_trace
                energy_values[rate_index][beta_index, n_copies] =
                    inner(normalized_vap, local_hamiltonian)
            end
        end
    end

    mkpath(output_dir)
    for (rate_index, error_rate) in enumerate(error_rates)
        for n_copies in copies
            file_name = @sprintf(
                "nA%d_dAC%d_dBB%d_hx%s_p%s_m%d_n%d.json",
                size_A,
                distance_AC,
                distance_BB,
                scaled_filename_token(hx, 10, 2),
                error_rate_filename_token(error_rate),
                max_copies,
                n_copies,
            )
            data = Dict(
                "parameters" => Dict(
                    "size_A" => size_A,
                    "distance_AC" => distance_AC,
                    "distance_BB" => distance_BB,
                    "hx" => hx,
                    "error_rate" => error_rate,
                    "number_of_copies" => n_copies,
                    "requested_copy_counts" => collect(copies),
                    "max_copy_count" => max_copies,
                    "total_size" => total_size,
                    "selected_site_indices" => subregion_indices[n_copies],
                    "beta_min" => beta_min,
                    "beta_max" => beta_max,
                    "beta_step" => beta_step,
                    "ancilla_trace_cutoff" => ANCILLA_TRACE_CUTOFF,
                ),
                "beta" => beta_values,
                "energy_vap" => energy_values[rate_index][:, n_copies],
                "cost_vap" => cost_values[rate_index][:, n_copies],
            )
            path = joinpath(output_dir, file_name)
            open(path, "w") do io
                JSON3.pretty(io, data)
            end
            println("Saved $path")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
