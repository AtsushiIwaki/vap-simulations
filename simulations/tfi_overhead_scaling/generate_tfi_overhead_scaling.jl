# ------------------------------------------------------------------------------
# VAP/LVP simulation for the 1D transverse-field Ising (TFI) model using ITensors.
#
# This script:
#   - Builds a purified thermal state of a 1D TFI chain via imaginary-time
#     evolution using TDVP on an enlarged (physical + ancilla) system.
#   - Traces out ancillas to obtain a mixed state on the physical sites.
#   - For a user-specified list of copy numbers {n}, constructs:
#       * VAP  (Virtual Autopurification): a construction based on multiple
#          spatial copies of an A–B–C subregion arranged along the chain.
#       * LVP  (Localized Virtual Purification): a localized construction based
#          on the reduced state of a single A–B–C block, iteratively applied n times.
#   - Evaluates the local energy on an A–C subregion and cost functions for
#     VAP and LVP as functions of inverse temperature β.
#   - Outputs the results for each n in JSON format under results/tfi_vap_lvp_overhead/.
#
# All physical and geometric parameters (|A|, distances, field strength, and
# copy numbers) are given as command-line arguments, parsed via ArgParse.
# ------------------------------------------------------------------------------

using ArgParse
using ITensors
using ITensorMPS
using JSON3
using Printf
using VAPSimulations

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_OUTPUT_DIR =
    joinpath(REPOSITORY_ROOT, "outputs", "tfi_overhead_scaling", "data")

"""
Parse command-line options using ArgParse.

Each option corresponds to a physical parameter of the model or
a list of copy numbers for the VAP/LVP constructions:

  --size_A           : number of spins in region A inside each copy |A|
  --distance_AC      : distance between regions A and C
  --distance_BB      : distance between B regions across different copies
  --transverse_field : transverse field strength hx in the TFI model
  --number_of_copies : list of copy numbers used in the VAP (Virtual Autopurification)
                       and LVP (Localized Virtual Purification) constructions (e.g. 1 2 3 4)
"""
function parse_commandline(args=ARGS)
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--size_A"
            arg_type = Int
            default = 2
            help = "Size of region A inside each copy"
        "--distance_AC"
            arg_type = Int
            default = 4
            help = "Distance between region A and C"
        "--distance_BB"
            arg_type = Int
            default = 4
            help = "Distance between the B regions across copies"
        "--transverse_field"
            arg_type = Float64
            default = 1.0
            help = "Transverse field strength hx in the TFI model"
        "--number_of_copies"
            arg_type = Int
            nargs = '+'
            default = [1, 2, 3, 4]
            help = "List of copy numbers used in the VAP/LVP constructions, e.g. 1 2 3 4"
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

function validate_parameters(
    size_A,
    distance_AC,
    distance_BB,
    copies,
    beta_min,
    beta_max,
    beta_step,
)
    size_A > 0 || throw(ArgumentError("size_A must be positive"))
    distance_AC > 0 || throw(ArgumentError("distance_AC must be positive"))
    distance_BB >= 0 || throw(ArgumentError("distance_BB must be nonnegative"))
    isempty(copies) && throw(ArgumentError("number_of_copies must not be empty"))
    all(>(0), copies) || throw(ArgumentError("copy counts must be positive"))
    length(unique(copies)) == length(copies) ||
        throw(ArgumentError("copy counts must be unique"))
    validate_beta_grid(beta_min, beta_max, beta_step)
    return nothing
end

imaginary_time_steps(beta, beta_step) = round(Int, beta / beta_step)

function main(cli_args=ARGS)

    # === Parse arguments ===
    args = parse_commandline(cli_args)

    # Physical parameters and copy list
    N_A           = args["size_A"]            # number of spins in subsystem A
    dist_AC       = args["distance_AC"]       # separation between regions A and C
    dist_BB       = args["distance_BB"]       # spacing between B regions (between copies)
    hx            = args["transverse_field"]  # transverse field strength
    n_copies_list = args["number_of_copies"]  # list of copy numbers for VAP/LVP
    beta_min      = args["beta_min"]
    beta_max      = args["beta_max"]
    beta_step     = args["beta_step"]
    output_dir    = args["output_dir"]
    validate_parameters(
        N_A,
        dist_AC,
        dist_BB,
        n_copies_list,
        beta_min,
        beta_max,
        beta_step,
    )

    # Use the maximum number of copies in the list to determine the system size
    max_n = maximum(n_copies_list)

    # === Compute total system size ===
    # Total number of physical sites derived from the repeating A–B–C block geometry
    total_sites    = (max_n + 1) * dist_BB + 2 * max_n * dist_AC + max_n * N_A
    @show total_sites
    subregion_size = 2 * dist_AC + N_A  # size of one A–B–C block

    # Site indices:
    #   phys_sites : physical chain
    #   ext_sites  : extended system (physical + ancilla)
    #   sub_sites  : indices of a single A–B–C region (used for local Hamiltonian)
    phys_sites = siteinds("Qubit", total_sites)
    ext_sites  = siteinds("Qubit", 2 * total_sites)
    sub_sites  = siteinds("Qubit", subregion_size)

    # === Indices of the subregion for partial trace ===
    # For each n_copies = 1..max_n, define the set of physical sites that make up
    # the union of n_copies consecutive A–B–C blocks starting from the left.
    #
    # sub_indices_dict[n] is a vector of site indices used in the partial trace
    # when constructing the reduced density matrix for n copies.
    sub_indices_dict = Vector{Vector{Int}}(undef, max_n)
    for n_copies in 1:max_n
        sub_indices = Int[]
        for a = 1:n_copies
            # Starting site of the a-th A–B–C block
            start = (a - 1) * (dist_BB + subregion_size) + dist_BB + 1
            stop  = start + subregion_size - 1
            append!(sub_indices, collect(start:stop))
        end
        sub_indices_dict[n_copies] = sub_indices
    end

    # === Construct global Hamiltonian on extended system ===
    # H_ext: MPO representation of the TFI Hamiltonian acting on the extended
    # system (physical + ancilla), where physical sites are on odd indices
    # (1, 3, 5, ...) of ext_sites.
    os = OpSum()
    # Nearest-neighbor ZZ interactions between physical sites
    for j = 1:2:2 * total_sites - 3
        os -= "Z", j, "Z", j + 2
    end
    # Transverse field terms on physical sites
    for j = 1:2:2 * total_sites - 1
        os -= hx, "X", j
    end
    H_ext = MPO(os, ext_sites)

    # === Construct local Hamiltonian (A–C region only) ===
    # H_local acts only on a single A–C region inside sub_sites.
    os = OpSum()
    os -= "Z", dist_AC + 1, "Z", dist_AC + 2
    os -= hx / 2, "X", dist_AC + 1
    os -= hx / 2, "X", dist_AC + 2
    H_local = MPO(os, sub_sites)

    # === β (imaginary time) sweep ===
    beta_values = collect(beta_min:beta_step:beta_max)

    # Tables of observables:
    #   rows    : beta index
    #   columns : copy number (indexed by n = 1..max_n)
    energy_values = Matrix{Float64}(undef, length(beta_values), max_n)
    cost_vap      = Matrix{Float64}(undef, length(beta_values), max_n)
    cost_lvp      = Matrix{Float64}(undef, length(beta_values), max_n)

    # === Loop over imaginary-time β ===
    for i in eachindex(beta_values)

        beta   = beta_values[i]
        nsteps = imaginary_time_steps(beta, beta_step)
        @show beta

        # --- Construct initial purification state ---
        # Initialize each physical + ancilla pair in a maximally entangled state
        # using a Hadamard on the physical spin followed by a CNOT.
        psi = MPS(ext_sites, "0")
        for j = 1:2:2 * total_sites - 1
            psi = apply(op("H", ext_sites[j]), psi)
            psi = apply(op("CNOT", ext_sites[j:j+1]), psi)
        end

        # --- Imaginary-time evolution using TDVP ---
        # Approximate ρEβ) = exp(-β H / 2) ρE0) via TDVP with time step 0.1.
        psi = tdvp(
            -0.5 * H_ext,
            beta_step * nsteps,
            psi;
            time_step  = beta_step,
            nsteps     = nsteps,
            cutoff     = 1e-8,
            normalize  = true,
            nsite      = 2,
            outputlevel = 1,
        )

        # --- Trace out ancillas to build mixed state ---
        # Trace out the ancilla degrees of freedom to obtain the physical
        # mixed state ρEon phys_sites.
        rho = trace_out_ancilla(psi; cutoff=1e-10)
        rho = replace_site_indices(rho, phys_sites)

        # === Evaluate observables for each requested number of copies ===
        for n_copies in n_copies_list

            # --- Reduced density matrix on selected subregion ---
            # rho_sub is the reduced density matrix on the union of n_copies
            # A–B–C blocks.
            rho_sub = partial_trace_mpo(rho, sub_indices_dict[n_copies])

            # For LVP (Localized Virtual Purification), always start from the
            # reduced density matrix of the first A–B–C block.
            rho_lvp0 = partial_trace_mpo(rho, sub_indices_dict[1])

            # --- Build VAP (Virtual Autopurification) density matrix ---
            # Initialize rho_combined with the first subregion block. VAP is
            # obtained by composing multiple spatial copies with alternating
            # orientation (even copies reversed).
            rho_combined = MPO(sub_sites)
            for k = 1:length(sub_sites)
                rho_combined[k] = rho_sub[k]
            end
            rho_combined = replace_site_indices(rho_combined, sub_sites)

            # For a = 2..n_copies, successively apply additional copies,
            # reversing the order for even copies.
            for a = 2:n_copies
                rho_next = MPO(sub_sites)

                if iseven(a)
                    # Even copies: take reversed ordering of the subregion block
                    for k = 1:length(sub_sites)
                        rho_next[k] = rho_sub[a * subregion_size - k + 1]
                    end
                    rho_next = replace_site_indices(rho_next, sub_sites)
                    rho_combined = apply(rho_combined, rho_next)

                    # Contract the rightmost link index to connect MPOs
                    link = siteinds(rho_combined; tags = "Link", plev = 0)[end]
                    rho_combined[end] = rho_combined[end] * delta(link, link')

                else
                    # Odd copies: take subregion block in the original order
                    for k = 1:length(sub_sites)
                        rho_next[k] = rho_sub[(a - 1) * subregion_size + k]
                    end
                    rho_next = replace_site_indices(rho_next, sub_sites)
                    rho_combined = apply(rho_combined, rho_next)

                    # Contract the leftmost link index to connect MPOs
                    link = siteinds(rho_combined; tags = "Link", plev = 0)[1]
                    rho_combined[1] = rho_combined[1] * delta(link, link')
                end
            end

            # --- LVP (Localized Virtual Purification) construction ---
            # Replace indices of rho_lvp by sub_sites and apply it to itself
            # (n_copies - 1) times to build the localized virtual purification
            # density matrix.
            rho_lvp0 = replace_site_indices(rho_lvp0, sub_sites)
            rho_lvp = rho_lvp0
            for a = 2:n_copies
                rho_lvp = apply(rho_lvp0, rho_lvp)
            end

            # --- Normalize density matrices and evaluate observables ---
            rho_vap = rho_combined / tr(rho_combined)

            energy_values[i, n_copies] = inner(rho_vap, H_local)
            cost_vap[i, n_copies]      = tr(rho_combined)^(-2)
            cost_lvp[i, n_copies]      = tr(rho_lvp)^(-2) * n_copies
        end
    end

    # === Output results as JSON ===
    # For each copy number, write a JSON file containing β, energy, and VAP/LVP costs.
    mkpath(output_dir)

    for n_copies in n_copies_list

        file_name = @sprintf(
            "nA%d_dAC%d_dBB%d_hx%s_m%d_n%d.json",
            N_A,
            dist_AC,
            dist_BB,
            scaled_filename_token(hx, 10, 2),
            max_n,
            n_copies,
        )

        data = Dict(
            "parameters" => Dict(
                "size_A" => N_A,
                "distance_AC" => dist_AC,
                "distance_BB" => dist_BB,
                "hx" => hx,
                "number_of_copies" => n_copies,
                "requested_copy_counts" => collect(n_copies_list),
                "max_copy_count" => max_n,
                "total_size" => total_sites,
                "selected_site_indices" => sub_indices_dict[n_copies],
                "beta_min" => beta_min,
                "beta_max" => beta_max,
                "beta_step" => beta_step,
            ),
            "beta"      => beta_values,
            "energy"    => energy_values[:, n_copies],
            "cost_vap"  => cost_vap[:, n_copies],
            "cost_lvp"  => cost_lvp[:, n_copies],
        )

        path = joinpath(output_dir, file_name)
        open(path, "w") do io
            JSON3.pretty(io, data)
        end
        println("Saved $path")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
