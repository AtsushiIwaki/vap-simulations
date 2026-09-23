using ArgParse
using ITensors
using ITensorMPS
using JSON3
using LinearAlgebra
using Printf
using VAPSimulations


const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_OUTPUT_DIR = joinpath(
    REPOSITORY_ROOT,
    "outputs",
    "finite_temperature_method_comparison",
    "data",
)


function parse_commandline(args=ARGS)
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--size_A"
            arg_type = Int
            default = 1
        "--buffer_distances"
            arg_type = Int
            nargs = '+'
            default = [1, 2, 3, 4, 5]
        "--separation_distances"
            arg_type = Int
            nargs = '+'
            default = [1, 2, 3, 4, 5]
        "--distance_boundary"
            arg_type = Int
            default = 5
        "--transverse_field"
            arg_type = Float64
            default = 1.0
        "--inverse_temperature"
            arg_type = Float64
            default = 1.0
        "--output_dir"
            default = DEFAULT_OUTPUT_DIR
    end
    return parse_args(args, settings)
end


function validate_arguments(args)
    args["size_A"] > 0 || throw(ArgumentError("size_A must be positive"))
    isempty(args["buffer_distances"]) &&
        throw(ArgumentError("buffer_distances must not be empty"))
    isempty(args["separation_distances"]) &&
        throw(ArgumentError("separation_distances must not be empty"))
    all(>(0), args["buffer_distances"]) ||
        throw(ArgumentError("buffer distances must be positive"))
    all(>(0), args["separation_distances"]) ||
        throw(ArgumentError("separation distances must be positive"))
    args["distance_boundary"] >= 0 ||
        throw(ArgumentError("distance_boundary must be nonnegative"))
    args["inverse_temperature"] >= 0 ||
        throw(ArgumentError("inverse_temperature must be nonnegative"))
    return nothing
end


function trace_distance(rho1::MPO, rho2::MPO)::Float64
    difference = rho1 - rho2
    tensor = prod(difference)
    lower = [siteind(difference, i) for i in eachindex(difference)]
    upper = prime(lower)
    matrix = reshape(Array(tensor, lower..., upper...), (dim(lower), dim(upper)))
    return 0.5 * sum(abs, eigvals(matrix))
end


function swap_partial_trace(
    rho::MPO,
    boundary_distance::Int,
    size_A::Int,
    buffer_distance::Int,
    separation_distance::Int,
)::MPO
    nqubits = length(rho)
    sites = [siteind(rho, j) for j in eachindex(rho)]
    first_A = boundary_distance + buffer_distance + 1
    result = MPO(sites[first_A:(first_A + size_A - 1)])

    left = ITensor(1.0)
    for j1 in 1:boundary_distance
        j2 = nqubits - j1 + 1
        left *= rho[j1] * delta(sites[j1], sites[j1]')
        left *= rho[j2] * delta(sites[j2], sites[j2]')
    end
    for j1 in (boundary_distance + 1):(boundary_distance + buffer_distance)
        j2 = nqubits - j1 + 1
        left *= rho[j1] * rho[j2] * delta(sites[j1], sites[j2]')
        left *= delta(sites[j2], sites[j1]')
    end

    j1 = first_A
    j2 = nqubits - j1 + 1
    result[1] = rho[j1] * rho[j2] * delta(sites[j1], sites[j2]') * left
    result[1] *= delta(sites[j2], sites[j1])

    right = ITensor(1.0)
    gap_start = boundary_distance + 2buffer_distance + size_A + 1
    gap_stop = gap_start + separation_distance - 1
    for j in gap_start:gap_stop
        right *= rho[j] * delta(sites[j], sites[j]')
    end
    reverse_start = boundary_distance + 2buffer_distance + size_A
    reverse_stop = boundary_distance + buffer_distance + size_A + 1
    for right_site in reverse_start:-1:reverse_stop
        mirror = nqubits - right_site + 1
        right *= rho[right_site] * rho[mirror]
        right *= delta(sites[right_site], sites[mirror]')
        right *= delta(sites[right_site]', sites[mirror])
    end

    if size_A == 1
        result[1] *= right
    else
        last_A = first_A + size_A - 1
        mirror = nqubits - last_A + 1
        result[end] =
            rho[last_A] * rho[mirror] * delta(sites[last_A], sites[mirror]') * right
        result[end] *= delta(sites[mirror], sites[last_A])
        for i in 2:(size_A - 1)
            site_A = first_A + i - 1
            mirror = nqubits - site_A + 1
            result[i] = rho[site_A] * rho[mirror] * delta(sites[site_A], sites[mirror]')
            result[i] *= delta(sites[mirror], sites[site_A])
        end
    end

    truncate!(result)
    return result
end


function main(args=ARGS)
    parsed = parse_commandline(args)
    validate_arguments(parsed)

    size_A = parsed["size_A"]
    boundary_distance = parsed["distance_boundary"]
    hx = parsed["transverse_field"]
    beta = parsed["inverse_temperature"]
    output_dir = parsed["output_dir"]
    mkpath(output_dir)

    for d_buf in parsed["buffer_distances"]
        for d_sep in parsed["separation_distances"]
            started = time()
            nqubits = 2 * (size_A + 2d_buf + boundary_distance) + d_sep
            sites = siteinds("Qubit", 2nqubits)

            terms = OpSum()
            for j in 1:2:(2nqubits - 3)
                terms -= "Z", j, "Z", j + 2
            end
            for j in 1:2:(2nqubits - 1)
                terms -= hx, "X", j
            end
            hamiltonian = MPO(terms, sites)

            psi = MPS(sites, "0")
            for j in 1:2:(2nqubits - 1)
                psi = apply(op("H", sites[j]), psi)
                psi = apply(op("CNOT", sites[j:(j + 1)]), psi)
            end
            psi = tdvp(
                -0.5 * hamiltonian,
                beta,
                psi;
                time_step=0.1,
                cutoff=1e-8,
                normalize=true,
                nsite=2,
                outputlevel=0,
            )
            rho = trace_out_ancilla(psi; cutoff=1e-10)

            rho_squared = apply(rho, rho)
            trace_squared = tr(rho_squared)
            cost_fvp = 1 / trace_squared^2
            rho_squared /= trace_squared
            A_start = boundary_distance + d_buf + 1
            rho_fvp = partial_trace_mpo(
                rho_squared,
                collect(A_start:(A_start + size_A - 1));
                cutoff=1e-15,
            )

            local_start = boundary_distance + 1
            local_stop = boundary_distance + 2d_buf + size_A
            rho_local = partial_trace_mpo(
                rho,
                collect(local_start:local_stop);
                cutoff=1e-15,
            )
            rho_local_squared = apply(rho_local, rho_local)
            trace_local = tr(rho_local_squared)
            cost_lvp = 1 / trace_local^2
            rho_local_squared /= trace_local
            rho_lvp = partial_trace_mpo(
                rho_local_squared,
                collect((d_buf + 1):(d_buf + size_A));
                cutoff=1e-15,
            )

            rho_vap = swap_partial_trace(
                rho,
                boundary_distance,
                size_A,
                d_buf,
                d_sep,
            )
            trace_vap = tr(rho_vap)
            cost_vap = 1 / trace_vap^2
            rho_vap /= trace_vap

            filename = @sprintf(
                "nA%d_dbuf%02d_dsep%02d_hx%s_beta%s.json",
                size_A,
                d_buf,
                d_sep,
                scaled_filename_token(hx, 10, 2),
                scaled_filename_token(beta, 10, 2),
            )
            data = Dict(
                "parameters" => Dict(
                    "size_A" => size_A,
                    "d_buf" => d_buf,
                    "d_sep" => d_sep,
                    "distance_boundary" => boundary_distance,
                    "hx" => hx,
                    "beta" => beta,
                ),
                "trace_distance_fvp_lvp" => trace_distance(rho_fvp, rho_lvp),
                "trace_distance_lvp_vap" => trace_distance(rho_lvp, rho_vap),
                "trace_distance_fvp_vap" => trace_distance(rho_fvp, rho_vap),
                "cost_fvp" => cost_fvp,
                "cost_lvp" => cost_lvp,
                "cost_vap" => cost_vap,
            )
            path = joinpath(output_dir, filename)
            open(path, "w") do io
                JSON3.pretty(io, data)
            end
            println("Saved $path in $(round(time() - started; digits=2)) s")
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
