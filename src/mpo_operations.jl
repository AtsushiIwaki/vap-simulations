function trace_out_ancilla(psi::MPS; cutoff::Real=1e-10)::MPO
    iseven(length(psi)) ||
        throw(ArgumentError("psi must contain physical/ancilla pairs"))

    nphysical = div(length(psi), 2)
    projected = projector(psi)
    rho = MPO(nphysical)

    for i in 1:nphysical
        physical = 2i - 1
        ancilla = siteind(projected, physical + 1)
        rho[i] =
            projected[physical] * projected[physical + 1] * delta(ancilla, ancilla')
    end

    truncate!(rho; cutoff=cutoff)
    return rho
end

function replace_site_indices(rho::MPO, sites::AbstractVector{<:Index})::MPO
    length(rho) == length(sites) ||
        throw(DimensionMismatch("rho and sites must have equal length"))

    replaced = MPO(length(rho))
    for i in eachindex(rho)
        old = siteind(rho, i)
        new = sites[i]
        replaced[i] = rho[i] * delta(old, new) * delta(old', new')
    end
    return replaced
end

function partial_trace_mpo(
    rho::MPO,
    keep_indices::AbstractVector{<:Integer};
    cutoff::Union{Nothing,Real}=nothing,
)::MPO
    keep = sort(unique(Int.(keep_indices)))
    isempty(keep) && throw(ArgumentError("keep_indices must not be empty"))
    all(i -> 1 <= i <= length(rho), keep) ||
        throw(ArgumentError("keep index outside MPO"))

    sites = [siteind(rho, i) for i in eachindex(rho)]
    reduced = MPO(sites[keep])
    pending = ITensor(1.0)
    position = Dict(site => i for (i, site) in enumerate(keep))

    for i in eachindex(rho)
        if haskey(position, i)
            reduced[position[i]] = pending * rho[i]
            pending = ITensor(1.0)
        else
            pending *= rho[i] * delta(sites[i], sites[i]')
        end
    end

    reduced[end] *= pending
    cutoff === nothing || truncate!(reduced; cutoff=cutoff)
    return reduced
end
