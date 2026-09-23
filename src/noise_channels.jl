function depolarizing_superoperator(p::Real)::ITensor
    0 <= p <= 1 || throw(ArgumentError("p must satisfy 0 <= p <= 1"))

    ancillas = siteinds("Qubit", 3; addtags="Anc")
    a1, a2, a3 = ancillas
    super_index = Index(2, "SuperOp")

    proj00 = op("Proj0", a2) * op("Proj0", a3)
    proj01 = op("Proj0", a2) * op("Proj1", a3)
    proj10 = op("Proj1", a2) * op("Proj0", a3)
    proj11 = op("Proj1", a2) * op("Proj1", a3)

    controlled_x =
        op("X", a1) * proj01 + op("I", a1) * (proj00 + proj10 + proj11)
    controlled_y =
        op("Y", a1) * proj10 + op("I", a1) * (proj00 + proj01 + proj11)
    controlled_z =
        op("Z", a1) * proj11 + op("I", a1) * (proj00 + proj01 + proj10)
    prime!(controlled_x; plev=1)
    prime!(controlled_y; plev=1)
    prime!(controlled_z; plev=1)

    ancilla_state = (1 - p) * proj00 + (p / 3) * (proj01 + proj10 + proj11)
    channel = ancilla_state * controlled_x * prime(dag(controlled_x))
    channel *= delta(a1, super_index) * delta(a1', super_index')
    prime!(channel, -2; tags="Anc")
    channel *= controlled_y * prime(dag(controlled_y))
    prime!(channel, -2; tags="Anc")
    channel *= controlled_z * prime(dag(controlled_z))
    prime!(channel, -2; tags="Anc")
    channel *= delta(a1, super_index^2) * delta(a1', super_index^3)
    channel *= delta(a2, a2') * delta(a3, a3')
    return real(channel)
end

function superoperator_index(channel::ITensor, prime_level::Integer)::Index
    matches = collect(inds(channel; tags="SuperOp", plev=prime_level))
    length(matches) == 1 ||
        throw(ArgumentError("channel must contain one SuperOp index at prime level $prime_level"))
    return only(matches)
end

function apply_local_channel(rho::MPO, site::Integer, channel::ITensor)::MPO
    1 <= site <= length(rho) || throw(BoundsError(rho, site))
    sp0 = superoperator_index(channel, 0)
    sp1 = superoperator_index(channel, 1)
    sp2 = superoperator_index(channel, 2)
    sp3 = superoperator_index(channel, 3)

    result = copy(rho)
    physical = siteind(result, site)
    result[site] *= channel * delta(physical, sp0) * delta(physical', sp1)
    result[site] *= delta(sp2, physical) * delta(sp3, physical')
    return result
end

function apply_local_channel_all(rho::MPO, channel::ITensor)::MPO
    return foldl(
        (state, site) -> apply_local_channel(state, site, channel),
        eachindex(rho);
        init=copy(rho),
    )
end
