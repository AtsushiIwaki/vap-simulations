module RandomFieldObservableTests

using Test
using LinearAlgebra
import ..PaperDisorderDefaults

const GENERATOR = PaperDisorderDefaults
const SZ = [0.5 0.0; 0.0 -0.5]
const SP = [0.0 1.0; 0.0 0.0]
const SM = SP'
const IDENTITY = Matrix{Float64}(I, 2, 2)

# Site 1 is the least significant bit, independently of ITensor conventions.
spin_operator(n, site, op) = reduce(kron, [i == site ? op : IDENTITY for i in n:-1:1])

function local_energy(n, left, right, hleft, hright)
    return spin_operator(n, left, SZ) * spin_operator(n, right, SZ) +
           0.5 * spin_operator(n, left, SP) * spin_operator(n, right, SM) +
           0.5 * spin_operator(n, left, SM) * spin_operator(n, right, SP) +
           0.5 * hleft * spin_operator(n, left, SZ) +
           0.5 * hright * spin_operator(n, right, SZ)
end

function reduced_density(rho, kept, n)
    omitted = setdiff(1:n, kept)
    embed(bits, sites) = sum(((bits >> (j - 1)) & 1) << (site - 1) for (j, site) in enumerate(sites))
    result = zeros(2^length(kept), 2^length(kept))
    for a in axes(result, 1), b in axes(result, 2), c in 0:(2^length(omitted) - 1)
        outside = embed(c, omitted)
        result[a, b] += rho[1 + embed(a - 1, kept) + outside, 1 + embed(b - 1, kept) + outside]
    end
    return result
end

@testset "folded random-field energy agrees with dense joint thermal contraction" begin
    # Two blocks of four spins; the second block is reversed before multiplication.
    _, fields = GENERATOR.sample_random_fields(8, 1.0, 1234, 1)
    hamiltonian = zeros(256, 256)
    for site in 1:7
        hamiltonian += local_energy(8, site, site + 1, 0.0, 0.0)
    end
    for site in 1:8
        hamiltonian += fields[site] * spin_operator(8, site, SZ)
    end
    beta = 0.1
    thermal = exp(-beta * Symmetric(hamiltonian))
    thermal /= tr(thermal)
    # Explicit bit reversal on a four-spin block, rather than generator indexing.
    reversed_basis = [1, 9, 5, 13, 3, 11, 7, 15, 2, 10, 6, 14, 4, 12, 8, 16]
    # Retain inter-block correlations and contract the first block's column
    # with the reflected second block's row, including their shared MPO bond.
    folded = zeros(16, 16)
    for a in 1:16, e in 1:16, c in 1:16
        folded[a, e] += thermal[a + 16 * (reversed_basis[c] - 1), c + 16 * (reversed_basis[e] - 1)]
    end
    folded /= tr(folded)
    first_observable = local_energy(4, 2, 3, fields[2], fields[3])
    second_observable = local_energy(4, 2, 3, fields[7], fields[6])
    expected_two = tr(folded * (first_observable + second_observable)) / 2
    single = reduced_density(thermal, [3, 4, 5, 6], 8)
    expected_one = tr(single * local_energy(4, 2, 3, fields[4], fields[5]))

    mktempdir(get(ENV, "VAP_TEST_TMPDIR", tempdir())) do output
        GENERATOR.main([
            "--distance_AC", "1", "--distance_BB", "0", "--random_fields", "1.0",
            "--random_rots", "1:1", "--number_of_copies", "1", "2",
            "--beta_max", "0.1", "--output_dir", output,
        ])
        for (copies, expected) in [(1, expected_one), (2, expected_two)]
            file = only(filter(name -> occursin("_n$(copies)_", name), readdir(output)))
            record = GENERATOR.JSON3.read(read(joinpath(output, file), String))
            @test isapprox(record.energy_vap[end], expected; atol=1e-4, rtol=0)
            @test get(record.parameters, :energy_observable_convention, nothing) == "folded_physical_sites"
        end
    end
end

end
