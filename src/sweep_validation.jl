function validate_beta_grid(beta_min::Real, beta_max::Real, beta_step::Real)
    beta_step > 0 || throw(ArgumentError("beta_step must be positive"))
    beta_max >= beta_min >= 0 || throw(ArgumentError("invalid beta interval"))
    for (name, value) in (("beta_min", beta_min), ("beta_max", beta_max))
        quotient = value / beta_step
        isapprox(quotient, round(quotient); rtol=1e-10, atol=1e-10) ||
            throw(
                ArgumentError(
                    "$name must be an integer multiple of beta_step"
                ),
            )
    end
    return nothing
end

function scaled_filename_token(value::Real, scale::Real, width::Integer)
    legacy_code = round(Int, scale * value)
    if isapprox(value, legacy_code / scale; rtol=0, atol=1e-14)
        return lpad(string(legacy_code), width, '0')
    end
    exact = string(Float64(value))
    return "x" * replace(exact, "-" => "m", "." => "d", "+" => "p")
end
