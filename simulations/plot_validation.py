import json
import math


def load_json_object(path):
    try:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name}: invalid JSON: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path.name}: top-level JSON value must be an object")
    return record


def require_parameters(record, path):
    parameters = record.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{path.name}: parameters must be an object")
    return parameters


def finite_number(value, path, label):
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path.name}: {label} must be numeric") from error
    if not math.isfinite(converted):
        raise ValueError(f"{path.name}: {label} must be finite")
    return converted


def finite_array(record, key, path):
    values = record.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path.name}: {key} must be a nonempty array")
    return [finite_number(value, path, key) for value in values]


def fixed_parameter_signature(parameters, keys, path):
    missing = set(keys).difference(parameters)
    if missing:
        raise ValueError(
            f"{path.name}: parameters missing {', '.join(sorted(missing))}"
        )
    return {
        key: tuple(parameters[key])
        if isinstance(parameters[key], list)
        else parameters[key]
        for key in keys
    }


def require_compatible_signature(reference, current, path):
    if reference is None or current == reference:
        return current if reference is None else reference
    differing = sorted(key for key in reference if reference[key] != current[key])
    raise ValueError(
        f"{path.name}: incompatible fixed parameters: {', '.join(differing)}"
    )
