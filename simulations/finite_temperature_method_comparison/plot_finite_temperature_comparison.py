#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_validation import (  # noqa: E402
    finite_number,
    fixed_parameter_signature,
    load_json_object,
    require_compatible_signature,
    require_parameters,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    REPOSITORY_ROOT / "outputs" / "finite_temperature_method_comparison" / "data"
)
DEFAULT_FIGURE_DIR = (
    REPOSITORY_ROOT
    / "outputs"
    / "finite_temperature_method_comparison"
    / "figures"
)
FILENAME = re.compile(
    r"^nA(?P<size_A>\d+)_dbuf(?P<d_buf>\d+)_dsep(?P<d_sep>\d+)_"
    r"hx(?P<hx>[A-Za-z0-9]+)_beta(?P<beta>[A-Za-z0-9]+)\.json$"
)
VALUE_KEYS = (
    "trace_distance_fvp_lvp",
    "trace_distance_lvp_vap",
    "trace_distance_fvp_vap",
    "cost_fvp",
    "cost_lvp",
    "cost_vap",
)
FIXED_PARAMETER_KEYS = ("size_A", "distance_boundary", "hx", "beta")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot finite-temperature FVP/LVP/VAP comparison data."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURE_DIR / "finite_temperature_comparison.pdf",
    )
    return parser.parse_args(argv)


def load_records(input_dir: Path):
    records = []
    seen_sweep_points = set()
    signature_reference = None
    for path in sorted(input_dir.glob("*.json")):
        match = FILENAME.match(path.name)
        if match is None:
            continue
        record = load_json_object(path)
        missing = ({"parameters", *VALUE_KEYS}).difference(record)
        if missing:
            raise ValueError(f"{path.name}: missing keys: {', '.join(sorted(missing))}")
        parameters = require_parameters(record, path)
        parameter_missing = {"d_buf", "d_sep"}.difference(parameters)
        if parameter_missing:
            raise ValueError(
                f"{path.name}: missing parameter keys: "
                f"{', '.join(sorted(parameter_missing))}"
            )
        try:
            sweep_point = (int(parameters["d_buf"]), int(parameters["d_sep"]))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path.name}: d_buf and d_sep must be integers"
            ) from error
        if sweep_point in seen_sweep_points:
            raise ValueError(f"{path.name}: duplicate sweep point {sweep_point}")
        seen_sweep_points.add(sweep_point)
        signature = fixed_parameter_signature(
            parameters, FIXED_PARAMETER_KEYS, path
        )
        signature_reference = require_compatible_signature(
            signature_reference, signature, path
        )
        for key in VALUE_KEYS:
            record[key] = finite_number(record[key], path, key)
        records.append(record)
    if not records:
        raise FileNotFoundError(
            f"No finite-temperature comparison JSON files found in {input_dir.resolve()}. "
            "Run generate_finite_temperature_comparison.jl first."
        )
    return records


def series(records, varying, fixed, fixed_value, keys):
    selected = [
        record
        for record in records
        if int(record["parameters"][fixed]) == fixed_value
    ]
    selected.sort(key=lambda record: int(record["parameters"][varying]))
    if not selected:
        raise ValueError(f"No records found with {fixed}={fixed_value}")
    x_values = [int(record["parameters"][varying]) for record in selected]
    values = {
        key: [float(record[key]) for record in selected]
        for key in keys
    }
    return x_values, values


def use_log_scale(axis, values):
    if all(value > 0 for group in values for value in group):
        axis.set_yscale("log")


def make_figure(records):
    min_d_buf = min(int(record["parameters"]["d_buf"]) for record in records)
    min_d_sep = min(int(record["parameters"]["d_sep"]) for record in records)

    trace_keys = VALUE_KEYS[:3]
    cost_keys = VALUE_KEYS[3:]
    labels = {
        "trace_distance_fvp_lvp": "FVP–LVP",
        "trace_distance_lvp_vap": "LVP–VAP",
        "trace_distance_fvp_vap": "FVP–VAP",
        "cost_fvp": "FVP",
        "cost_lvp": "LVP",
        "cost_vap": "VAP",
    }

    figure, axes = plt.subplots(2, 2, figsize=(9, 7), sharex="col")
    configurations = (
        (0, "d_buf", "d_sep", min_d_sep, r"$d_{\mathrm{buf}}$"),
        (1, "d_sep", "d_buf", min_d_buf, r"$d_{\mathrm{sep}}$"),
    )
    for column, varying, fixed, fixed_value, xlabel in configurations:
        x_trace, trace_values = series(
            records, varying, fixed, fixed_value, trace_keys
        )
        x_cost, cost_values = series(records, varying, fixed, fixed_value, cost_keys)
        # Raw costs are measurement costs; FVP and LVP use two physical copies.
        for key in ("cost_fvp", "cost_lvp"):
            cost_values[key] = [2 * value for value in cost_values[key]]
        for key in trace_keys:
            axes[0, column].plot(
                x_trace, trace_values[key], marker="o", label=labels[key]
            )
        for key in cost_keys:
            axes[1, column].plot(
                x_cost, cost_values[key], marker="o", label=labels[key]
            )
        axes[0, column].set_title(f"{fixed} = {fixed_value}")
        axes[1, column].set_xlabel(xlabel)
        use_log_scale(axes[0, column], trace_values.values())
        use_log_scale(axes[1, column], cost_values.values())
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)

    axes[0, 0].set_ylabel("Trace distance")
    axes[1, 0].set_ylabel("Space-time overhead")
    axes[0, 0].legend()
    axes[1, 0].legend()
    figure.tight_layout()
    return figure


def main(argv=None):
    args = parse_args(argv)
    records = load_records(args.input_dir)
    figure = make_figure(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
