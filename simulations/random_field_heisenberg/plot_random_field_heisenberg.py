#!/usr/bin/env python3

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_validation import (  # noqa: E402
    finite_array,
    finite_number,
    fixed_parameter_signature,
    load_json_object,
    require_compatible_signature,
    require_parameters,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    REPOSITORY_ROOT / "outputs" / "random_field_heisenberg" / "data"
)
DEFAULT_FIGURE_DIR = (
    REPOSITORY_ROOT / "outputs" / "random_field_heisenberg" / "figures"
)
FILENAME = re.compile(
    r"^nA(?P<size_A>\d+)_dAC(?P<dAC>\d+)_dBB(?P<dBB>\d+)_"
    r"D(?P<Delta>[A-Za-z0-9]+)_W(?P<W>[A-Za-z0-9]+)(?:_m(?P<max_n>\d+))?_n(?P<n>\d+)_r(?P<r>\d+)\.json$"
)
ARRAY_KEYS = ("beta", "energy_vap", "cost_vap")
FIXED_PARAMETER_KEYS = (
    "size_A",
    "distance_AC",
    "distance_BB",
    "Delta",
    "base_seed",
    "requested_copy_counts",
    "max_copy_count",
    "total_size",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot realization-averaged random-field Heisenberg VAP data."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURE_DIR / "random_field_heisenberg.pdf",
    )
    parser.add_argument(
        "--field-strengths",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
    )
    return parser.parse_args(argv)


def load_records(input_dir: Path):
    records = []
    seen_realizations = set()
    beta_reference = None
    signature_reference = None
    for path in sorted(input_dir.glob("*.json")):
        if FILENAME.match(path.name) is None:
            continue
        record = load_json_object(path)
        missing = ({"parameters", *ARRAY_KEYS}).difference(record)
        if missing:
            raise ValueError(f"{path.name}: missing keys: {', '.join(sorted(missing))}")
        parameters = require_parameters(record, path)
        signature = fixed_parameter_signature(
            parameters, FIXED_PARAMETER_KEYS, path
        )
        signature_reference = require_compatible_signature(
            signature_reference, signature, path
        )
        for key in ARRAY_KEYS:
            record[key] = finite_array(record, key, path)
        lengths = {len(record[key]) for key in ARRAY_KEYS}
        if len(lengths) != 1:
            raise ValueError(f"{path.name}: beta and observable lengths differ")
        required_parameters = {
            "random_field",
            "number_of_copies",
            "realization",
        }
        missing_parameters = required_parameters.difference(parameters)
        if missing_parameters:
            raise ValueError(
                f"{path.name}: parameters missing "
                f"{', '.join(sorted(missing_parameters))}"
            )
        beta = record["beta"]
        if beta_reference is None:
            beta_reference = beta
        elif beta != beta_reference:
            raise ValueError(f"{path.name}: beta grid does not match other records")
        record["field_strength"] = finite_number(
            parameters["random_field"], path, "random_field"
        )
        try:
            record["number_of_copies"] = int(parameters["number_of_copies"])
            record["realization"] = int(parameters["realization"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path.name}: copy count and realization must be integers"
            ) from error
        realization_key = (
            record["field_strength"],
            record["number_of_copies"],
            record["realization"],
        )
        if realization_key in seen_realizations:
            raise ValueError(
                f"{path.name}: duplicate field/copy/realization {realization_key}"
            )
        seen_realizations.add(realization_key)
        records.append(record)
    if not records:
        raise FileNotFoundError(
            f"No random-field Heisenberg JSON files found in {input_dir.resolve()}. "
            "Run generate_random_field_heisenberg.jl first."
        )
    return records


def matches_field(value, requested):
    return math.isclose(value, requested, rel_tol=1e-9, abs_tol=1e-12)


def mean_and_sample_std(records):
    energies = np.asarray([record["energy_vap"] for record in records], dtype=float)
    mean = energies.mean(axis=0)
    if len(records) == 1:
        standard_deviation = np.zeros_like(mean)
    else:
        standard_deviation = energies.std(axis=0, ddof=1)
    return mean, standard_deviation


def main(argv=None):
    args = parse_args(argv)
    records = load_records(args.input_dir)
    figure, axis = plt.subplots(figsize=(7.0, 4.5))

    for field_strength in args.field_strengths:
        field_records = [
            record
            for record in records
            if matches_field(record["field_strength"], field_strength)
        ]
        if not field_records:
            raise ValueError(
                f"No data found for requested field strength W={field_strength:g}"
            )
        copies = sorted(
            {record["number_of_copies"] for record in field_records}
        )
        for n_copies in copies:
            group = [
                record
                for record in field_records
                if record["number_of_copies"] == n_copies
            ]
            mean, standard_deviation = mean_and_sample_std(group)
            beta = np.asarray(group[0]["beta"], dtype=float)
            axis.errorbar(
                n_copies * beta,
                mean,
                yerr=standard_deviation,
                marker="o",
                capsize=2,
                label=f"W={field_strength:g}, n={n_copies}",
            )

    axis.set_xlabel(r"$n\beta$")
    axis.set_ylabel("Mean VAP energy")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small", ncols=2)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
