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
    finite_array,
    fixed_parameter_signature,
    load_json_object,
    require_compatible_signature,
    require_parameters,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "outputs" / "tfi_overhead_scaling" / "data"
DEFAULT_FIGURE_DIR = (
    REPOSITORY_ROOT / "outputs" / "tfi_overhead_scaling" / "figures"
)
FILENAME = re.compile(
    r"^nA(?P<size_A>\d+)_dAC(?P<dAC>\d+)_dBB(?P<dBB>\d+)_"
    r"hx(?P<hx>[A-Za-z0-9]+)(?:_m(?P<max_n>\d+))?_n(?P<n>\d+)\.json$"
)
ARRAY_KEYS = ("beta", "energy", "cost_vap", "cost_lvp")
FIXED_PARAMETER_KEYS = (
    "size_A",
    "distance_AC",
    "distance_BB",
    "hx",
    "requested_copy_counts",
    "max_copy_count",
    "total_size",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot TFI VAP/LVP overhead data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURE_DIR / "tfi_overhead_scaling.pdf",
    )
    return parser.parse_args(argv)


def load_records(input_dir: Path):
    records = []
    seen_copy_counts = set()
    beta_reference = None
    signature_reference = None
    for path in sorted(input_dir.glob("*.json")):
        match = FILENAME.match(path.name)
        if match is None:
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
        beta = record["beta"]
        if beta_reference is None:
            beta_reference = beta
        elif beta != beta_reference:
            raise ValueError(f"{path.name}: beta grid does not match other records")
        record["number_of_copies"] = int(match.group("n"))
        if record["number_of_copies"] in seen_copy_counts:
            raise ValueError(
                f"{path.name}: duplicate copy count {record['number_of_copies']}"
            )
        seen_copy_counts.add(record["number_of_copies"])
        records.append(record)
    if not records:
        raise FileNotFoundError(
            f"No TFI overhead JSON files found in {input_dir.resolve()}. "
            "Run generate_tfi_overhead_scaling.jl first."
        )
    return sorted(records, key=lambda record: record["number_of_copies"])


def main(argv=None):
    args = parse_args(argv)
    records = load_records(args.input_dir)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for record in records:
        n = record["number_of_copies"]
        beta = [float(value) for value in record["beta"]]
        axes[0].plot(beta, record["energy"], marker="o", label=f"n={n}")
        axes[1].plot(
            beta,
            record["cost_vap"],
            marker="o",
            label=f"VAP, n={n}",
        )
        axes[1].plot(
            beta,
            record["cost_lvp"],
            marker="s",
            linestyle="--",
            label=f"LVP, n={n}",
        )

    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_ylabel("Energy")
    axes[1].set_xlabel(r"$\beta$")
    axes[1].set_ylabel("Sampling overhead")
    if all(
        float(value) > 0
        for record in records
        for key in ("cost_vap", "cost_lvp")
        for value in record[key]
    ):
        axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
