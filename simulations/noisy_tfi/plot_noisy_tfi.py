#!/usr/bin/env python3

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "outputs" / "noisy_tfi" / "data"
DEFAULT_FIGURE_DIR = REPOSITORY_ROOT / "outputs" / "noisy_tfi" / "figures"
FILENAME = re.compile(
    r"^nA(?P<size_A>\d+)_dAC(?P<dAC>\d+)_dBB(?P<dBB>\d+)_"
    r"hx(?P<hx>[A-Za-z0-9]+)_p(?P<p>[A-Za-z0-9]+)(?:_m(?P<max_n>\d+))?_n(?P<n>\d+)\.json$"
)
ARRAY_KEYS = ("beta", "energy_vap", "cost_vap")
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
    parser = argparse.ArgumentParser(description="Plot noisy TFI VAP data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURE_DIR / "noisy_tfi.pdf",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Inverse temperature used for the noise-dependence panel.",
    )
    return parser.parse_args(argv)


def load_records(input_dir: Path):
    records = []
    seen_sweep_points = set()
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
        for key in ("number_of_copies", "error_rate"):
            if key not in parameters:
                raise ValueError(f"{path.name}: parameters missing {key}")
        beta = record["beta"]
        if beta_reference is None:
            beta_reference = beta
        elif beta != beta_reference:
            raise ValueError(f"{path.name}: beta grid does not match other records")
        try:
            record["number_of_copies"] = int(parameters["number_of_copies"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path.name}: number_of_copies must be an integer"
            ) from error
        record["error_rate"] = finite_number(
            parameters["error_rate"], path, "error_rate"
        )
        sweep_point = (record["number_of_copies"], record["error_rate"])
        if sweep_point in seen_sweep_points:
            raise ValueError(f"{path.name}: duplicate sweep point {sweep_point}")
        seen_sweep_points.add(sweep_point)
        records.append(record)
    if not records:
        raise FileNotFoundError(
            f"No noisy TFI JSON files found in {input_dir.resolve()}. "
            "Run generate_noisy_tfi.jl first."
        )
    return records


def beta_index(beta_values, requested_beta):
    for index, value in enumerate(beta_values):
        if math.isclose(float(value), requested_beta, rel_tol=1e-9, abs_tol=1e-12):
            return index
    raise ValueError(f"Requested beta={requested_beta:g} is absent from the data")


def make_figure(records, *, beta=0.5):
    copies = sorted({record["number_of_copies"] for record in records})
    by_pair = {
        (record["number_of_copies"], record["error_rate"]): record
        for record in records
    }
    for n_copies in copies:
        if (n_copies, 0.0) not in by_pair:
            raise ValueError(f"Missing required p=0 baseline for n={n_copies}")

    # Panel (a) uses p=0.01, matching the manuscript caption.
    copies = [n for n in copies if n <= 4]
    if not copies:
        raise ValueError("The paper layout needs data for at least one of n=1..4")
    for n in copies:
        if (n, 0.01) not in by_pair:
            raise ValueError(f"Missing p=0.01 energy data for n={n}")
    index = beta_index(records[0]["beta"], beta)
    rates = sorted({record["error_rate"] for record in records if record["error_rate"] > 0})
    if not rates:
        raise ValueError("The paper layout requires positive error-rate data")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for n in copies:
        noisy = by_pair[(n, 0.01)]
        axes[0].plot([n * b for b in noisy["beta"]], noisy["energy_vap"],
                     marker="o", linestyle="None", label=f"n={n}")
    for n in copies:
        baseline_energy = by_pair[(n, 0.0)]["energy_vap"][index]
        points = [(p, abs(by_pair[(n, p)]["energy_vap"][index] - baseline_energy))
                  for p in rates if (n, p) in by_pair]
        points = [(p, error) for p, error in points if error > 0]
        if points:
            axes[1].plot([p for p, _ in points], [error for _, error in points],
                         marker="o", label=f"n={n}")
    axes[0].set(xlabel=r"$n\beta$", ylabel="VAP energy", xlim=(-0.1, 2.1),
                title="p = 0.01")
    axes[1].set(xlabel="p", ylabel=r"$|E(p)-E(0)|$", xscale="log", yscale="log")
    for axis in axes:
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


def main(argv=None):
    args = parse_args(argv)
    records = load_records(args.input_dir)
    figure = make_figure(records, beta=args.beta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
