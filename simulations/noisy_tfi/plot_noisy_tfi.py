#!/usr/bin/env python3

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "outputs" / "noisy_tfi" / "data"
DEFAULT_FIGURE_DIR = REPOSITORY_ROOT / "outputs" / "noisy_tfi" / "figures"
FILENAME = re.compile(
    r"^nA(?P<size_A>\d+)_dAC(?P<dAC>\d+)_dBB(?P<dBB>\d+)_"
    r"hx(?P<hx>\d+)_p(?P<p>\d+)_n(?P<n>\d+)\.json$"
)
ARRAY_KEYS = ("beta", "energy_vap", "cost_vap")


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
    beta_reference = None
    for path in sorted(input_dir.glob("*.json")):
        match = FILENAME.match(path.name)
        if match is None:
            continue
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        missing = ({"parameters", *ARRAY_KEYS}).difference(record)
        if missing:
            raise ValueError(f"{path.name}: missing keys: {', '.join(sorted(missing))}")
        lengths = {len(record[key]) for key in ARRAY_KEYS}
        if len(lengths) != 1:
            raise ValueError(f"{path.name}: beta and observable lengths differ")
        for key in ARRAY_KEYS:
            if not all(math.isfinite(float(value)) for value in record[key]):
                raise ValueError(f"{path.name}: {key} must contain finite values")
        parameters = record["parameters"]
        for key in ("number_of_copies", "error_rate"):
            if key not in parameters:
                raise ValueError(f"{path.name}: parameters missing {key}")
        beta = [float(value) for value in record["beta"]]
        if beta_reference is None:
            beta_reference = beta
        elif beta != beta_reference:
            raise ValueError(f"{path.name}: beta grid does not match other records")
        record["number_of_copies"] = int(parameters["number_of_copies"])
        record["error_rate"] = float(parameters["error_rate"])
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


def main(argv=None):
    args = parse_args(argv)
    records = load_records(args.input_dir)
    copies = sorted({record["number_of_copies"] for record in records})
    by_pair = {
        (record["number_of_copies"], record["error_rate"]): record
        for record in records
    }
    for n_copies in copies:
        if (n_copies, 0.0) not in by_pair:
            raise ValueError(f"Missing required p=0 baseline for n={n_copies}")

    selected_index = beta_index(records[0]["beta"], args.beta)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    all_positive = True
    for n_copies in copies:
        baseline = by_pair[(n_copies, 0.0)]
        beta = [float(value) for value in baseline["beta"]]
        axes[0].plot(
            [n_copies * value for value in beta],
            baseline["energy_vap"],
            marker="o",
            label=f"n={n_copies}",
        )

        noisy_records = sorted(
            (
                record
                for record in records
                if record["number_of_copies"] == n_copies
                and record["error_rate"] > 0
            ),
            key=lambda record: record["error_rate"],
        )
        error_rates = [record["error_rate"] for record in noisy_records]
        baseline_energy = float(baseline["energy_vap"][selected_index])
        differences = [
            abs(float(record["energy_vap"][selected_index]) - baseline_energy)
            for record in noisy_records
        ]
        if error_rates:
            axes[1].plot(
                error_rates,
                differences,
                marker="o",
                label=f"n={n_copies}",
            )
            all_positive = all_positive and all(value > 0 for value in differences)

    axes[0].set_xlabel(r"$n\beta$")
    axes[0].set_ylabel("VAP energy")
    axes[1].set_xlabel("Depolarizing error rate")
    axes[1].set_ylabel(r"$|E(p)-E(0)|$")
    if axes[1].lines and all_positive:
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize="small")

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
