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
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
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
        signature["energy_observable_convention"] = parameters.get(
            "energy_observable_convention", "unreflected_fields"
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


def mean_and_bootstrap_error(records, samples, rng):
    energies = np.asarray([record["energy_vap"] for record in records], dtype=float)
    mean = energies.mean(axis=0)
    if len(records) < 2 or samples <= 1:
        return mean, np.zeros_like(mean)
    # Match the historical per-group seed and scalar bootstrap loop exactly.
    group_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
    bootstrap_means = np.empty((samples, energies.shape[1]))
    for sample in range(samples):
        indices = group_rng.integers(0, len(records), size=len(records))
        bootstrap_means[sample] = energies[indices].mean(axis=0)
    return mean, bootstrap_means.std(axis=0, ddof=1)


def make_figure(records, *, field_strengths=None, bootstrap_samples=2000,
                bootstrap_seed=0):
    field_strengths = [0.5, 1.0] if field_strengths is None else field_strengths
    figure, axes = plt.subplots(1, len(field_strengths), figsize=(10, 4.5),
                                squeeze=False, layout="constrained")
    axes = axes[0]
    rng = np.random.default_rng(bootstrap_seed)
    all_copies = sorted({record["number_of_copies"] for record in records})
    color_map = matplotlib.colormaps["Blues"]
    normalization = matplotlib.colors.Normalize(min(all_copies), max(all_copies))
    for axis, field in zip(axes, field_strengths):
        field_records = [record for record in records
                         if matches_field(record["field_strength"], field)]
        if not field_records:
            raise ValueError(f"No data found for requested field strength W={field:g}")
        for copies in sorted({record["number_of_copies"] for record in field_records}):
            group = sorted([record for record in field_records
                            if record["number_of_copies"] == copies],
                           key=lambda record: record["realization"])
            mean, error = mean_and_bootstrap_error(group, bootstrap_samples, rng)
            axis.errorbar(copies * np.asarray(group[0]["beta"]), mean, yerr=error,
                          marker="o", linestyle="None", capsize=0,
                          color=color_map(normalization(copies)))
        axis.set(xlabel=r"$n\beta$", title=f"W = {field:g}")
    # Keep field strengths on the same energy scale, as in the paper plot.
    lower = min(axis.get_ylim()[0] for axis in axes)
    upper = max(axis.get_ylim()[1] for axis in axes)
    for axis in axes:
        axis.set_ylim(lower, upper)
    axes[0].set_ylabel("VAP energy")
    figure.colorbar(matplotlib.cm.ScalarMappable(norm=normalization, cmap=color_map),
                    ax=list(axes), label="n")
    return figure


def main(argv=None):
    args = parse_args(argv)
    records = load_records(args.input_dir)
    figure = make_figure(records,
                         field_strengths=args.field_strengths,
                         bootstrap_samples=args.bootstrap_samples,
                         bootstrap_seed=args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
