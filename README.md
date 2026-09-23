# VAP Simulations

## Overview

This repository contains the numerical pipelines associated with *Virtual
Autopurification in Translationally Invariant Quantum States*. The Julia
programs generate tensor-network simulation data, and the Python programs
create compact diagnostic plots of those data.

The plotting programs are intended for inspecting the calculations. They do
not reproduce the final publication figures, and they do not require exact or
reference-energy data.

## Repository layout

```text
src/                                           shared Julia MPO operations
simulations/finite_temperature_method_comparison/
simulations/tfi_overhead_scaling/
simulations/noisy_tfi/
simulations/random_field_heisenberg/
test/                                          Julia and Python tests
outputs/                                       generated data and figures
```

Every pipeline has one semantically named `generate_*.jl` program and one
`plot_*.py` program. Generator defaults are the manuscript calculation
conditions; the reduced commands below are only end-to-end checks.

## Current manuscript mapping

Figure numbering in the manuscript is provisional. The stable mapping is
therefore expressed by scientific content rather than figure number:

| Pipeline | Manuscript content |
| --- | --- |
| `finite_temperature_method_comparison` | FVP, LVP, and VAP finite-temperature comparison |
| `tfi_overhead_scaling` | VAP/LVP energy and sampling-overhead scaling for the transverse-field Ising model |
| `noisy_tfi` | Depolarizing-noise dependence of transverse-field Ising VAP |
| `random_field_heisenberg` | Realization-averaged random-field Heisenberg VAP |

## Requirements

- Julia 1.12
- Python 3 with NumPy and Matplotlib

Paper-default runs can require substantial memory and compute time. Begin with
the reduced commands when checking a new installation.

## Installation

From the repository root, instantiate the committed Julia environment:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
```

Create an optional Python virtual environment and install the plotting
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

All commands below are run from the repository root. Scripts derive their
default paths from their own locations, so they also work from another current
working directory when invoked by absolute path.

## Finite-temperature method comparison

Generate the paper-default sweep:

```bash
julia --startup-file=no --project=. simulations/finite_temperature_method_comparison/generate_finite_temperature_comparison.jl
```

Run a reduced end-to-end calculation:

```bash
julia --startup-file=no --project=. simulations/finite_temperature_method_comparison/generate_finite_temperature_comparison.jl --buffer_distances 1 2 --separation_distances 1 2 --distance_boundary 1 --inverse_temperature 0.1
```

Plot whichever compatible sweep is present in the pipeline data directory:

```bash
python3 simulations/finite_temperature_method_comparison/plot_finite_temperature_comparison.py
```

PDF: `outputs/finite_temperature_method_comparison/figures/finite_temperature_comparison.pdf`

## TFI overhead scaling

Generate the paper-default sweep:

```bash
julia --startup-file=no --project=. simulations/tfi_overhead_scaling/generate_tfi_overhead_scaling.jl
```

Run a reduced end-to-end calculation:

```bash
julia --startup-file=no --project=. simulations/tfi_overhead_scaling/generate_tfi_overhead_scaling.jl --distance_AC 1 --distance_BB 0 --number_of_copies 1
```

Plot the energy and VAP/LVP costs:

```bash
python3 simulations/tfi_overhead_scaling/plot_tfi_overhead_scaling.py
```

PDF: `outputs/tfi_overhead_scaling/figures/tfi_overhead_scaling.pdf`

## Noisy TFI

Generate the paper-default error-rate and copy-count sweep:

```bash
julia --startup-file=no --project=. simulations/noisy_tfi/generate_noisy_tfi.jl
```

Run a reduced end-to-end calculation:

```bash
julia --startup-file=no --project=. simulations/noisy_tfi/generate_noisy_tfi.jl --distance_AC 1 --distance_BB 1 --error_rates 0.0 0.001 --number_of_copies 1 --beta_max 0.1
```

Plot the zero-noise energy and noise-induced energy difference at
`beta = 0.1`:

```bash
python3 simulations/noisy_tfi/plot_noisy_tfi.py --beta 0.1
```

PDF: `outputs/noisy_tfi/figures/noisy_tfi.pdf`

## Random-field Heisenberg

Generate the paper-default field-strength, realization, and copy-count sweep:

```bash
julia --startup-file=no --project=. simulations/random_field_heisenberg/generate_random_field_heisenberg.jl
```

Run a reduced end-to-end calculation:

```bash
julia --startup-file=no --project=. simulations/random_field_heisenberg/generate_random_field_heisenberg.jl --distance_AC 1 --distance_BB 1 --random_fields 1.0 --random_rots 1:1 --number_of_copies 1 --beta_max 0.1
```

Plot the realization-averaged energy for the generated field strength:

```bash
python3 simulations/random_field_heisenberg/plot_random_field_heisenberg.py --field-strengths 1.0
```

PDF: `outputs/random_field_heisenberg/figures/random_field_heisenberg.pdf`

## Generated outputs

Generators write JSON to `outputs/<pipeline>/data/`; plotters write PDF files
to `outputs/<pipeline>/figures/`. Generated data and figures are intentionally
not versioned. Use `--output_dir`, `--input-dir`, and `--output` to override
the relevant defaults.

If output from incompatible parameter sweeps shares a directory, use a fresh
output directory or pass a matching input directory to the plotter.

## Testing

Run the Julia tests, Python plot tests, and Python syntax check from the
repository root:

```bash
julia --startup-file=no --project=. test/runtests.jl
python3 -m unittest test.test_plots -v
python3 -m compileall -q simulations test
```

The test suite uses small MPOs and fixture JSON. Full paper-default
calculations are not part of the automated tests.

## Citation

Please cite the associated paper, *Virtual Autopurification in
Translationally Invariant Quantum States*. Machine-readable author and title
metadata are provided in `CITATION.cff`.

## License

This software is distributed under the MIT License. See `LICENSE`.
