# VAP Simulations

Numerical simulation code for:

> Atsushi Iwaki, Hideaki Hakoshima, and Nobuyuki Yoshioka,
> *Virtual Autopurification in Translationally Invariant Quantum States*.
> [arXiv:2611.24448](https://arxiv.org/abs/2611.24448) [quant-ph] (2026).

If you use this code, please cite the paper. Machine-readable citation metadata
is provided in [CITATION.cff](CITATION.cff).

Julia generates the simulation data and Python plots the results. Generated
JSON data and figures are not included in the repository.

## Setup

- Julia 1.12 (tested with 1.12.5)
- Python 3 with NumPy and Matplotlib

Run from the repository root:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Reproduce the VAP results

Run the commands below from the repository root. Each generator defaults to
the simulation conditions described in the paper. The plotting scripts omit
the reference curves in Figure 4 and Figure 5(a), and all of Figure 5(c).

### Figure 2: Finite-temperature method comparison

```bash
julia --startup-file=no --project=. simulations/finite_temperature_method_comparison/generate_finite_temperature_comparison.jl
python3 simulations/finite_temperature_method_comparison/plot_finite_temperature_comparison.py
```

### Figure 3: TFI overhead scaling

```bash
julia --startup-file=no --project=. simulations/tfi_overhead_scaling/generate_tfi_overhead_scaling.jl
python3 simulations/tfi_overhead_scaling/plot_tfi_overhead_scaling.py
```

### Figure 4: Random-field Heisenberg

```bash
julia --startup-file=no --project=. simulations/random_field_heisenberg/generate_random_field_heisenberg.jl
python3 simulations/random_field_heisenberg/plot_random_field_heisenberg.py
```

### Figure 5: Noisy TFI

```bash
julia --startup-file=no --project=. simulations/noisy_tfi/generate_noisy_tfi.jl
python3 simulations/noisy_tfi/plot_noisy_tfi.py
```

## Outputs

Data are written to `outputs/<pipeline>/data/` and PDFs to
`outputs/<pipeline>/figures/`. Use `--output_dir` on generators or
`--input-dir` and `--output` on plotters to select other locations.
Use separate directories for calculations with different parameters.
Each script lists its options with `--help`.

## Tests

```bash
julia --startup-file=no --project=. test/runtests.jl
python3 -m unittest discover -s test -p 'test_*.py' -v
sh test/test_multicopy_generators.sh
```

Tests use small systems and temporary input fixtures. The shell script checks
all three copy-based generators with three copies. The test suite uses smaller
parameter sweeps than those in the paper.

## License

MIT License. See [LICENSE](LICENSE).
