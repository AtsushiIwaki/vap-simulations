#!/bin/sh
set -eu

JULIA_BIN=${JULIA_BIN:-julia}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_ROOT=$(mktemp -d /private/tmp/vap-multicopy.XXXXXX)

"$JULIA_BIN" --startup-file=no --project="$REPO_ROOT" "$REPO_ROOT/simulations/noisy_tfi/generate_noisy_tfi.jl" --distance_AC 1 --distance_BB 1 --error_rates 0.0 --number_of_copies 1 2 3 --beta_max 0.1 --output_dir "$OUTPUT_ROOT/noisy"
"$JULIA_BIN" --startup-file=no --project="$REPO_ROOT" "$REPO_ROOT/simulations/random_field_heisenberg/generate_random_field_heisenberg.jl" --distance_AC 1 --distance_BB 1 --random_fields 1.0 --random_rots 1:1 --number_of_copies 1 2 3 --beta_max 0.1 --output_dir "$OUTPUT_ROOT/heisenberg"
"$JULIA_BIN" --startup-file=no --project="$REPO_ROOT" "$REPO_ROOT/simulations/tfi_overhead_scaling/generate_tfi_overhead_scaling.jl" --distance_AC 1 --distance_BB 0 --number_of_copies 1 2 3 --beta_max 0.1 --output_dir "$OUTPUT_ROOT/tfi"

test "$(find "$OUTPUT_ROOT/noisy" -name '*.json' | wc -l | tr -d ' ')" -eq 3
test "$(find "$OUTPUT_ROOT/heisenberg" -name '*.json' | wc -l | tr -d ' ')" -eq 3
test "$(find "$OUTPUT_ROOT/tfi" -name '*.json' | wc -l | tr -d ' ')" -eq 3

python3 - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in root.glob("*/*.json"):
    record = json.loads(path.read_text(encoding="utf-8"))
    parameters = record["parameters"]
    assert parameters["requested_copy_counts"] == [1, 2, 3], path
    assert parameters["max_copy_count"] == 3, path
    expected_size = 12 if path.parent.name == "tfi" else 16
    assert parameters["total_size"] == expected_size, path
    assert len(parameters["selected_site_indices"]) == 4 * parameters["number_of_copies"], path
PY
