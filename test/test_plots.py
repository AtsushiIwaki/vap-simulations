import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_plot(relative_script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / relative_script), *args],
        cwd=tempfile.gettempdir(),
        check=False,
        capture_output=True,
        text=True,
    )


class FiniteTemperaturePlotTest(unittest.TestCase):
    def test_plot_accepts_semantic_sweep_and_writes_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            payload = {
                "parameters": {
                    "size_A": 1,
                    "distance_boundary": 1,
                    "hx": 1.0,
                    "beta": 1.0,
                },
                "trace_distance_fvp_lvp": 0.2,
                "trace_distance_lvp_vap": 0.1,
                "trace_distance_fvp_vap": 0.3,
                "cost_fvp": 2.0,
                "cost_lvp": 3.0,
                "cost_vap": 4.0,
            }
            for d_buf, d_sep in ((1, 1), (2, 1), (1, 2)):
                item = dict(payload)
                item["parameters"] = {
                    **payload["parameters"],
                    "d_buf": d_buf,
                    "d_sep": d_sep,
                }
                name = f"nA1_dbuf{d_buf:02d}_dsep{d_sep:02d}_hx10_beta10.json"
                (data_dir / name).write_text(json.dumps(item), encoding="utf-8")
            output = Path(tmp) / "comparison.pdf"
            result = run_plot(
                "simulations/finite_temperature_method_comparison/plot_finite_temperature_comparison.py",
                "--input-dir",
                str(data_dir),
                "--output",
                str(output),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater(output.stat().st_size, 0)

    def test_matching_malformed_json_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "nA1_dbuf01_dsep01_hx10_beta10.json"
            bad.write_text('{"parameters": {}}', encoding="utf-8")
            result = run_plot(
                "simulations/finite_temperature_method_comparison/plot_finite_temperature_comparison.py",
                "--input-dir",
                tmp,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)

    def test_invalid_json_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "nA1_dbuf01_dsep01_hx10_beta10.json"
            bad.write_text('{"parameters":', encoding="utf-8")
            result = run_plot(
                "simulations/finite_temperature_method_comparison/plot_finite_temperature_comparison.py",
                "--input-dir",
                tmp,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)

    def test_incompatible_fixed_parameters_name_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            base = {
                "parameters": {
                    "size_A": 1,
                    "d_buf": 1,
                    "d_sep": 1,
                    "distance_boundary": 1,
                    "hx": 1.0,
                    "beta": 1.0,
                },
                "trace_distance_fvp_lvp": 0.2,
                "trace_distance_lvp_vap": 0.1,
                "trace_distance_fvp_vap": 0.3,
                "cost_fvp": 2.0,
                "cost_lvp": 3.0,
                "cost_vap": 4.0,
            }
            (data_dir / "nA1_dbuf01_dsep01_hx10_beta10.json").write_text(
                json.dumps(base), encoding="utf-8"
            )
            incompatible = json.loads(json.dumps(base))
            incompatible["parameters"].update({"d_buf": 2, "hx": 2.0})
            bad = data_dir / "nA1_dbuf02_dsep01_hx20_beta10.json"
            bad.write_text(json.dumps(incompatible), encoding="utf-8")
            result = run_plot(
                "simulations/finite_temperature_method_comparison/plot_finite_temperature_comparison.py",
                "--input-dir",
                tmp,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)


class TfiOverheadPlotTest(unittest.TestCase):
    script = "simulations/tfi_overhead_scaling/plot_tfi_overhead_scaling.py"

    @staticmethod
    def payload(n, beta=None):
        return {
            "parameters": {
                "size_A": 2,
                "distance_AC": 4,
                "distance_BB": 4,
                "hx": 1.0,
                "number_of_copies": n,
                "requested_copy_counts": [1, 2],
                "max_copy_count": 2,
                "total_size": 32,
            },
            "beta": beta or [0.0, 0.1],
            "energy": [-1.0, -1.1],
            "cost_vap": [1.0, 1.2],
            "cost_lvp": [1.0, 1.4],
        }

    def write_record(self, directory, n, payload=None):
        path = Path(directory) / f"nA2_dAC4_dBB4_hx10_n{n}.json"
        path.write_text(json.dumps(payload or self.payload(n)), encoding="utf-8")
        return path

    def test_plot_writes_nonempty_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            self.write_record(data_dir, 1)
            self.write_record(data_dir, 2)
            output = Path(tmp) / "tfi.pdf"
            result = run_plot(
                self.script,
                "--input-dir",
                str(data_dir),
                "--output",
                str(output),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater(output.stat().st_size, 0)

    def test_malformed_record_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = self.write_record(tmp, 1, {"parameters": {}})
            result = run_plot(self.script, "--input-dir", tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)

    def test_beta_grid_mismatch_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_record(tmp, 1)
            bad_payload = self.payload(2, beta=[0.0, 0.2])
            bad = self.write_record(tmp, 2, bad_payload)
            result = run_plot(self.script, "--input-dir", tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)

    def test_empty_arrays_name_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.payload(1)
            for key in ("beta", "energy", "cost_vap", "cost_lvp"):
                payload[key] = []
            bad = self.write_record(tmp, 1, payload)
            result = run_plot(self.script, "--input-dir", tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)

    def test_invalid_json_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "nA2_dAC4_dBB4_hx10_n1.json"
            bad.write_text('{"parameters":', encoding="utf-8")
            result = run_plot(self.script, "--input-dir", tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)

    def test_incompatible_fixed_parameters_name_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_record(tmp, 1)
            incompatible = self.payload(2)
            incompatible["parameters"]["distance_AC"] = 5
            bad = Path(tmp) / "nA2_dAC5_dBB4_hx10_n2.json"
            bad.write_text(json.dumps(incompatible), encoding="utf-8")
            result = run_plot(self.script, "--input-dir", tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)


class NoisyTfiPlotTest(unittest.TestCase):
    script = "simulations/noisy_tfi/plot_noisy_tfi.py"

    @staticmethod
    def payload(n, error_rate):
        return {
            "parameters": {
                "size_A": 2,
                "distance_AC": 4,
                "distance_BB": 4,
                "hx": 1.0,
                "number_of_copies": n,
                "error_rate": error_rate,
                "requested_copy_counts": [1, 2],
                "max_copy_count": 2,
                "total_size": 32,
            },
            "beta": [0.0, 0.1],
            "energy_vap": [-1.0, -1.0 - n * error_rate],
            "cost_vap": [1.0, 1.1],
        }

    def write_record(self, directory, n, p_raw):
        path = Path(directory) / f"nA2_dAC4_dBB4_hx10_p{p_raw:03d}_n{n}.json"
        path.write_text(
            json.dumps(self.payload(n, p_raw / 1000.0)), encoding="utf-8"
        )
        return path

    def test_plot_needs_no_reference_data_and_writes_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in (1, 2):
                self.write_record(tmp, n, 0)
                self.write_record(tmp, n, 1)
            output = Path(tmp) / "noisy.pdf"
            result = run_plot(
                self.script,
                "--input-dir",
                tmp,
                "--output",
                str(output),
                "--beta",
                "0.1",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater(output.stat().st_size, 0)

    def test_missing_zero_noise_baseline_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_record(tmp, 1, 1)
            result = run_plot(
                self.script, "--input-dir", tmp, "--beta", "0.1"
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("p=0", result.stderr)

    def test_zero_noise_difference_uses_linear_scale_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_record(tmp, 1, 0)
            noisy = self.payload(1, 0.001)
            noisy["energy_vap"] = [-1.0, -1.0]
            path = Path(tmp) / "nA2_dAC4_dBB4_hx10_p001_n1.json"
            path.write_text(json.dumps(noisy), encoding="utf-8")
            output = Path(tmp) / "zero-difference.pdf"
            result = run_plot(
                self.script,
                "--input-dir",
                tmp,
                "--output",
                str(output),
                "--beta",
                "0.1",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater(output.stat().st_size, 0)

    def test_matching_malformed_record_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "nA2_dAC4_dBB4_hx10_p000_n1.json"
            bad.write_text('{"parameters": {}}', encoding="utf-8")
            result = run_plot(self.script, "--input-dir", tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)

    def test_invalid_json_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "nA2_dAC4_dBB4_hx10_p000_n1.json"
            bad.write_text('{"parameters":', encoding="utf-8")
            result = run_plot(self.script, "--input-dir", tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)

    def test_incompatible_fixed_parameters_name_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_record(tmp, 1, 0)
            incompatible = self.payload(1, 0.001)
            incompatible["parameters"]["distance_AC"] = 5
            bad = Path(tmp) / "nA2_dAC5_dBB4_hx10_p001_n1.json"
            bad.write_text(json.dumps(incompatible), encoding="utf-8")
            result = run_plot(self.script, "--input-dir", tmp, "--beta", "0.1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)


class RandomFieldHeisenbergPlotTest(unittest.TestCase):
    script = (
        "simulations/random_field_heisenberg/"
        "plot_random_field_heisenberg.py"
    )

    @staticmethod
    def payload(field_strength, realization=1):
        return {
            "parameters": {
                "size_A": 2,
                "distance_AC": 3,
                "distance_BB": 3,
                "Delta": 1.0,
                "random_field": field_strength,
                "number_of_copies": 1,
                "realization": realization,
                "sampled_fields": [0.1, -0.2],
                "base_seed": 1234,
                "requested_copy_counts": [1],
                "max_copy_count": 1,
                "total_size": 14,
            },
            "beta": [0.0, 0.1],
            "energy_vap": [0.0, -0.1 * field_strength],
            "cost_vap": [1.0, 1.1],
        }

    def write_record(self, directory, field_strength, realization=1, payload=None):
        field_code = round(10 * field_strength)
        path = Path(directory) / (
            f"nA2_dAC3_dBB3_D10_W{field_code:02d}_n1_r{realization:02d}.json"
        )
        path.write_text(
            json.dumps(payload or self.payload(field_strength, realization)),
            encoding="utf-8",
        )
        return path

    def test_single_realization_writes_pdf_with_zero_error_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_record(tmp, 0.5)
            self.write_record(tmp, 1.0)
            output = Path(tmp) / "random-field.pdf"
            result = run_plot(
                self.script,
                "--input-dir",
                tmp,
                "--output",
                str(output),
                "--field-strengths",
                "0.5",
                "1.0",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater(output.stat().st_size, 0)

    def test_missing_requested_field_strength_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_record(tmp, 0.5)
            result = run_plot(
                self.script,
                "--input-dir",
                tmp,
                "--field-strengths",
                "0.5",
                "1.0",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("W=1", result.stderr)

    def test_matching_malformed_record_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = self.write_record(tmp, 0.5, payload={"parameters": {}})
            result = run_plot(
                self.script,
                "--input-dir",
                tmp,
                "--field-strengths",
                "0.5",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)

    def test_invalid_json_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "nA2_dAC3_dBB3_D10_W05_n1_r01.json"
            bad.write_text('{"parameters":', encoding="utf-8")
            result = run_plot(
                self.script, "--input-dir", tmp, "--field-strengths", "0.5"
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)

    def test_incompatible_fixed_parameters_name_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_record(tmp, 0.5)
            incompatible = self.payload(1.0)
            incompatible["parameters"]["distance_AC"] = 4
            bad = Path(tmp) / "nA2_dAC4_dBB3_D10_W10_n1_r01.json"
            bad.write_text(json.dumps(incompatible), encoding="utf-8")
            result = run_plot(
                self.script,
                "--input-dir",
                tmp,
                "--field-strengths",
                "0.5",
                "1.0",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(bad.name, result.stderr)

if __name__ == "__main__":
    unittest.main()
