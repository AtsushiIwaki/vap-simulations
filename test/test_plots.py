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
                "parameters": {"size_A": 1, "hx": 1.0, "beta": 1.0},
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


if __name__ == "__main__":
    unittest.main()
