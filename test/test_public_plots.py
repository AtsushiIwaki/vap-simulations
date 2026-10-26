import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "finite_temperature_method_comparison": "plot_finite_temperature_comparison.py",
    "tfi_overhead_scaling": "plot_tfi_overhead_scaling.py",
    "noisy_tfi": "plot_noisy_tfi.py",
    "random_field_heisenberg": "plot_random_field_heisenberg.py",
}


def module_for(pipeline):
    spec = importlib.util.spec_from_file_location(
        f"public_{pipeline}", ROOT / "simulations" / pipeline / SCRIPTS[pipeline])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixtures(directory, pipeline):
    if pipeline == "finite_temperature_method_comparison":
        for buffer, separation in [(1, 1), (2, 1), (1, 2)]:
            write(directory / f"nA1_dbuf{buffer:02d}_dsep{separation:02d}_hx10_beta10.json", {
                "parameters": {"size_A": 1, "distance_boundary": 1, "hx": 1.0,
                               "beta": 1.0, "d_buf": buffer, "d_sep": separation},
                "trace_distance_fvp_lvp": 0.2, "trace_distance_lvp_vap": 0.1,
                "trace_distance_fvp_vap": 0.3, "cost_fvp": 2.0,
                "cost_lvp": 3.0, "cost_vap": 4.0})
    elif pipeline == "tfi_overhead_scaling":
        for n in (1, 2):
            write(directory / f"nA2_dAC4_dBB4_hx10_m2_n{n}.json", {
                "parameters": {"size_A": 2, "distance_AC": 4, "distance_BB": 4,
                               "hx": 1.0, "number_of_copies": n, "time_step": 0.1,
                               "requested_copy_counts": [1, 2], "max_copy_count": 2, "total_size": 32},
                "beta": [0.0, 0.6, 0.8, 1.0], "energy": [-1.0] * 4,
                "cost_vap": [2.0] * 4, "cost_lvp": [6.0] * 4})
    elif pipeline == "noisy_tfi":
        for n in (1, 2):
            for rate in (0, 10, 20):
                p = rate / 1000
                write(directory / f"nA2_dAC4_dBB4_hx10_p{rate:03d}_m2_n{n}.json", {
                    "parameters": {"size_A": 2, "distance_AC": 4, "distance_BB": 4,
                                   "hx": 1.0, "number_of_copies": n, "error_rate": p,
                                   "requested_copy_counts": [1, 2], "max_copy_count": 2, "total_size": 32},
                    "beta": [0.0, 0.5], "energy_vap": [0.0, -0.1 * n + p * n], "cost_vap": [1.0, 2.0]})
    else:
        for w in (0.5, 1.0):
            for n in (1, 2):
                for r in (1, 2):
                    write(directory / f"nA2_dAC3_dBB3_D10_W{round(w*10):02d}_m2_n{n}_r{r:02d}.json", {
                        "parameters": {"size_A": 2, "distance_AC": 3, "distance_BB": 3,
                                       "Delta": 1.0, "random_field": w, "realization": r,
                                       "number_of_copies": n, "base_seed": 1234,
                                       "requested_copy_counts": [1, 2], "max_copy_count": 2, "total_size": 25,
                                       "energy_observable_convention": "folded_physical_sites"},
                        "beta": [0.0, 0.5], "energy_vap": [0.0, -0.1 * w * n - r * 0.01], "cost_vap": [1.0, 2.0]})


class PublicPlotTest(unittest.TestCase):
    def test_help_exposes_one_manuscript_plotting_mode(self):
        for pipeline, script in SCRIPTS.items():
            with self.subTest(pipeline=pipeline):
                result = subprocess.run([sys.executable, str(ROOT / "simulations" / pipeline / script), "--help"],
                                        cwd=tempfile.gettempdir(), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("--paper-layout", result.stdout)
                self.assertNotIn("--paper-data", result.stdout)
                self.assertNotIn("--reference-dir", result.stdout)

    def test_default_commands_accept_generated_inputs(self):
        for pipeline, script in SCRIPTS.items():
            with self.subTest(pipeline=pipeline), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                data = root / "data"
                data.mkdir()
                fixtures(data, pipeline)
                args = [sys.executable, str(ROOT / "simulations" / pipeline / script),
                        "--input-dir", str(data), "--output", str(root / "figure.pdf")]
                result = subprocess.run(args, cwd=tempfile.gettempdir(), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertGreater((root / "figure.pdf").stat().st_size, 0)

    def test_paper_costs_keep_physical_copy_factors(self):
        for pipeline in ("finite_temperature_method_comparison", "tfi_overhead_scaling"):
            with self.subTest(pipeline=pipeline), tempfile.TemporaryDirectory() as temp:
                data = Path(temp); fixtures(data, pipeline)
                plotter = module_for(pipeline)
                figure = plotter.make_figure(plotter.load_records(data))
                try:
                    if pipeline == "finite_temperature_method_comparison":
                        for line, expected in zip(figure.axes[2].lines, [4.0, 6.0, 4.0]):
                            np.testing.assert_allclose(line.get_ydata(), expected)
                    else:
                        for line in figure.axes[1].lines:
                            np.testing.assert_allclose(line.get_ydata(), [3.0, 3.0])
                finally:
                    plt.close(figure)

    def test_tfi_inputs_reject_mixed_recorded_integration_steps(self):
        for step in (0.2, None):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                fixtures(directory, "tfi_overhead_scaling")
                path = directory / "nA2_dAC4_dBB4_hx10_m2_n2.json"
                record = json.loads(path.read_text())
                if step is None:
                    del record["parameters"]["time_step"]
                else:
                    record["parameters"]["time_step"] = step
                write(path, record)
                with self.assertRaisesRegex(ValueError, "time_step"):
                    module_for("tfi_overhead_scaling").load_records(directory)

    def test_noisy_paper_energy_uses_p001_and_keeps_zero_noise_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp); fixtures(data, "noisy_tfi")
            plotter = module_for("noisy_tfi"); records = plotter.load_records(data)
            figure = plotter.make_figure(records)
            try:
                self.assertEqual(len(figure.axes), 2)
                self.assertEqual(len(figure.axes[0].lines), 2)
                self.assertEqual(len(figure.axes[1].lines), 2)
                np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [0, -0.09])
                np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), [0.01, 0.02])
            finally:
                plt.close(figure)
            missing = [r for r in records if not (r["number_of_copies"] == 2 and r["error_rate"] == 0.01)]
            with self.assertRaisesRegex(ValueError, "Missing p=0.01"):
                plotter.make_figure(missing)

    def test_plots_accept_model_parameters_without_reference_restrictions(self):
        for pipeline, parameter, value in [("noisy_tfi", "hx", 1.04),
                                            ("random_field_heisenberg", "Delta", 1.04),
                                            ("random_field_heisenberg", "random_field", 0.54)]:
            with self.subTest(pipeline=pipeline, parameter=parameter), tempfile.TemporaryDirectory() as temp:
                data = Path(temp); fixtures(data, pipeline)
                plotter = module_for(pipeline); records = plotter.load_records(data)
                if parameter == "random_field":
                    records = [r for r in records if r["field_strength"] == 0.5]
                for record in records:
                    record["parameters"][parameter] = value
                    if parameter == "random_field": record["field_strength"] = value
                kwargs = {"field_strengths": [value]} if parameter == "random_field" else {}
                figure = plotter.make_figure(records, **kwargs)
                plt.close(figure)

    def test_random_field_paper_errors_are_bootstrap_errors_of_mean(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp); fixtures(data, "random_field_heisenberg")
            plotter = module_for("random_field_heisenberg")
            figure = plotter.make_figure(plotter.load_records(data))
            try:
                self.assertEqual(len(figure.axes), 3)
                for axis, w in zip(figure.axes[:2], [0.5, 1.0]):
                    self.assertEqual(len(axis.lines), 2)
                    for n, container in enumerate(axis.containers, 1):
                        line, _, bars = container.lines
                        np.testing.assert_allclose(line.get_xdata(), [0, n * 0.5])
                        np.testing.assert_allclose(line.get_ydata(), [0, -0.1 * w * n - 0.015])
                        error = np.diff(np.asarray(bars[0].get_segments())[:, :, 1], axis=1).ravel() / 2
                        self.assertAlmostEqual(error[1], 0.01 / np.sqrt(8), delta=0.0002)
            finally:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
