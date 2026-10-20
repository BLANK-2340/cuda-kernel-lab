"""Protocol and aggregation tests; the CLI smoke test executes real subprocesses."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_suite", ROOT / "benchmarks/cpu_suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def records():
    result = []
    for pair in range(3):
        for mode in suite.MODES:
            timing = {"minimum_ms": 0, "median_ms": pair + 1, "mean_ms": pair + 1, "maximum_ms": 5}
            report = {
                "schema_version": 1, "runtime": "cpu", "correctness_verified": True,
                "metadata": {"compiler": "fixture"},
                "methodology": {"mode": mode, "scope": mode, "timer": "fixture",
                                "warmup_iterations": 1, "measured_iterations": 3},
                "benchmarks": {
                    "vector_add": dict(timing, elements=17),
                    "reduction": dict(timing, elements=17),
                    "matrix_multiply": dict(timing, rows=2, inner=2, columns=2),
                },
            }
            result.append({"pair": pair, "mode": mode, "report": report})
    return result


class CpuSuiteTest(unittest.TestCase):
    def test_known_process_statistics_and_no_mutation(self):
        data = records()
        before = copy.deepcopy(data)
        result = suite.summarize(data, 3)
        for mode in suite.MODES:
            for name in suite.NAMES:
                self.assertEqual(result[mode][name], {
                    "process_medians_ms": [1, 2, 3], "median_ms": 2,
                    "sample_stddev_ms": 1, "minimum_ms": 1, "maximum_ms": 3,
                })
        self.assertEqual(data, before)

    def test_missing_and_duplicate_process_pairs(self):
        for data in (records()[:-1], records()[:2], records()[:-1] + [records()[0]]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                suite.summarize(data, 3)

    def test_mixed_protocols_are_rejected(self):
        changes = (
            ("runtime", "gpu"), ("correctness_verified", False), ("schema_version", 2),
            ("metadata", {"compiler": "other"}),
        )
        for key, value in changes:
            data = records()
            data[-1]["report"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                suite.summarize(data, 3)
        for key, value in (("scope", "changed"), ("mode", "allocating"),
                           ("warmup_iterations", 2), ("measured_iterations", 4), ("timer", "other")):
            data = records()
            data[-1]["report"]["methodology"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                suite.summarize(data, 3)
        data = records()
        data[-1]["report"]["benchmarks"]["vector_add"]["elements"] = 18
        with self.assertRaises(ValueError):
            suite.summarize(data, 3)

    def test_bad_statistics(self):
        for value in (float("nan"), float("inf"), -1, True, 10):
            data = records()
            data[-1]["report"]["benchmarks"]["reduction"]["median_ms"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                suite.summarize(data, 3)

    def test_real_cli_build_and_process_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite.json"
            argv = [sys.executable, str(ROOT / "benchmarks/cpu_suite.py"), "--processes", "2",
                    "--warmup", "1", "--iterations", "3", "--vector-size", "257",
                    "--matrix-size", "17", "--output", str(output)]
            subprocess.run(argv, check=True, capture_output=True, text=True, timeout=120)
            report = json.loads(output.read_text())
            self.assertEqual(len(report["records"]), 4)
            self.assertEqual([r["mode"] for r in report["records"]],
                             ["allocating", "preallocated", "preallocated", "allocating"])
            self.assertEqual(report["summary"], suite.summarize(report["records"], 2))
            self.assertTrue(report["host"]["processor"])
            self.assertEqual(len(report["build"]["executable_sha256"]), 64)
            for path, sha in report["build"]["input_sha256"].items():
                self.assertEqual(suite.digest(ROOT / path), sha)
            original = output.read_bytes()
            process = subprocess.run(argv, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(output.read_bytes(), original)

    def test_missing_compiler_and_workload_bounds(self):
        with self.assertRaises(ValueError):
            suite.collect("missing-kernel-lab-compiler", 2, 1, 2, 0, 0)
        for value in ("1", "11"):
            with self.assertRaises(suite.argparse.ArgumentTypeError):
                suite.bounded(2, 10)(value)

    def test_published_measurement_artifact(self):
        artifact = json.loads((ROOT / "benchmarks/cpu-suite-20261020.json").read_text())
        self.assertEqual(artifact["summary"], suite.summarize(
            artifact["records"], artifact["protocol"]["processes_per_mode"]))
        self.assertEqual(artifact["build"]["runner_sha256"],
                         suite.digest(ROOT / "benchmarks/cpu_suite.py"))
        for path, sha in artifact["build"]["input_sha256"].items():
            self.assertEqual(suite.digest(ROOT / path), sha)


if __name__ == "__main__":
    unittest.main()
