"""Black-box checks for the compiled CPU benchmark (Python standard library only)."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BINARY = str(Path(sys.argv.pop(1)).resolve())


class BenchmarkCliTest(unittest.TestCase):
    def run_benchmark(self, *arguments):
        return subprocess.run(
            [BINARY, *arguments], capture_output=True, text=True, timeout=20,
            check=False,
        )

    def check_report(self, report, vector_size, matrix_size, preallocated=False):
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["runtime"], "cpu")
        self.assertIs(report["correctness_verified"], True)
        metadata = report["metadata"]
        self.assertTrue(metadata["compiler"])
        self.assertTrue(metadata["compiler_version"])
        self.assertGreaterEqual(metadata["cplusplus"], 201703)
        self.assertIn(metadata["build_mode"], ("release", "debug"))
        self.assertGreaterEqual(metadata["hardware_threads"], 0)
        methodology = report["methodology"]
        self.assertEqual(methodology["warmup_iterations"], 0)
        self.assertEqual(methodology["measured_iterations"], 3)
        scope = methodology["scope"]
        self.assertEqual(methodology["mode"], "preallocated" if preallocated else "allocating")
        if preallocated:
            self.assertIn("including shape/alias validation", scope)
            self.assertIn("all allocation, checksum traversal, destruction", scope)
            self.assertIn("excluded", scope)
        else:
            self.assertIn("result allocation, checksum traversal and destruction included", scope)
            self.assertIn("correctness verification and report serialization excluded", scope)
        results = report["benchmarks"]
        self.assertEqual(set(results), {"vector_add", "reduction", "matrix_multiply"})
        for name in ("vector_add", "reduction"):
            self.assertEqual(results[name]["elements"], vector_size)
        matrix = results["matrix_multiply"]
        self.assertEqual(
            (matrix["rows"], matrix["inner"], matrix["columns"]),
            (matrix_size,) * 3,
        )
        for result in results.values():
            for key in ("minimum_ms", "median_ms", "mean_ms", "maximum_ms"):
                self.assertIsInstance(result[key], (int, float))
                self.assertTrue(math.isfinite(result[key]))
                self.assertGreaterEqual(result[key], 0)
            self.assertLessEqual(result["minimum_ms"], result["median_ms"])
            self.assertLessEqual(result["median_ms"], result["maximum_ms"])
            self.assertLessEqual(result["minimum_ms"], result["mean_ms"])
            self.assertLessEqual(result["mean_ms"], result["maximum_ms"])

    def test_stdout_reports_for_empty_and_boundary_workloads(self):
        for vector_size, matrix_size in ((0, 0), (1, 1), (257, 17)):
            with self.subTest(vector_size=vector_size, matrix_size=matrix_size):
                process = self.run_benchmark(
                    "--warmup", "0", "--iterations", "3",
                    "--vector-size", str(vector_size),
                    "--matrix-size", str(matrix_size),
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(process.stderr, "")
                self.check_report(json.loads(process.stdout), vector_size, matrix_size)

    def test_output_file_and_open_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report with spaces.json"
            process = self.run_benchmark(
                "--warmup", "0", "--iterations", "3", "--vector-size", "1",
                "--matrix-size", "1", "--output", str(output),
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stdout, f"wrote {output}\n")
            self.check_report(json.loads(output.read_text()), 1, 1)
            process = self.run_benchmark(
                "--iterations", "1", "--vector-size", "0", "--matrix-size", "0",
                "--output", directory,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(process.stdout, "")
            self.assertIn("could not open output file", process.stderr)

    def test_preallocated_reports(self):
        for vector_size, matrix_size in ((0, 0), (1, 1), (257, 17)):
            with self.subTest(vector_size=vector_size, matrix_size=matrix_size):
                process = self.run_benchmark(
                    "--preallocated", "--warmup", "0", "--iterations", "3",
                    "--vector-size", str(vector_size), "--matrix-size", str(matrix_size),
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                self.check_report(json.loads(process.stdout), vector_size, matrix_size, True)

    def test_invalid_arguments_fail_without_a_report(self):
        cases = (
            ("--iterations", "0"), ("--iterations", "10001"),
            ("--warmup", "1001"), ("--vector-size", "16777217"),
            ("--matrix-size", "513"), ("--matrix-size", "-1"),
            ("--iterations", "1.5"), ("--iterations", "1x"),
            ("--vector-size", "18446744073709551616"),
            ("--iterations", ""), ("--iterations",), ("--unknown", "1"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                process = self.run_benchmark(*arguments)
                self.assertNotEqual(process.returncode, 0)
                self.assertEqual(process.stdout, "")
                self.assertIn("error:", process.stderr)

    def test_help_does_not_run_workloads(self):
        process = self.run_benchmark("--help")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Usage: cpu_benchmark", process.stdout)
        self.assertEqual(process.stderr, "")


if __name__ == "__main__":
    unittest.main()
