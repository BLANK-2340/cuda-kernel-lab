"""Adversarial checks for the independent committed-suite validator."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks/cpu-suite-20261020.json"
VALIDATOR = ROOT / "benchmarks/validate_cpu_suite.py"
spec = importlib.util.spec_from_file_location("validate_cpu_suite", VALIDATOR)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class CpuSuiteArtifactTest(unittest.TestCase):
    def setUp(self):
        self.document = validator.load_artifact(ARTIFACT)

    def test_committed_artifact_and_cli(self):
        self.assertEqual(validator.validate(self.document),
                         {"records": 6, "processes_per_mode": 3, "input_hashes": 4})
        process = subprocess.run([sys.executable, str(VALIDATOR), str(ARTIFACT)],
                                 check=True, capture_output=True, text=True)
        self.assertEqual(process.stdout.strip(),
                         "validated_records=6 processes_per_mode=3 input_hashes=4")

    def test_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, text in (("duplicate.json", '{"schema_version":1,"schema_version":1}'),
                               ("nonfinite.json", '{"value":NaN}')):
                path = Path(directory) / name
                path.write_text(text)
                with self.subTest(name=name), self.assertRaises(validator.ValidationError):
                    validator.load_artifact(path)

    def test_tampered_or_unsafe_source_hashes_are_rejected(self):
        for mutate in (
                lambda data: data["build"]["input_sha256"].update({"../outside": "0" * 64}),
                lambda data: data["build"]["input_sha256"].update(
                    {next(iter(data["build"]["input_sha256"])): "0" * 64}),
                lambda data: data["build"].update({"runner_sha256": "0" * 64})):
            data = copy.deepcopy(self.document)
            mutate(data)
            with self.assertRaises(validator.ValidationError):
                validator.validate(data)

    def test_process_identity_order_and_argv_mismatches_are_rejected(self):
        mutations = []
        duplicate = copy.deepcopy(self.document)
        duplicate["records"][1] = copy.deepcopy(duplicate["records"][0])
        mutations.append(duplicate)
        reordered = copy.deepcopy(self.document)
        reordered["records"][0], reordered["records"][1] = reordered["records"][1], reordered["records"][0]
        mutations.append(reordered)
        mismatched = copy.deepcopy(self.document)
        mismatched["records"][0]["argv"][6] = "257"
        mutations.append(mismatched)
        unknown = copy.deepcopy(self.document)
        unknown["records"][0]["argv"][1] = "--unknown"
        mutations.append(unknown)
        for data in mutations:
            with self.subTest(data=data["records"][0]), self.assertRaises(validator.ValidationError):
                validator.validate(data)

    def test_report_and_aggregate_tampering_are_rejected(self):
        mutations = []
        changed_report = copy.deepcopy(self.document)
        changed_report["records"][0]["report"]["benchmarks"]["reduction"]["minimum_ms"] = 99
        mutations.append(changed_report)
        changed_summary = copy.deepcopy(self.document)
        changed_summary["summary"]["allocating"]["vector_add"]["median_ms"] += 0.001
        mutations.append(changed_summary)
        extra_field = copy.deepcopy(self.document)
        extra_field["host"]["hostname"] = "should-not-be-persisted"
        mutations.append(extra_field)
        for data in mutations:
            with self.assertRaises(validator.ValidationError):
                validator.validate(data)


if __name__ == "__main__":
    unittest.main()
