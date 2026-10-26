"""Validate a persisted CPU suite without importing its producing collector."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics


ROOT = Path(__file__).resolve().parents[1]
MODES = ("allocating", "preallocated")
OPERATIONS = ("vector_add", "reduction", "matrix_multiply")
MAX_BYTES = 2 * 1024 * 1024


class ValidationError(ValueError):
    """The artifact is unsafe, malformed or internally inconsistent."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def exact_keys(value, expected, label):
    require(type(value) is dict, f"{label} must be an object")
    require(set(value) == set(expected), f"{label} has unexpected or missing fields")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"non-finite JSON number: {value}")


def load_artifact(path):
    path = Path(path)
    require(path.exists() and path.is_file() and not path.is_symlink(),
            "artifact must be a regular non-symlink file")
    size = path.stat().st_size
    require(0 < size <= MAX_BYTES, "artifact size is outside the accepted range")
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid JSON artifact: {error}") from error


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value, minimum, maximum, label):
    require(type(value) is int and minimum <= value <= maximum,
            f"{label} must be an integer in {minimum}..{maximum}")


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"{label} must be a finite non-negative number")


def sha256(value, label):
    require(type(value) is str and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            f"{label} must be a lowercase SHA-256 digest")


def parse_run_argv(argv, mode):
    require(type(argv) is list and all(type(item) is str and item for item in argv),
            "record argv must be a non-empty string array")
    require(len(argv) in (9, 10), "record argv has an unexpected number of arguments")
    require(argv[1:9:2] == ["--warmup", "--iterations", "--vector-size", "--matrix-size"],
            "record argv flags or order are invalid")
    require((argv[9:] == ["--preallocated"]) == (mode == "preallocated"),
            "record argv mode does not match its record")
    values = []
    for offset, limits in zip((2, 4, 6, 8), ((0, 20), (1, 100), (0, 1048576), (0, 128))):
        try:
            value = int(argv[offset])
        except ValueError as error:
            raise ValidationError("record argv workload value is not an integer") from error
        require(str(value) == argv[offset], "record argv integer is not canonical")
        integer(value, limits[0], limits[1], "record argv workload")
        values.append(value)
    return tuple(values)


def validate_timing(result, dimensions, label):
    expected = {*dimensions, "minimum_ms", "median_ms", "mean_ms", "maximum_ms"}
    exact_keys(result, expected, label)
    for key in dimensions:
        integer(result[key], 0, 1048576, f"{label}.{key}")
    values = [result[key] for key in ("minimum_ms", "median_ms", "mean_ms", "maximum_ms")]
    for key, value in zip(("minimum_ms", "median_ms", "mean_ms", "maximum_ms"), values):
        number(value, f"{label}.{key}")
    require(values[0] <= values[1] <= values[3] and values[0] <= values[2] <= values[3],
            f"{label} timings are unordered")


def validate_report(report, mode, requested):
    exact_keys(report, ("benchmarks", "correctness_verified", "limitations", "metadata",
                        "methodology", "runtime", "schema_version"), "report")
    require(report["schema_version"] == 1 and report["runtime"] == "cpu" and
            report["correctness_verified"] is True, "report is not a verified CPU schema v1 report")
    exact_keys(report["metadata"], ("build_mode", "compiler", "compiler_version",
                                    "cplusplus", "hardware_threads"), "report.metadata")
    metadata = report["metadata"]
    require(all(type(metadata[key]) is str and metadata[key] for key in
                ("build_mode", "compiler", "compiler_version")), "metadata strings must be non-empty")
    integer(metadata["cplusplus"], 201703, 999999, "metadata.cplusplus")
    integer(metadata["hardware_threads"], 0, 1048576, "metadata.hardware_threads")
    exact_keys(report["methodology"], ("measured_iterations", "mode", "scope", "timer",
                                       "warmup_iterations"), "report.methodology")
    method = report["methodology"]
    require(method["mode"] == mode and type(method["scope"]) is str and method["scope"] and
            type(method["timer"]) is str and method["timer"], "methodology identity is invalid")
    warmup, iterations, vector_size, matrix_size = requested
    require(method["warmup_iterations"] == warmup and method["measured_iterations"] == iterations,
            "report iteration counts do not match argv")
    exact_keys(report["benchmarks"], OPERATIONS, "report.benchmarks")
    validate_timing(report["benchmarks"]["vector_add"], ("elements",), "vector_add")
    validate_timing(report["benchmarks"]["reduction"], ("elements",), "reduction")
    validate_timing(report["benchmarks"]["matrix_multiply"],
                    ("rows", "inner", "columns"), "matrix_multiply")
    require(report["benchmarks"]["vector_add"]["elements"] == vector_size and
            report["benchmarks"]["reduction"]["elements"] == vector_size,
            "vector workload does not match argv")
    matrix = report["benchmarks"]["matrix_multiply"]
    require((matrix["rows"], matrix["inner"], matrix["columns"]) ==
            (matrix_size, matrix_size, matrix_size), "matrix workload does not match argv")
    require(type(report["limitations"]) is list and report["limitations"] and
            all(type(item) is str and item for item in report["limitations"]),
            "report limitations must be non-empty strings")


def validate_hashes(build, root):
    sha256(build["executable_sha256"], "build.executable_sha256")
    sha256(build["runner_sha256"], "build.runner_sha256")
    require(build["runner_sha256"] == digest(root / "benchmarks/cpu_suite.py"),
            "collector source hash mismatch")
    inputs = build["input_sha256"]
    require(type(inputs) is dict and inputs, "build.input_sha256 must be a non-empty object")
    for name, expected in inputs.items():
        require(type(name) is str and name, "source path must be a non-empty string")
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts and "." not in relative.parts,
                "source path must be a normalized repository-relative path")
        candidate = root.joinpath(*relative.parts)
        require(candidate.is_file() and not candidate.is_symlink(), f"source input is missing: {name}")
        sha256(expected, f"source hash for {name}")
        require(digest(candidate) == expected, f"source hash mismatch: {name}")


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def validate_summary(summary, records):
    exact_keys(summary, MODES, "summary")
    for mode in MODES:
        exact_keys(summary[mode], OPERATIONS, f"summary.{mode}")
        for operation in OPERATIONS:
            item = summary[mode][operation]
            exact_keys(item, ("process_medians_ms", "median_ms", "sample_stddev_ms",
                              "minimum_ms", "maximum_ms"), f"summary.{mode}.{operation}")
            observed = [record["report"]["benchmarks"][operation]["median_ms"]
                        for record in records if record["mode"] == mode]
            require(type(item["process_medians_ms"]) is list and
                    len(item["process_medians_ms"]) == len(observed), "summary sample count mismatch")
            require(all(close(a, b) for a, b in zip(item["process_medians_ms"], observed)),
                    "summary process medians do not match raw records")
            expected = (statistics.median(observed), statistics.stdev(observed),
                        min(observed), max(observed))
            actual = tuple(item[key] for key in
                           ("median_ms", "sample_stddev_ms", "minimum_ms", "maximum_ms"))
            for value in actual:
                number(value, "summary statistic")
            require(all(close(a, b) for a, b in zip(actual, expected)),
                    "summary aggregate does not match raw records")


def validate(document, root=ROOT):
    exact_keys(document, ("build", "collected_utc", "host", "limitations", "protocol",
                          "records", "schema_version", "summary"), "artifact")
    require(document["schema_version"] == 1, "artifact schema version must be 1")
    try:
        timestamp = datetime.fromisoformat(document["collected_utc"])
    except (TypeError, ValueError) as error:
        raise ValidationError("collected_utc is not an ISO-8601 timestamp") from error
    require(timestamp.utcoffset() is not None and timestamp.utcoffset().total_seconds() == 0,
            "collected_utc must explicitly use UTC")
    exact_keys(document["build"], ("argv", "compiler_version", "executable_sha256",
                                    "input_sha256", "runner_sha256"), "build")
    require(type(document["build"]["argv"]) is list and document["build"]["argv"] and
            all(type(item) is str for item in document["build"]["argv"]), "build argv is invalid")
    require(type(document["build"]["compiler_version"]) is str and
            document["build"]["compiler_version"], "compiler version is missing")
    validate_hashes(document["build"], Path(root))
    exact_keys(document["host"], ("logical_cpus", "machine", "processor", "python", "release", "system"),
               "host")
    integer(document["host"]["logical_cpus"], 1, 1048576, "host.logical_cpus")
    require(all(type(value) is str and value for key, value in document["host"].items()
                if key != "logical_cpus"), "host strings must be non-empty")
    exact_keys(document["protocol"], ("aggregation", "execution", "processes_per_mode"), "protocol")
    protocol = document["protocol"]
    integer(protocol["processes_per_mode"], 2, 10, "protocol.processes_per_mode")
    require(protocol["execution"] == "serial alternating mode order" and
            protocol["aggregation"] == "median and sample standard deviation of process medians",
            "unsupported collection protocol")
    records = document["records"]
    processes = protocol["processes_per_mode"]
    require(type(records) is list and len(records) == processes * 2, "record count is incomplete")
    identities = []
    reference = None
    scopes = {}
    for index, record in enumerate(records):
        exact_keys(record, ("argv", "mode", "pair", "report"), f"record {index}")
        mode, pair = record["mode"], record["pair"]
        require(mode in MODES, "record mode is invalid")
        integer(pair, 0, processes - 1, "record pair")
        expected_mode = MODES[(index % 2) if pair % 2 == 0 else 1 - (index % 2)]
        require(index // 2 == pair and mode == expected_mode, "records are not in alternating pair order")
        identities.append((pair, mode))
        requested = parse_run_argv(record["argv"], mode)
        validate_report(record["report"], mode, requested)
        report = record["report"]
        identity = (report["metadata"], report["methodology"]["timer"],
                    report["methodology"]["warmup_iterations"],
                    report["methodology"]["measured_iterations"], requested[2:])
        reference = identity if reference is None else reference
        require(identity == reference, "record metadata or workload changed within the suite")
        scopes.setdefault(mode, report["methodology"]["scope"])
        require(scopes[mode] == report["methodology"]["scope"], "measurement scope changed within a mode")
    require(len(set(identities)) == len(identities), "duplicate process identity")
    validate_summary(document["summary"], records)
    require(type(document["limitations"]) is list and document["limitations"] and
            all(type(item) is str and item for item in document["limitations"]),
            "artifact limitations must be non-empty strings")
    return {"records": len(records), "processes_per_mode": processes,
            "input_hashes": len(document["build"]["input_sha256"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    try:
        result = validate(load_artifact(args.artifact))
    except ValidationError as error:
        parser.exit(1, f"error: {error}\n")
    print("validated_records={records} processes_per_mode={processes_per_mode} "
          "input_hashes={input_hashes}".format(**result))


if __name__ == "__main__":
    main()
