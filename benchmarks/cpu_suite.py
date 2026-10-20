"""Build and collect serial, multi-process CPU reports using only the stdlib."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODES = ("allocating", "preallocated")
NAMES = ("vector_add", "reduction", "matrix_multiply")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_command(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=120).stdout


def cpu_model():
    description = platform.processor()
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return description or "unavailable"


def summarize(records, processes):
    """Reject mixed protocols instead of silently aggregating unrelated runs."""
    if len(records) != processes * 2 or processes < 2:
        raise ValueError("need two modes for every process pair")
    reference = records[0]["report"]
    identity = lambda report: (
        report["metadata"], report["methodology"]["timer"],
        report["methodology"]["warmup_iterations"],
        report["methodology"]["measured_iterations"],
        [(report["benchmarks"][name].get("elements"),
          report["benchmarks"][name].get("rows"),
          report["benchmarks"][name].get("inner"),
          report["benchmarks"][name].get("columns")) for name in NAMES],
    )
    seen = set()
    scopes = {}
    for record in records:
        mode, pair, report = record["mode"], record["pair"], record["report"]
        if mode not in MODES or type(pair) is not int or not 0 <= pair < processes:
            raise ValueError("invalid mode/process pair")
        if (mode, pair) in seen:
            raise ValueError("duplicate mode/process pair")
        seen.add((mode, pair))
        if (report["schema_version"] != 1 or report["runtime"] != "cpu" or
                report["correctness_verified"] is not True or
                report["methodology"]["mode"] != mode or identity(report) != identity(reference)):
            raise ValueError("incompatible or unverified benchmark report")
        scope = report["methodology"]["scope"]
        if mode in scopes and scopes[mode] != scope:
            raise ValueError("measurement scope changed within a mode")
        scopes[mode] = scope
        for name in NAMES:
            result = report["benchmarks"][name]
            values = [result[key] for key in ("minimum_ms", "median_ms", "mean_ms", "maximum_ms")]
            if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
                raise ValueError("invalid timing sample")
            if not values[0] <= values[1] <= values[3] or not values[0] <= values[2] <= values[3]:
                raise ValueError("unordered timing summary")
    result = {}
    for mode in MODES:
        result[mode] = {}
        for name in NAMES:
            medians = [r["report"]["benchmarks"][name]["median_ms"]
                       for r in records if r["mode"] == mode]
            result[mode][name] = {
                "process_medians_ms": medians,
                "median_ms": statistics.median(medians),
                "sample_stddev_ms": statistics.stdev(medians),
                "minimum_ms": min(medians), "maximum_ms": max(medians),
            }
    return result


def collect(compiler, processes, warmup, iterations, vector_size, matrix_size):
    compiler = shutil.which(compiler)
    if compiler is None:
        raise ValueError("C++ compiler unavailable")
    source = ROOT / "benchmarks/cpu_benchmark.cpp"
    inputs = [source, *sorted((ROOT / "include").rglob("*.hpp"))]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in inputs}
    with tempfile.TemporaryDirectory(prefix="kernel-lab-cpu-") as directory:
        binary = Path(directory) / "cpu_benchmark"
        command = [compiler, "-std=c++17", "-O2", "-DNDEBUG", "-Wall", "-Wextra",
                   "-Werror", "-pedantic", "-I", str(ROOT / "include"), str(source), "-o", str(binary)]
        compiler_version = run_command([compiler, "--version"])
        run_command(command)
        executable_hash = digest(binary)
        records = []
        for pair in range(processes):
            # Alternate mode order to avoid always measuring one mode first.
            order = MODES if pair % 2 == 0 else MODES[::-1]
            for mode in order:
                argv = [str(binary), "--warmup", str(warmup), "--iterations", str(iterations),
                        "--vector-size", str(vector_size), "--matrix-size", str(matrix_size)]
                if mode == "preallocated":
                    argv.append("--preallocated")
                report = json.loads(run_command(argv))
                if (report["methodology"]["warmup_iterations"] != warmup or
                        report["methodology"]["measured_iterations"] != iterations or
                        any(report["benchmarks"][name]["elements"] != vector_size
                            for name in ("vector_add", "reduction")) or
                        any(report["benchmarks"]["matrix_multiply"][key] != matrix_size
                            for key in ("rows", "inner", "columns"))):
                    raise ValueError("report differs from requested workload")
                records.append({"pair": pair, "mode": mode, "argv": argv, "report": report})
        if digest(binary) != executable_hash or any(digest(ROOT / path) != sha for path, sha in hashes.items()):
            raise ValueError("build inputs or executable changed during collection")
        return {
            "schema_version": 1, "collected_utc": datetime.now(timezone.utc).isoformat(),
            "host": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "processor": cpu_model(),
                     "logical_cpus": os.cpu_count(),
                     "python": platform.python_version()},
            "build": {"argv": command, "compiler_version": compiler_version,
                      "input_sha256": hashes, "executable_sha256": executable_hash,
                      "runner_sha256": digest(Path(__file__))},
            "protocol": {"processes_per_mode": processes, "execution": "serial alternating mode order",
                         "aggregation": "median and sample standard deviation of process medians"},
            "records": records, "summary": summarize(records, processes),
            "limitations": ["Single host; no CPU affinity, thermal or power control",
                            "Allocating and preallocated modes have different scopes; no kernel speedup claim",
                            "Shared inputs and checksum traversal can warm caches",
                            "Temporary build paths record exact argv but no longer exist after collection",
                            "CPU only; GPU correctness and measurements remain pending"],
        }


def bounded(minimum, maximum):
    def parse(value):
        number = int(value)
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(f"expected {minimum}..{maximum}")
        return number
    return parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--processes", type=bounded(2, 10), default=3)
    parser.add_argument("--warmup", type=bounded(0, 20), default=3)
    parser.add_argument("--iterations", type=bounded(1, 100), default=20)
    parser.add_argument("--vector-size", type=bounded(0, 1048576), default=65537)
    parser.add_argument("--matrix-size", type=bounded(0, 128), default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new artifact path")
    try:
        result = collect(args.compiler, args.processes, args.warmup, args.iterations,
                         args.vector_size, args.matrix_size)
        # Exclusive creation also protects against a concurrent writer.
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
