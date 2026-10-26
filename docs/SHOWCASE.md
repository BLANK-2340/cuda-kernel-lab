# CUDA kernel lab showcase

This project pairs dependency-free C++17 correctness references with optional,
bounds-checked CUDA vector addition, reduction and tiled matrix multiplication.
CPU tests and benchmarks run without a GPU. The repository's original root
`README.md` remains unchanged.

## Setup and clean-checkout verification

Required CPU tools: Python 3, a C++17 compiler, `git` and `tar`.

```sh
./scripts/verify_showcase.sh
```

The verifier exports committed `HEAD` with `git archive`, builds every CPU test,
the demo and benchmark under strict release flags, and then runs five C++ tests,
five black-box benchmark tests, seven collector tests and five adversarial artifact
tests. It also validates the published CPU report's strict schema, current source
hashes, process identities and independently recomputed aggregates, compiles Python
bytecode and reruns all C++ tests under AddressSanitizer/UndefinedBehaviorSanitizer. Leak
detection is disabled because the execution environment blocks it.

## Runnable targets

| Target | Command | Scope |
| --- | --- | --- |
| CPU demo | `g++ -std=c++17 -O2 -I include examples/cpu_demo.cpp -o /tmp/cpu_demo && /tmp/cpu_demo` | Vector addition and reduction smoke path |
| Single-process benchmark | `g++ -std=c++17 -O2 -DNDEBUG -I include benchmarks/cpu_benchmark.cpp -o /tmp/cpu_benchmark && /tmp/cpu_benchmark --warmup 5 --iterations 20 --vector-size 1048576 --matrix-size 128` | Verified CPU references with explicit timing scope |
| Multi-process collection | `python3 benchmarks/cpu_suite.py --output /tmp/cpu-suite.json` | Alternating allocating/preallocated processes with raw reports, hashes and variability |
| Validate committed evidence | `python3 benchmarks/validate_cpu_suite.py benchmarks/cpu-suite-20261020.json` | Bounded strict JSON, source hashes, process protocol and independent aggregates |
| Complete CPU verification | `./scripts/verify_showcase.sh` | Clean committed archive, strict builds, CLI contracts, artifact consistency and sanitizers |
| Optional CUDA benchmarks | See [CUDA setup](GETTING_STARTED.md#cuda-setup-validation-pending) | Requires an existing compatible NVIDIA toolkit and GPU; validation is pending |

## Recorded CPU result

The checked-in [`cpu-suite-20261020.json`](../benchmarks/cpu-suite-20261020.json)
contains six raw processes collected on one Linux/x86-64 host with GCC 13.3.0.
These figures are the median and sample standard deviation of three process
medians in milliseconds.

| Operation | Allocating median ± SD | Preallocated median ± SD |
| --- | ---: | ---: |
| Vector addition | `0.077863 ± 0.001668` | `0.027052 ± 0.000423` |
| Reduction | `0.045291 ± 0.000868` | `0.043788 ± 0.000862` |
| Matrix multiplication | `0.203346 ± 0.018275` | `0.177501 ± 0.014039` |

The two modes have different timing boundaries, so their ratios are not kernel
speedups. Reduction executes the same implementation in both modes. The snapshot
is host-specific and does not imply portable performance.

## Architecture and evidence map

- [`reference.hpp`](../include/kernel_lab/reference.hpp): allocating CPU oracles
- [`preallocated.hpp`](../include/kernel_lab/preallocated.hpp): checked
  caller-owned vector and matrix outputs
- [`benchmark.hpp`](../include/kernel_lab/benchmark.hpp): timing statistics,
  compiler identity and JSON escaping
- [Getting started, methodology and milestone evidence](GETTING_STARTED.md)
- [`cpu_suite.py`](../benchmarks/cpu_suite.py): controlled build and process
  collection with input/executable hashes
- [`validate_cpu_suite.py`](../benchmarks/validate_cpu_suite.py): independent,
  dependency-free persisted-evidence validator

## Known limitations

- CPU timings come from one shared host without affinity, clock/power, thermal or
  competing-load control. They are observations, not performance guarantees.
- Allocating and preallocated modes intentionally measure different scopes.
- The CPU references prioritize clarity and verification over optimized-library
  performance; there is no Eigen, BLAS or vendor-library comparison.
- NVCC compilation, CUDA execution, compute-sanitizer, GPU correctness and GPU
  measurements remain unvalidated because no NVIDIA runtime is available here.
- No CPU/GPU speedup, end-to-end application rate or production claim is made.

The next portfolio milestone is clean-checkout showcase verification for the
pendulum simulation while GPU work remains pending actual CUDA access.
