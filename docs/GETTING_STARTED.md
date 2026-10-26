# Kernel lab foundation

Original educational C++17 CPU references and optional CUDA vector-add,
reduction and tiled-matrix benchmarks. The existing root README is preserved. No downloaded data or external
CPU dependencies are required.

For the concise setup, runnable-target, measured-result and limitation index,
see [SHOWCASE.md](SHOWCASE.md).

## CPU setup and demo

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
./build/cpu_demo
```

Without CMake, from the repository root:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I include tests/reference_test.cpp -o /tmp/reference_test
/tmp/reference_test
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I include examples/cpu_demo.cpp -o /tmp/cpu_demo
/tmp/cpu_demo
```

Demo output: `5 7 9`, followed by `sum = 21`.

## Reproducible CPU benchmark report

Build in Release mode for measurements, then run the dependency-free benchmark:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target cpu_benchmark
./build-release/cpu_benchmark --warmup 5 --iterations 20 \
  --vector-size 1048576 --matrix-size 128 \
  --output cpu-benchmark.json
```

The JSON report has a versioned schema and records compiler, C++ language level,
build mode, hardware thread count, workload sizes, warmup/repetition counts,
measurement scope and minimum/median/mean/maximum milliseconds. All three
workloads are verified before timing. Input allocation, input generation,
verification and JSON serialization are excluded from timed regions. Vector-add
and matrix samples include the reference call's result allocation, a full checksum
traversal and result destruction; reduction samples include the scalar-returning
reference call. These are deliberately not allocation-free kernel timings. The
benchmark does not claim an end-to-end application rate or CPU/GPU speedup.

Executable-level report and error-path checks use only the Python standard library:

```sh
python3 tests/cpu_benchmark_cli_test.py ./build-release/cpu_benchmark
```

For comparisons, use the same executable, workload flags, machine power mode and
idle conditions. Record OS and CPU model externally because portable C++17 does
not expose reliable identifiers. Run several processes and compare medians rather
than treating one run as a universal result. The benchmark intentionally does not
mix CPU timings with the CUDA executables or calculate a speedup.

### Caller-owned output buffers

`kernel_lab/preallocated.hpp` adds `vector_add_into(lhs, rhs, output)` and
`matrix_multiply_into(lhs, rhs, output, m, k, n)`. Allocate the exact-sized output
before calling. Successful calls do not allocate or resize it. Every shape and
overflow check happens before writes. Vector addition permits exact input/output
aliasing; matrix multiplication rejects output aliasing either input, including
empty vectors. Matrix inputs may alias each other. Zero inner dimension writes
zeros to the entire correctly sized output. Accumulation remains double.

Use the same benchmark binary and workload controls for the separate mode:

```sh
./build-release/cpu_benchmark --preallocated --warmup 5 --iterations 20 \
  --vector-size 1048576 --matrix-size 128 --output cpu-preallocated.json
```

The additive `methodology.mode` field distinguishes `preallocated` from the
default `allocating` mode; schema version 1 and existing fields are retained.
Preallocated vector/matrix samples include the CPU reference and its shape/alias
checks, but exclude buffer allocation, checksum traversal, destruction and
correctness checks. Every output is consumed after stopping the timer and checked
before and after the complete measurement sequence. Reduction is unchanged in
both modes: one scalar-returning reference call. Warmup, repeated timings,
metadata and deterministic inputs are shared. Input/output data and post-timing
checksums can warm caches, and clock-call overhead remains significant for tiny
workloads. The two modes measure different scopes, not different algorithms;
their ratio is not a kernel optimization or GPU speedup. No measured comparison
is published in this milestone.

## CUDA setup (validation pending)

Requires an existing NVIDIA CUDA toolkit, compatible driver and GPU. This task
does not provision paid compute. Compile with an architecture supported by your
GPU/toolkit; replace 86 below as appropriate:

```sh
nvcc -std=c++17 -O3 -arch=sm_86 -I include cuda/vector_add_bench.cu -o /tmp/vector_add_bench
/tmp/vector_add_bench 1048576 100
/tmp/vector_add_bench 257 100
compute-sanitizer --tool memcheck /tmp/vector_add_bench 65537 10
nvcc -std=c++17 -O3 -arch=sm_86 -I include cuda/reduction_bench.cu -o /tmp/reduction_bench
/tmp/reduction_bench 1048576 100
compute-sanitizer --tool memcheck /tmp/reduction_bench 65537 10
nvcc -std=c++17 -O3 -arch=sm_86 -I include cuda/tiled_matmul_bench.cu -o /tmp/tiled_matmul_bench
/tmp/tiled_matmul_bench 512 512 512 100
/tmp/tiled_matmul_bench 17 9 23 10
compute-sanitizer --tool memcheck /tmp/tiled_matmul_bench 31 7 33 1
```

Alternatively configure CMake with `-DKERNEL_LAB_CUDA=ON` and
`-DCMAKE_CUDA_ARCHITECTURES=86`, then run `./build/vector_add_bench`,
`./build/reduction_bench` or `./build/tiled_matmul_bench`.

## Architecture and measurement scope

`reference.hpp` returns a new vector for elementwise addition, rejecting unequal
lengths. Reduction accumulates float inputs in double; the empty sum is zero.
Rectangular row-major matrix multiplication also accumulates in double before
rounding each output to float, with checked dimensions and storage shapes. These
are correctness oracles, not promises of exact real-number arithmetic.

The CUDA kernel uses bounds-checked grid-stride indexing with 256 threads per
block. The executable checks 12 sizes around warp/block boundaries, including
zero and 65537, then the requested size. Inputs are deterministic binary fractions;
their exact sums allow exact comparison with the CPU addition reference. This
does not establish correctness for all possible floating-point inputs.

Ten warmup launches and device synchronization precede CUDA-event timing of
repeated launches on the default stream. Every launch/API operation is checked;
output is copied back and verified before reporting a result. Reported mean
milliseconds include the event interval for the launch sequence, excluding host
allocation, transfers and CPU verification. Small workloads may include gaps
between launches. Effective GB/s counts two float reads and one float write per
element; it is not measured physical DRAM traffic or an end-to-end speedup.
Repeated inputs may benefit from caches. No CPU/GPU speedup claim is made.

The executable prints device name, compute capability, SM count, memory size,
driver/runtime versions and NVCC major/minor. Record the exact build command,
`nvcc --version`, OS and GPU power/clock conditions alongside future measurements.
Defaults: 1048576 elements, 100 timed launches. Sizes are bounded at 16777216,
repeats at 1–10000. Zero size skips allocation and launch.

The reduction benchmark performs a first float-to-double block reduction followed
by double partial-sum reductions until one value remains. Each thread consumes two
elements per iteration and uses grid-stride coverage, so capped grid sizes still
cover the entire input. Its event interval includes every kernel in each reduction
but excludes allocation, host/device transfer and verification. Deterministic
dyadic inputs make the expected sum exactly representable in double, allowing an
exact correctness check despite parallel ordering. This is not a claim of exact
summation for arbitrary floats. No bandwidth or speedup is reported before GPU
validation.

The matrix benchmark uses 16x16 shared-memory tiles and zero-fills partial tiles,
so arbitrary rectangular dimensions are bounds safe. It verifies shapes below,
at and above the tile boundary before timing the requested shape. Verification
uses a documented absolute/relative tolerance against the double-accumulating CPU
oracle because the float kernel has a different summation order. Ten warmups
precede CUDA-event timing of kernel launches only; allocation, transfers and CPU
verification are excluded. Reported arithmetic GFLOP/s uses the conventional
`2*M*K*N` operation count and is not an end-to-end application rate.

Methodology reference: [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html),
especially timing, effective bandwidth and CUDA error checking. No third-party
source code or assets were incorporated.

## Actual validation and continuity — October 5, 2026

- GCC 13.3.0: strict warning compilation and CPU tests passed for 12 boundary
  sizes, cancellation, fractional inputs and mismatched lengths.
- AddressSanitizer/UndefinedBehaviorSanitizer: CPU tests passed with
  `ASAN_OPTIONS=detect_leaks=0`. LeakSanitizer was attempted but failed because
  this environment blocks process inspection; leak validation is unavailable.
- CPU demo passed with the expected values.
- CMake/CTest not executed locally: CMake unavailable. A CPU CI workflow is
  included; its result must be checked after publication.
- NVCC compilation, GPU correctness, compute-sanitizer and all GPU measurements
  are pending: no NVCC or GPU tooling is available in this environment.

## Reduction milestone — October 6, 2026

- Added a separate CPU oracle stress test across 19 sizes through 1000003 values,
  cancellation and IEEE infinity/NaN propagation.
- Added a bounds-safe multi-stage CUDA reduction with double accumulation, checked
  API/kernel launches, edge-size validation, warmup and full reduction timing.
- GCC 13.3.0 strict-warning builds passed for both CPU test binaries and the demo.
  Both CPU tests passed AddressSanitizer/UndefinedBehaviorSanitizer with leak
  detection disabled because process inspection remains blocked here.
- CMake/CTest was unavailable locally. NVCC, GPU execution and compute-sanitizer
  remain unavailable and unvalidated; no GPU result is claimed here. CPU CI covers
  CMake/CTest and sanitizers, and its result must be checked after publication.

The vector-add and reduction implementations still require NVCC,
compute-sanitizer and actual-GPU validation when a free runtime becomes available.
The subsequent numeric MLP save/load and vision serialization/PGM portfolio
rotations were completed before the matrix-multiplication milestone below.

## Tiled matrix multiplication milestone — October 8, 2026

- Added a rectangular CPU oracle with double accumulation, dimension-overflow and
  storage-shape checks. CPU tests cover 11 shapes including empty dimensions and
  sizes 15, 16 and 17 around the CUDA tile boundary.
- Added an optional bounds-safe 16x16 shared-memory CUDA kernel, deterministic
  edge-shape checks, warmup, kernel-only event timing and hardware/toolchain
  metadata. No CUDA performance result is recorded without actual GPU execution.
- GCC 13.3.0 strict-warning builds passed for all three CPU tests and the demo.
  AddressSanitizer/UndefinedBehaviorSanitizer passed for all CPU tests with leak
  detection disabled consistently with the earlier environment limitation.
- CMake/CTest was unavailable locally and is delegated to CPU CI. NVCC,
  compute-sanitizer, GPU correctness and measurements remain pending because no
  CUDA compiler or NVIDIA GPU tooling was available; no GPU result is claimed.

## CPU benchmark reporting milestone — October 13, 2026

- Added a dependency-free C++17 benchmark for vector addition, reduction and
  square matrix multiplication with deterministic inputs and pre-timing checks.
- Added versioned JSON output, warmup/repetition controls, workload bounds,
  steady-clock timing summaries and portable compiler/build/thread metadata.
- Added unit coverage for statistics, invalid samples and JSON escaping. CPU CI
  builds the benchmark, runs a small boundary workload and parses/asserts the
  generated JSON schema.
- Local compilation was unavailable during this session because the execution
  environment could not be reached. Publication therefore requires the task-owned
  branch CI to pass before this commit is advanced to `main`.
- NVCC, compute-sanitizer, GPU correctness and measurements remain pending. No GPU
  result is claimed.

Next portfolio milestone: add numeric tensor primitives and finite-difference
gradient checks to the C++ autograd engine. CUDA GPU validation remains queued for
the first available free NVIDIA environment.

## CPU measurement-scope correction — October 19, 2026

- Corrected the JSON and documentation: vector/matrix timings include result
  allocation, checksum traversal and destruction. No timing boundaries or kernels
  changed, and schema version 1 and its existing field structure are preserved.
- Added four standard-library Python executable-contract tests: empty/singleton/
  boundary reports, finite and ordered statistics, metadata, file output (including
  spaces), open failure, help and twelve invalid-argument cases. CI runs these
  against its compiled benchmark.
- GCC 13.3.0 strict-warning Release builds passed for all four C++ tests, the CPU
  demo and benchmark. All tests and the demo passed. All four C++ tests and all
  four CLI tests also passed with AddressSanitizer/UndefinedBehaviorSanitizer;
  leak detection was disabled because of the previously documented environment
  limitation. The CLI regression checks deliberately reject the previous report's
  inaccurate scope description.
- CMake/CTest, NVCC, NVIDIA GPU execution and compute-sanitizer remain unavailable
  locally. Remote CI still requires separate confirmation. No GPU measurements,
  CPU/GPU comparisons or new performance conclusions are published here.

Next CUDA milestone: add an explicitly caller-owned output-buffer CPU reference
and separately scoped benchmark if allocation-free comparisons are needed; retain
the current allocating-call measurements under their accurate scope. The C++
rotation has already progressed through tensor primitives, classification loss,
row broadcasting and deterministic multiclass training (latest commit
`75f0673e7998598c5b014d92157a297fe57124b6`); do not rebuild those milestones.

## Preallocated CPU references — October 20, 2026

- Added caller-owned vector/matrix output APIs in a new header, retaining the
  existing allocating references unchanged. Added explicit shape, overflow and
  alias behavior and a separate `--preallocated` benchmark mode.
- Added arithmetic and rejection tests across nine vector boundary sizes,
  twenty rectangular matrix shapes, a known nonuniform product, zero-column and
  zero-inner dimensions, exact vector aliases, invalid matrix aliases, repeated
  output reuse and all three dimension-overflow checks. A test-only new/delete
  probe observes zero allocations during valid API calls.
- GCC 13.3.0 strict-warning Release builds passed for five C++ test binaries,
  the CPU demo and benchmark. Five CLI tests passed, including reports for both
  modes with empty, singleton and boundary workloads. ASan/UBSan checks passed
  for the C++ tests and CLI suite with local leak detection disabled, consistent
  with the environment limitation documented above.
- CMake/CTest, NVCC, GPU execution and compute-sanitizer are unavailable locally;
  remote CI requires separate confirmation. No GPU or comparative performance
  results are claimed. Original root README and CUDA sources are preserved.

Next concrete CUDA milestone: persist exact build/host metadata and controlled
multi-process CPU reports before comparing timing distributions. Keep actual GPU
correctness and measurement pending until a free NVIDIA runtime is available.

## Multi-process CPU collection — October 20, 2026

The standard-library Python collector builds the current C++ benchmark once with
explicit C++17, `-O2`, `-DNDEBUG` and strict-warning flags. It launches three fresh
processes per mode serially, alternating which mode runs first in each pair.
Each process uses three warmups, twenty measured calls, 65537 vector elements and
64x64 matrices by default. No third-party code, data or paid resources are used.

```sh
python3 benchmarks/cpu_suite.py --output /tmp/cpu-suite-new.json
python3 -W error tests/cpu_suite_test.py
```

Choose a new output path each time: existing artifacts are never overwritten.
The collector retains all six raw reports, exact build/run argument arrays,
compiler version, SHA-256 hashes of the executable and build inputs, the runner
hash, collection time, OS/kernel, architecture, logical CPU count and CPU model
when exposed by the host. It deliberately omits hostname and account identity.
Temporary build paths in the commands cease to exist after collection; rerun the
collector to rebuild from recorded source inputs and flags.

Aggregation reports the median and sample standard deviation of the three
**process medians**, not a pooled per-call standard deviation or confidence
interval. Mixed metadata, workloads, warmups, sample counts, timers or per-mode
scopes, duplicate/missing process pairs, failed correctness checks and non-finite
or unordered timing statistics are rejected before publication of an artifact.

Actual CPU-only snapshot: `benchmarks/cpu-suite-20261020.json`, collected with
GCC 13.3.0 on Linux/x86_64, reporting INTEL XEON PLATINUM 8573C and nine logical
CPUs. The table rounds the persisted milliseconds to six decimal places.

| Operation | Mode | Median of process medians (ms) | Sample SD (ms) |
| --- | --- | ---: | ---: |
| Vector addition | Allocating | 0.077863 | 0.001668 |
| Vector addition | Preallocated | 0.027052 | 0.000423 |
| Reduction | Allocating | 0.045291 | 0.000868 |
| Reduction | Preallocated | 0.043788 | 0.000862 |
| Matrix multiplication | Allocating | 0.203346 | 0.018275 |
| Matrix multiplication | Preallocated | 0.177501 | 0.014039 |

These are observations from one shared execution environment, not portable
performance guarantees. The final snapshot was collected without another task
build running concurrently, but affinity, clock/power state, thermal behavior,
other host workloads and cache state were not controlled. The matrix variability
is retained rather than hidden. Different allocation/checksum timing scopes mean
mode ratios are **not kernel speedups**. Reduction executes the same code in both
modes. No GPU or CPU/GPU comparison is measured or inferred.

Verification: seven collector tests passed with warnings treated as errors,
including an actual compile/four-process CLI test, overwrite protection and
persisted-artifact/hash consistency. All five C++ test binaries, CPU demo and five
existing CLI tests passed strict GCC builds; all five C++ test binaries passed
ASan/UBSan with local leak detection disabled. Python bytecode compilation and
diff checks passed. CI now runs the collector tests, but remote CI remains to be
confirmed. CMake/CTest, NVCC, compute-sanitizer and actual GPU validation remain
unavailable locally.

Next concrete milestone: rotate to C++ multiclass model persistence and robust
round-trip/invalid-format checks, after inspecting its latest continuity notes.
Keep language work blocked only on the missing exact authorized repositories;
do not put it into a different project. CUDA GPU validation remains pending.

## Clean-checkout showcase verification — October 26, 2026

- Added a one-command verifier that exports committed `HEAD`, performs strict
  release builds, runs all five C++ tests, five executable-contract tests and
  seven collector tests, checks the committed CPU suite and compiles bytecode.
- The same isolated archive reruns all C++ tests with AddressSanitizer and
  UndefinedBehaviorSanitizer; leak detection remains disabled because the local
  environment blocks it.
- Added a compact evidence-backed showcase index with exact setup/demo commands,
  the recorded one-host CPU snapshot, architecture links and explicit limits.
- Local validation used GCC 13.3.0 and Python 3.12.14. CMake, NVCC, NVIDIA GPU
  execution and compute-sanitizer remained unavailable; no GPU result or speedup
  claim was added.

Next concrete milestone: add a strict standalone validator for the committed CPU
suite schema, source hashes, process pairs and aggregates. Keep GPU validation
pending until an actual compatible CUDA runtime is available.

## Standalone CPU evidence validation — October 26, 2026

- Added a dependency-free validator that does not import the producing collector.
  It bounds input size, rejects duplicate JSON keys and non-finite values, enforces
  exact schemas, verifies normalized repository-relative source hashes and checks
  that raw process identities follow the recorded alternating protocol.
- Run arguments are checked against every report's workload, mode and iteration
  counts. Timing fields must be finite, non-negative and ordered. Process medians,
  median-of-medians, sample standard deviation and extrema are recomputed from raw
  records instead of trusting the persisted summary.
- Five adversarial tests cover the committed artifact and CLI, duplicate/non-finite
  JSON, path traversal and hash tampering, duplicate/reordered process identities,
  argument/report mismatches, malformed timings, summary tampering and unexpected
  identity fields. The clean-archive verifier and CPU CI run both tests and CLI.

Next concrete milestone: rotate to the pendulum simulation for clean-checkout
showcase verification and an evidence index. CUDA compilation, GPU correctness,
compute-sanitizer and GPU measurements remain pending an actual compatible runtime.
