# Kernel lab foundation

Original educational C++17 CPU references and optional CUDA vector-add,
reduction and tiled-matrix benchmarks. The existing root README is preserved. No downloaded data or external
CPU dependencies are required.

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

Next portfolio milestone: add reproducible physics-state tests and a headless
simulation path alongside the existing pendulum visualization. CUDA runtime
validation remains queued for the first available free NVIDIA environment.
