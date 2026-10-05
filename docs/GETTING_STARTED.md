# Kernel lab foundation

Original educational C++17 CPU references and an optional CUDA vector-add
benchmark. The existing root README is preserved. No downloaded data or external
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
```

Alternatively configure CMake with `-DKERNEL_LAB_CUDA=ON` and
`-DCMAKE_CUDA_ARCHITECTURES=86`, then run `./build/vector_add_bench`.

## Architecture and measurement scope

`reference.hpp` returns a new vector for elementwise addition, rejecting unequal
lengths. Reduction accumulates float inputs in double; the empty sum is zero.
It is a correctness oracle, not a promise of exact summation for all inputs.

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

Next CUDA milestone: add a bounds-safe GPU reduction, compare it with the double
CPU oracle using a documented tolerance, and validate vector-add/reduction on an
actual GPU when free runtime access becomes available. Next portfolio session:
establish the self-contained VisionTrack-NN baseline; C++ MLP/XOR is already
published, with model save/load remaining as its next extension.
