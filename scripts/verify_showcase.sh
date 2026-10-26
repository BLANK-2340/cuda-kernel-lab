#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
archive_root="$(mktemp -d "${TMPDIR:-/tmp}/kernel-lab-showcase.XXXXXX")"
trap 'rm -rf "$archive_root"' EXIT

git -C "$repo_root" archive --format=tar HEAD | tar -xf - -C "$archive_root"
cd "$archive_root"

compiler="${CXX:-g++}"
command -v "$compiler" >/dev/null
mkdir -p "$archive_root/build"

strict_flags=(-std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror -pedantic -I include)
test_names=(reference reduction matmul benchmark preallocated)
for name in "${test_names[@]}"; do
  "$compiler" "${strict_flags[@]}" "tests/${name}_test.cpp" \
    -o "$archive_root/build/${name}_test"
  "$archive_root/build/${name}_test"
done

"$compiler" "${strict_flags[@]}" examples/cpu_demo.cpp -o "$archive_root/build/cpu_demo"
demo_output="$("$archive_root/build/cpu_demo")"
test "$demo_output" = $'5 7 9 \nsum = 21'

"$compiler" "${strict_flags[@]}" benchmarks/cpu_benchmark.cpp \
  -o "$archive_root/build/cpu_benchmark"
python3 -W error tests/cpu_benchmark_cli_test.py "$archive_root/build/cpu_benchmark"
python3 -W error tests/cpu_suite_test.py
python3 -W error tests/cpu_suite_artifact_test.py
python3 -W error benchmarks/validate_cpu_suite.py benchmarks/cpu-suite-20261020.json
python3 -m compileall -q benchmarks tests

sanitizer_flags=(-std=c++17 -O1 -g -Wall -Wextra -Werror -pedantic
  -fsanitize=address,undefined -fno-omit-frame-pointer -I include)
for name in "${test_names[@]}"; do
  "$compiler" "${sanitizer_flags[@]}" "tests/${name}_test.cpp" \
    -o "$archive_root/build/${name}_sanitized"
  ASAN_OPTIONS=detect_leaks=0 "$archive_root/build/${name}_sanitized"
done

echo "showcase verification passed for $(git -C "$repo_root" rev-parse HEAD)"
