#include "kernel_lab/reference.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    const std::vector<std::size_t> sizes = {
        0, 1, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513,
        1023, 1024, 1025, 65535, 65536, 65537, 1000003};
    for (const std::size_t n : sizes) {
        std::vector<float> values(n);
        double expected = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const int numerator = static_cast<int>((i * 37 + 11) % 257) - 128;
            values[i] = static_cast<float>(numerator) / 16.0f;
            expected += static_cast<double>(numerator) / 16.0;
        }
        require(kernel_lab::reduce_sum(values) == expected, "dyadic reduction mismatch");
    }

    require(kernel_lab::reduce_sum({16777216.f, 1.f, -16777216.f}) == 1.0,
            "double accumulator lost cancellation residual");
    require(std::isinf(kernel_lab::reduce_sum({1.0f, INFINITY})),
            "positive infinity was not propagated");
    require(std::isnan(kernel_lab::reduce_sum({1.0f, NAN})),
            "NaN was not propagated");
    std::cout << "PASS: 19 reduction sizes through 1000003 and special values\n";
}
