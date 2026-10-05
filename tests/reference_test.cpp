#include "kernel_lab/reference.hpp"
#include <iostream>
#include <stdexcept>

void require(bool ok) { if (!ok) throw std::runtime_error("check failed"); }
int main() {
    for (std::size_t n : {0u, 1u, 31u, 32u, 33u, 255u, 256u, 257u, 1023u, 1024u, 1025u, 65537u}) {
        std::vector<float> a(n), b(n);
        double expected_sum = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const int value = static_cast<int>(i % 17) - 8;
            a[i] = static_cast<float>(value);
            b[i] = static_cast<float>(3 - value);
            expected_sum += value;
        }
        const auto out = kernel_lab::vector_add(a, b);
        require(out.size() == n);
        for (float value : out) require(value == 3.0f);
        require(kernel_lab::reduce_sum(a) == expected_sum);
    }
    require(kernel_lab::reduce_sum({16777216.f, 1.f, -16777216.f}) == 1.0);
    require(kernel_lab::vector_add({0.25f, -0.5f}, {0.5f, 0.25f}) ==
            std::vector<float>({0.75f, -0.25f}));
    bool rejected = false;
    try { (void)kernel_lab::vector_add({1.f}, {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    std::cout << "PASS: 12 boundary sizes, cancellation, fractional inputs, size mismatch\n";
}
