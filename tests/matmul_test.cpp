#include "kernel_lab/reference.hpp"
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void exercise(std::size_t m, std::size_t k, std::size_t n) {
    std::vector<float> lhs(m * k);
    std::vector<float> rhs(k * n);
    for (std::size_t i = 0; i < lhs.size(); ++i)
        lhs[i] = static_cast<float>(static_cast<int>((i * 7 + 3) % 17) - 8) / 4.0f;
    for (std::size_t i = 0; i < rhs.size(); ++i)
        rhs[i] = static_cast<float>(static_cast<int>((i * 5 + 1) % 13) - 6) / 8.0f;
    const auto result = kernel_lab::matrix_multiply(lhs, rhs, m, k, n);
    require(result.size() == m * n, "incorrect output shape");
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            double expected = 0.0;
            for (std::size_t inner = 0; inner < k; ++inner)
                expected += static_cast<double>(lhs[row * k + inner]) * rhs[inner * n + column];
            require(result[row * n + column] == static_cast<float>(expected),
                    "matrix product mismatch");
        }
    }
}
} // namespace

int main() {
    exercise(0, 0, 0);
    exercise(0, 17, 9);
    exercise(9, 0, 17);
    exercise(1, 1, 1);
    exercise(1, 17, 1);
    exercise(3, 5, 2);
    exercise(15, 15, 15);
    exercise(16, 16, 16);
    exercise(17, 17, 17);
    exercise(31, 7, 33);
    exercise(33, 32, 15);

    require(kernel_lab::matrix_multiply({1, 2, 3, 4}, {5, 6, 7, 8}, 2, 2, 2) ==
                std::vector<float>({19, 22, 43, 50}),
            "known 2x2 product mismatch");
    require(kernel_lab::matrix_multiply({}, {}, 3, 0, 2) ==
                std::vector<float>(6, 0.0f),
            "zero-inner-dimension product mismatch");

    bool rejected = false;
    try { (void)kernel_lab::matrix_multiply({1.0f}, {}, 1, 2, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "storage mismatch was accepted");

    rejected = false;
    try {
        (void)kernel_lab::checked_matrix_size(std::numeric_limits<std::size_t>::max(), 2);
    } catch (const std::length_error&) { rejected = true; }
    require(rejected, "dimension overflow was accepted");

    std::cout << "PASS: 11 matrix shapes, known product, zero inner dimension, invalid inputs\n";
}
