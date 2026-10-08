#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace kernel_lab {
inline std::vector<float> vector_add(const std::vector<float>& a,
                                     const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("vector sizes differ");
    std::vector<float> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}
// Accumulate float inputs in double to provide a more accurate GPU oracle.
inline double reduce_sum(const std::vector<float>& a) {
    double sum = 0;
    for (float x : a) sum += static_cast<double>(x);
    return sum;
}

inline std::size_t checked_matrix_size(std::size_t rows, std::size_t columns) {
    if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns)
        throw std::length_error("matrix dimensions overflow");
    return rows * columns;
}

// Row-major MxK by KxN reference. Double accumulation makes this a useful
// oracle for float GPU kernels without claiming exact real-number arithmetic.
inline std::vector<float> matrix_multiply(const std::vector<float>& lhs,
                                          const std::vector<float>& rhs,
                                          std::size_t m,
                                          std::size_t k,
                                          std::size_t n) {
    const std::size_t lhs_size = checked_matrix_size(m, k);
    const std::size_t rhs_size = checked_matrix_size(k, n);
    const std::size_t output_size = checked_matrix_size(m, n);
    if (lhs.size() != lhs_size || rhs.size() != rhs_size)
        throw std::invalid_argument("matrix storage does not match dimensions");
    std::vector<float> output(output_size, 0.0f);
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < k; ++inner)
                sum += static_cast<double>(lhs[row * k + inner]) *
                       static_cast<double>(rhs[inner * n + column]);
            output[row * n + column] = static_cast<float>(sum);
        }
    }
    return output;
}
} // namespace kernel_lab
