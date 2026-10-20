#pragma once

#include "kernel_lab/reference.hpp"

namespace kernel_lab {

// Successful calls neither allocate nor resize. Exact input/output aliasing is
// safe for elementwise addition; owning std::vectors cannot partially overlap.
inline void vector_add_into(const std::vector<float>& lhs,
                            const std::vector<float>& rhs,
                            std::vector<float>& output) {
    if (lhs.size() != rhs.size() || output.size() != lhs.size())
        throw std::invalid_argument("vector storage sizes differ");
    for (std::size_t index = 0; index < lhs.size(); ++index)
        output[index] = lhs[index] + rhs[index];
}

// Row-major MxK by KxN, double accumulation. Reject aliasing and validate every
// dimension/storage constraint before writing, including for empty matrices.
inline void matrix_multiply_into(const std::vector<float>& lhs,
                                 const std::vector<float>& rhs,
                                 std::vector<float>& output,
                                 std::size_t m, std::size_t k, std::size_t n) {
    const auto lhs_size = checked_matrix_size(m, k);
    const auto rhs_size = checked_matrix_size(k, n);
    const auto output_size = checked_matrix_size(m, n);
    if (lhs.size() != lhs_size || rhs.size() != rhs_size ||
        output.size() != output_size)
        throw std::invalid_argument("matrix storage does not match dimensions");
    if (&output == &lhs || &output == &rhs)
        throw std::invalid_argument("matrix output aliases input");
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < k; ++inner)
                sum += static_cast<double>(lhs[row * k + inner]) *
                       static_cast<double>(rhs[inner * n + column]);
            output[row * n + column] = static_cast<float>(sum);
        }
    }
}

} // namespace kernel_lab
