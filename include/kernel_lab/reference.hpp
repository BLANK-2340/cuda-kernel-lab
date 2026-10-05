#pragma once
#include <cstddef>
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
} // namespace kernel_lab
