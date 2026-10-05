#include "kernel_lab/reference.hpp"
#include <iostream>
int main() {
    const auto out = kernel_lab::vector_add({1.f, 2.f, 3.f}, {4.f, 5.f, 6.f});
    for (float x : out) std::cout << x << ' ';
    std::cout << "\nsum = " << kernel_lab::reduce_sum(out) << '\n';
}
