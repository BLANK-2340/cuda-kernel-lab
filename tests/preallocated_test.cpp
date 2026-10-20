#include "kernel_lab/preallocated.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <new>

namespace {
bool count_allocations = false;
std::size_t allocations = 0;
}

// Successful API paths must not allocate, even transiently. This binary uses
// ordinary float vectors only (no over-aligned allocations).
#if defined(__GNUC__) || defined(__clang__)
// Keep malloc/free-backed replacements opaque to optimizer allocation-pair
// diagnostics; these matched new/delete replacements deliberately use C storage.
#define ALLOCATION_PROBE_NOINLINE __attribute__((noinline))
#else
#define ALLOCATION_PROBE_NOINLINE
#endif
ALLOCATION_PROBE_NOINLINE
void* operator new(std::size_t size) {
    if (count_allocations) ++allocations;
    if (void* memory = std::malloc(size == 0 ? 1 : size)) return memory;
    throw std::bad_alloc();
}
ALLOCATION_PROBE_NOINLINE void* operator new[](std::size_t size) { return ::operator new(size); }
ALLOCATION_PROBE_NOINLINE void operator delete(void* memory) noexcept { std::free(memory); }
ALLOCATION_PROBE_NOINLINE void operator delete[](void* memory) noexcept { std::free(memory); }
ALLOCATION_PROBE_NOINLINE void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
ALLOCATION_PROBE_NOINLINE void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }
#undef ALLOCATION_PROBE_NOINLINE

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename Exception, typename Function>
void rejected(Function function) {
    bool caught = false;
    try { function(); } catch (const Exception&) { caught = true; }
    require(caught, "invalid call accepted");
}
template <typename Function>
void no_allocation(Function function) {
    allocations = 0;
    count_allocations = true;
    function();
    count_allocations = false;
    require(allocations == 0, "successful call allocated");
}
}

int main() {
    count_allocations = true;
    void* probe = ::operator new(sizeof(float));
    count_allocations = false;
    ::operator delete(probe);
    require(allocations == 1, "allocation probe did not observe control allocation");
    for (const std::size_t size : {0u, 1u, 15u, 16u, 17u, 255u, 256u, 257u, 65537u}) {
        std::vector<float> lhs(size, 1.25f), rhs(size, -0.5f), output(size, 99.0f);
        const auto* address = output.data();
        const auto capacity = output.capacity();
        no_allocation([&] { kernel_lab::vector_add_into(lhs, rhs, output); });
        require(output == std::vector<float>(size, 0.75f), "vector sum incorrect");
        require(output.data() == address && output.capacity() == capacity,
                "output storage changed");
        no_allocation([&] { kernel_lab::vector_add_into(lhs, rhs, lhs); });
        require(lhs == output, "left alias incorrect");
        no_allocation([&] { kernel_lab::vector_add_into(lhs, rhs, rhs); });
        require(rhs == std::vector<float>(size, 0.25f), "right alias incorrect");
        no_allocation([&] { kernel_lab::vector_add_into(lhs, lhs, lhs); });
        require(lhs == std::vector<float>(size, 1.5f), "all aliases incorrect");
    }

    // Rectangular products around tile boundaries; constants give an analytic
    // expected result independent of the existing matrix multiplication loop.
    for (const std::size_t m : {0u, 1u, 15u, 16u, 17u}) {
        for (const std::size_t k : {0u, 1u, 9u, 17u}) {
            const std::size_t n = 7;
            std::vector<float> lhs(m * k, 0.5f), rhs(k * n, -0.25f);
            std::vector<float> output(m * n, 99.0f);
            no_allocation([&] { kernel_lab::matrix_multiply_into(lhs, rhs, output, m, k, n); });
            require(output == std::vector<float>(m * n, -0.125f * static_cast<float>(k)),
                    "rectangular matrix product incorrect");
            std::fill(output.begin(), output.end(), 55.0f);
            no_allocation([&] { kernel_lab::matrix_multiply_into(lhs, rhs, output, m, k, n); });
            require(output == kernel_lab::matrix_multiply(lhs, rhs, m, k, n),
                    "reused output differs from oracle");
        }
    }
    std::vector<float> lhs{1, 2, 3, 4, 5, 6}, rhs{7, 8, 9, 10, 11, 12}, output(4);
    no_allocation([&] { kernel_lab::matrix_multiply_into(lhs, rhs, output, 2, 3, 2); });
    require(output == std::vector<float>({58, 64, 139, 154}), "known product incorrect");
    std::vector<float> empty, empty_output;
    no_allocation([&] { kernel_lab::matrix_multiply_into(lhs, empty, empty_output, 2, 3, 0); });
    rejected<std::invalid_argument>([&] {
        kernel_lab::matrix_multiply_into(lhs, empty, empty, 2, 3, 0);
    });
    const auto saved = output;
    rejected<std::invalid_argument>([&] { kernel_lab::vector_add_into(lhs, empty, output); });
    rejected<std::invalid_argument>([&] { kernel_lab::vector_add_into(lhs, rhs, output); });
    rejected<std::invalid_argument>([&] {
        kernel_lab::matrix_multiply_into(lhs, rhs, output, 2, 2, 2);
    });
    rejected<std::invalid_argument>([&] {
        kernel_lab::matrix_multiply_into(lhs, rhs, output, 3, 2, 3);
    });
    rejected<std::length_error>([&] {
        kernel_lab::matrix_multiply_into(lhs, rhs, output,
            std::numeric_limits<std::size_t>::max(), 2, 1);
    });
    rejected<std::length_error>([&] {
        kernel_lab::matrix_multiply_into(empty, empty, empty_output, 0,
            std::numeric_limits<std::size_t>::max(), 2);
    });
    rejected<std::length_error>([&] {
        kernel_lab::matrix_multiply_into(empty, empty, empty_output,
            std::numeric_limits<std::size_t>::max(), 0, 2);
    });
    std::vector<float> wrong_rhs(5);
    rejected<std::invalid_argument>([&] {
        kernel_lab::matrix_multiply_into(lhs, wrong_rhs, output, 2, 3, 2);
    });
    require(output == saved, "invalid call modified output");
    std::vector<float> square{1, 2, 3, 4}, other{1, 0, 0, 1};
    const auto saved_square = square;
    rejected<std::invalid_argument>([&] {
        kernel_lab::matrix_multiply_into(square, other, square, 2, 2, 2);
    });
    rejected<std::invalid_argument>([&] {
        kernel_lab::matrix_multiply_into(other, square, square, 2, 2, 2);
    });
    require(square == saved_square, "alias rejection modified input");
    std::cout << "PASS: preallocated arithmetic, aliases, validation, reuse and zero allocations\n";
}
