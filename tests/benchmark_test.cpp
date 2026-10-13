#include "kernel_lab/benchmark.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_invalid(Function function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}
} // namespace

int main() {
    const auto odd = kernel_lab::summarize_milliseconds({3.0, 1.0, 2.0});
    require(odd.minimum_ms == 1.0, "odd minimum mismatch");
    require(odd.median_ms == 2.0, "odd median mismatch");
    require(odd.mean_ms == 2.0, "odd mean mismatch");
    require(odd.maximum_ms == 3.0, "odd maximum mismatch");

    const auto even = kernel_lab::summarize_milliseconds({8.0, 2.0, 4.0, 6.0});
    require(even.minimum_ms == 2.0, "even minimum mismatch");
    require(even.median_ms == 5.0, "even median mismatch");
    require(even.mean_ms == 5.0, "even mean mismatch");
    require(even.maximum_ms == 8.0, "even maximum mismatch");

    require_invalid(
        [] { (void)kernel_lab::summarize_milliseconds({}); },
        "empty sample set was accepted");
    require_invalid(
        [] { (void)kernel_lab::summarize_milliseconds({-1.0}); },
        "negative sample was accepted");
    require_invalid(
        [] { (void)kernel_lab::summarize_milliseconds({INFINITY}); },
        "infinite sample was accepted");
    require_invalid(
        [] { (void)kernel_lab::summarize_milliseconds({NAN}); },
        "NaN sample was accepted");

    const std::string input = std::string("quote\" slash\\ newline\n tab\t") +
                              static_cast<char>(1);
    require(kernel_lab::json_escape(input) ==
                "quote\\\" slash\\\\ newline\\n tab\\t\\u0001",
            "JSON escaping mismatch");
    require(std::string(kernel_lab::compiler_name()).size() > 0,
            "compiler name is empty");
    require(!kernel_lab::compiler_version().empty(), "compiler version is empty");

    std::cout << "PASS: timing statistics, invalid samples, JSON escaping, metadata\n";
}
