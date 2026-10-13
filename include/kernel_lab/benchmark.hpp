#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernel_lab {

struct TimingSummary {
    double minimum_ms;
    double median_ms;
    double mean_ms;
    double maximum_ms;
};

inline TimingSummary summarize_milliseconds(std::vector<double> samples) {
    if (samples.empty()) throw std::invalid_argument("timing samples are empty");
    for (const double sample : samples) {
        if (!std::isfinite(sample) || sample < 0.0)
            throw std::invalid_argument("timing sample must be finite and non-negative");
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    const double median = samples.size() % 2 == 0
                              ? (samples[middle - 1] + samples[middle]) / 2.0
                              : samples[middle];
    const double mean =
        std::accumulate(samples.begin(), samples.end(), 0.0) /
        static_cast<double>(samples.size());
    return {samples.front(), median, mean, samples.back()};
}

inline std::string json_escape(const std::string& input) {
    std::string output;
    output.reserve(input.size());
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char character : input) {
        switch (character) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (character < 0x20) {
                output += "\\u00";
                output += hex[character >> 4];
                output += hex[character & 0x0f];
            } else {
                output += static_cast<char>(character);
            }
        }
    }
    return output;
}

inline const char* compiler_name() {
#if defined(__clang__)
    return "clang";
#elif defined(__GNUC__)
    return "gcc";
#elif defined(_MSC_VER)
    return "msvc";
#else
    return "unknown";
#endif
}

inline std::string compiler_version() {
#if defined(__clang_version__)
    return __clang_version__;
#elif defined(__VERSION__)
    return __VERSION__;
#elif defined(_MSC_FULL_VER)
    return std::to_string(_MSC_FULL_VER);
#else
    return "unknown";
#endif
}

} // namespace kernel_lab
