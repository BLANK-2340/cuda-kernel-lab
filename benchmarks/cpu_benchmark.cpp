#include "kernel_lab/benchmark.hpp"
#include "kernel_lab/reference.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::size_t warmup = 3;
    std::size_t iterations = 10;
    std::size_t vector_size = 1u << 20;
    std::size_t matrix_size = 96;
    std::string output_path;
};

volatile double benchmark_sink = 0.0;

std::size_t parse_size(const std::string& value,
                       const char* option,
                       std::size_t maximum,
                       bool allow_zero) {
    std::size_t parsed_characters = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &parsed_characters);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " requires an integer");
    }
    if (parsed_characters != value.size() ||
        parsed > static_cast<unsigned long long>(maximum) ||
        (!allow_zero && parsed == 0)) {
        throw std::invalid_argument(std::string(option) + " is outside the supported range");
    }
    return static_cast<std::size_t>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help") {
            std::cout
                << "Usage: cpu_benchmark [--warmup N] [--iterations N] "
                   "[--vector-size N] [--matrix-size N] [--output FILE]\n";
            std::exit(0);
        }
        if (index + 1 >= argc)
            throw std::invalid_argument("missing value for " + option);
        const std::string value = argv[++index];
        if (option == "--warmup")
            options.warmup = parse_size(value, "--warmup", 1000, true);
        else if (option == "--iterations")
            options.iterations = parse_size(value, "--iterations", 10000, false);
        else if (option == "--vector-size")
            options.vector_size =
                parse_size(value, "--vector-size", 16777216, true);
        else if (option == "--matrix-size")
            options.matrix_size = parse_size(value, "--matrix-size", 512, true);
        else if (option == "--output")
            options.output_path = value;
        else
            throw std::invalid_argument("unknown option: " + option);
    }
    return options;
}

template <typename Function>
kernel_lab::TimingSummary measure(std::size_t warmup,
                                  std::size_t iterations,
                                  Function function) {
    for (std::size_t index = 0; index < warmup; ++index)
        benchmark_sink = benchmark_sink + function();

    std::vector<double> samples;
    samples.reserve(iterations);
    for (std::size_t index = 0; index < iterations; ++index) {
        const auto start = std::chrono::steady_clock::now();
        const double checksum = function();
        const auto stop = std::chrono::steady_clock::now();
        benchmark_sink = benchmark_sink + checksum;
        samples.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
    }
    return kernel_lab::summarize_milliseconds(std::move(samples));
}

void require_vector_correct(const std::vector<float>& lhs,
                            const std::vector<float>& rhs,
                            const std::vector<float>& output) {
    if (output.size() != lhs.size())
        throw std::runtime_error("vector-add output size mismatch");
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (output[index] != lhs[index] + rhs[index])
            throw std::runtime_error("vector-add verification failed");
    }
}

void require_matrix_correct(const std::vector<float>& lhs,
                            const std::vector<float>& rhs,
                            const std::vector<float>& output,
                            std::size_t dimension) {
    if (output.size() != dimension * dimension)
        throw std::runtime_error("matrix output size mismatch");
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            double expected = 0.0;
            for (std::size_t inner = 0; inner < dimension; ++inner) {
                expected += static_cast<double>(lhs[row * dimension + inner]) *
                            static_cast<double>(rhs[inner * dimension + column]);
            }
            if (output[row * dimension + column] !=
                static_cast<float>(expected))
                throw std::runtime_error("matrix verification failed");
        }
    }
}

double vector_checksum(const std::vector<float>& values) {
    double checksum = 0.0;
    for (const float value : values) checksum += static_cast<double>(value);
    return checksum;
}

void write_summary(std::ostream& output,
                   const kernel_lab::TimingSummary& summary,
                   const char* indentation) {
    output << indentation << "\"minimum_ms\": " << summary.minimum_ms << ",\n"
           << indentation << "\"median_ms\": " << summary.median_ms << ",\n"
           << indentation << "\"mean_ms\": " << summary.mean_ms << ",\n"
           << indentation << "\"maximum_ms\": " << summary.maximum_ms;
}

std::string make_report(const Options& options,
                        const kernel_lab::TimingSummary& vector_add,
                        const kernel_lab::TimingSummary& reduction,
                        const kernel_lab::TimingSummary& matrix) {
    std::ostringstream output;
    output << std::setprecision(9);
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"runtime\": \"cpu\",\n"
           << "  \"correctness_verified\": true,\n"
           << "  \"metadata\": {\n"
           << "    \"compiler\": \"" << kernel_lab::compiler_name() << "\",\n"
           << "    \"compiler_version\": \""
           << kernel_lab::json_escape(kernel_lab::compiler_version()) << "\",\n"
           << "    \"cplusplus\": " << __cplusplus << ",\n"
           << "    \"hardware_threads\": "
           << std::thread::hardware_concurrency() << ",\n"
#ifdef NDEBUG
           << "    \"build_mode\": \"release\"\n"
#else
           << "    \"build_mode\": \"debug\"\n"
#endif
           << "  },\n"
           << "  \"methodology\": {\n"
           << "    \"timer\": \"std::chrono::steady_clock\",\n"
           << "    \"scope\": \"CPU function call only; input allocation and correctness verification excluded\",\n"
           << "    \"warmup_iterations\": " << options.warmup << ",\n"
           << "    \"measured_iterations\": " << options.iterations << "\n"
           << "  },\n"
           << "  \"benchmarks\": {\n"
           << "    \"vector_add\": {\n"
           << "      \"elements\": " << options.vector_size << ",\n";
    write_summary(output, vector_add, "      ");
    output << "\n    },\n"
           << "    \"reduction\": {\n"
           << "      \"elements\": " << options.vector_size << ",\n";
    write_summary(output, reduction, "      ");
    output << "\n    },\n"
           << "    \"matrix_multiply\": {\n"
           << "      \"rows\": " << options.matrix_size << ",\n"
           << "      \"inner\": " << options.matrix_size << ",\n"
           << "      \"columns\": " << options.matrix_size << ",\n";
    write_summary(output, matrix, "      ");
    output << "\n    }\n"
           << "  },\n"
           << "  \"limitations\": ["
              "\"Results describe this CPU process only\", "
              "\"GPU validation and measurements remain pending\""
           << "]\n"
           << "}\n";
    return output.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        std::vector<float> lhs(options.vector_size);
        std::vector<float> rhs(options.vector_size);
        for (std::size_t index = 0; index < options.vector_size; ++index) {
            lhs[index] =
                static_cast<float>(static_cast<int>((index * 17 + 3) % 257) - 128) /
                16.0f;
            rhs[index] =
                static_cast<float>(static_cast<int>((index * 29 + 5) % 251) - 125) /
                8.0f;
        }

        const auto vector_output = kernel_lab::vector_add(lhs, rhs);
        require_vector_correct(lhs, rhs, vector_output);
        double expected_sum = 0.0;
        for (const float value : lhs) expected_sum += static_cast<double>(value);
        if (kernel_lab::reduce_sum(lhs) != expected_sum)
            throw std::runtime_error("reduction verification failed");

        const std::size_t matrix_elements =
            kernel_lab::checked_matrix_size(options.matrix_size, options.matrix_size);
        std::vector<float> matrix_lhs(matrix_elements);
        std::vector<float> matrix_rhs(matrix_elements);
        for (std::size_t index = 0; index < matrix_elements; ++index) {
            matrix_lhs[index] =
                static_cast<float>(static_cast<int>((index * 7 + 1) % 31) - 15) /
                8.0f;
            matrix_rhs[index] =
                static_cast<float>(static_cast<int>((index * 11 + 2) % 29) - 14) /
                16.0f;
        }
        const auto matrix_output = kernel_lab::matrix_multiply(
            matrix_lhs, matrix_rhs, options.matrix_size, options.matrix_size,
            options.matrix_size);
        require_matrix_correct(
            matrix_lhs, matrix_rhs, matrix_output, options.matrix_size);

        const auto vector_timing =
            measure(options.warmup, options.iterations, [&] {
                return vector_checksum(kernel_lab::vector_add(lhs, rhs));
            });
        const auto reduction_timing =
            measure(options.warmup, options.iterations, [&] {
                return kernel_lab::reduce_sum(lhs);
            });
        const auto matrix_timing =
            measure(options.warmup, options.iterations, [&] {
                return vector_checksum(kernel_lab::matrix_multiply(
                    matrix_lhs, matrix_rhs, options.matrix_size,
                    options.matrix_size, options.matrix_size));
            });

        const std::string report = make_report(
            options, vector_timing, reduction_timing, matrix_timing);
        if (options.output_path.empty()) {
            std::cout << report;
        } else {
            std::ofstream output(options.output_path);
            if (!output)
                throw std::runtime_error("could not open output file");
            output << report;
            if (!output)
                throw std::runtime_error("could not write output file");
            std::cout << "wrote " << options.output_path << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
