#include "kernel_lab/reference.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned kTile = 16;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Buffer {
    float* data = nullptr;
    explicit Buffer(std::size_t count) {
        if (count) check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Event {
    cudaEvent_t value{};
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

__global__ void tiled_matmul(const float* lhs, const float* rhs, float* output,
                             std::size_t m, std::size_t k, std::size_t n) {
    __shared__ float lhs_tile[kTile][kTile];
    __shared__ float rhs_tile[kTile][kTile];
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * kTile + threadIdx.y;
    const std::size_t column = static_cast<std::size_t>(blockIdx.x) * kTile + threadIdx.x;
    float sum = 0.0f;
    const std::size_t tile_count = (k + kTile - 1) / kTile;
    for (std::size_t tile = 0; tile < tile_count; ++tile) {
        const std::size_t lhs_column = tile * kTile + threadIdx.x;
        const std::size_t rhs_row = tile * kTile + threadIdx.y;
        lhs_tile[threadIdx.y][threadIdx.x] =
            row < m && lhs_column < k ? lhs[row * k + lhs_column] : 0.0f;
        rhs_tile[threadIdx.y][threadIdx.x] =
            rhs_row < k && column < n ? rhs[rhs_row * n + column] : 0.0f;
        __syncthreads();
        for (unsigned inner = 0; inner < kTile; ++inner)
            sum += lhs_tile[threadIdx.y][inner] * rhs_tile[inner][threadIdx.x];
        __syncthreads();
    }
    if (row < m && column < n) output[row * n + column] = sum;
}

void fill_inputs(std::vector<float>& lhs, std::vector<float>& rhs) {
    for (std::size_t i = 0; i < lhs.size(); ++i)
        lhs[i] = static_cast<float>(static_cast<int>((i * 7 + 3) % 17) - 8) / 4.0f;
    for (std::size_t i = 0; i < rhs.size(); ++i)
        rhs[i] = static_cast<float>(static_cast<int>((i * 5 + 1) % 13) - 6) / 8.0f;
}

void run(std::size_t m, std::size_t k, std::size_t n, int repeats, bool timing) {
    const std::size_t lhs_count = kernel_lab::checked_matrix_size(m, k);
    const std::size_t rhs_count = kernel_lab::checked_matrix_size(k, n);
    const std::size_t output_count = kernel_lab::checked_matrix_size(m, n);
    std::vector<float> lhs(lhs_count), rhs(rhs_count);
    fill_inputs(lhs, rhs);
    const auto expected = kernel_lab::matrix_multiply(lhs, rhs, m, k, n);
    if (m == 0 || n == 0 || k == 0) {
        if (expected != std::vector<float>(output_count, 0.0f))
            throw std::runtime_error("empty-dimension oracle failed");
        std::cout << "shape=" << m << 'x' << k << " * " << k << 'x' << n
                  << " PASS (no device allocation/launch)\n";
        return;
    }

    Buffer device_lhs(lhs_count), device_rhs(rhs_count), device_output(output_count);
    check(cudaMemcpy(device_lhs.data, lhs.data(), lhs_count * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_rhs.data, rhs.data(), rhs_count * sizeof(float), cudaMemcpyHostToDevice));
    const dim3 block(kTile, kTile);
    const dim3 grid(static_cast<unsigned>((n + kTile - 1) / kTile),
                    static_cast<unsigned>((m + kTile - 1) / kTile));
    auto launch = [&] {
        tiled_matmul<<<grid, block>>>(device_lhs.data, device_rhs.data, device_output.data, m, k, n);
        check(cudaGetLastError());
    };

    float elapsed_ms = 0.0f;
    if (timing) {
        for (int i = 0; i < 10; ++i) launch();
        check(cudaDeviceSynchronize());
        Event start, stop;
        check(cudaEventRecord(start.value));
        for (int i = 0; i < repeats; ++i) launch();
        check(cudaEventRecord(stop.value));
        check(cudaEventSynchronize(stop.value));
        check(cudaEventElapsedTime(&elapsed_ms, start.value, stop.value));
    } else {
        launch();
        check(cudaDeviceSynchronize());
    }

    std::vector<float> observed(output_count);
    check(cudaMemcpy(observed.data(), device_output.data, output_count * sizeof(float),
                     cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < output_count; ++i) {
        const float tolerance = 2.0e-5f * std::max(1.0f, std::fabs(expected[i])) +
                                2.0e-6f * static_cast<float>(k);
        if (!std::isfinite(observed[i]) || std::fabs(observed[i] - expected[i]) > tolerance)
            throw std::runtime_error("matrix mismatch at output index " + std::to_string(i));
    }
    std::cout << "shape=" << m << 'x' << k << " * " << k << 'x' << n << " PASS";
    if (timing) {
        const double mean_ms = elapsed_ms / repeats;
        const double operations = 2.0 * static_cast<double>(m) * k * n;
        std::cout << " warmup=10 repeats=" << repeats << " kernel_ms=" << mean_ms;
        if (mean_ms > 0.0) std::cout << " arithmetic_GFLOP_s=" << operations / (mean_ms * 1.0e6);
    }
    std::cout << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 5) throw std::invalid_argument("usage: tiled_matmul_bench [m] [k] [n] [repeats]");
        auto parse = [](const char* text) {
            const std::string value(text);
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("arguments must be unsigned decimal integers");
            return std::stoull(value);
        };
        const auto m = argc > 1 ? parse(argv[1]) : 512ULL;
        const auto k = argc > 2 ? parse(argv[2]) : 512ULL;
        const auto n = argc > 3 ? parse(argv[3]) : 512ULL;
        const auto repeats = argc > 4 ? parse(argv[4]) : 100ULL;
        if (m > 8192 || k > 8192 || n > 8192 || repeats == 0 || repeats > 10000)
            throw std::invalid_argument("dimensions <= 8192 and repeats in [1,10000] required");
        const auto lhs_count = kernel_lab::checked_matrix_size(m, k);
        const auto rhs_count = kernel_lab::checked_matrix_size(k, n);
        const auto output_count = kernel_lab::checked_matrix_size(m, n);
        if (lhs_count + rhs_count + output_count > 67108864ULL)
            throw std::invalid_argument("total matrix elements must not exceed 67108864");

        check(cudaSetDevice(0));
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0));
        int driver = 0, runtime = 0;
        check(cudaDriverGetVersion(&driver));
        check(cudaRuntimeGetVersion(&runtime));
        std::cout << "GPU=" << properties.name << " cc=" << properties.major << '.' << properties.minor
                  << " SMs=" << properties.multiProcessorCount << " memory_bytes=" << properties.totalGlobalMem
                  << " driver=" << driver << " runtime=" << runtime
                  << " nvcc=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__
                  << " scope=kernel-only tile=" << kTile << 'x' << kTile << '\n';
        run(0, 0, 0, 1, false);
        run(1, 1, 1, 1, false);
        run(3, 5, 2, 1, false);
        run(15, 15, 15, 1, false);
        run(16, 16, 16, 1, false);
        run(17, 17, 17, 1, false);
        run(31, 7, 33, 1, false);
        run(static_cast<std::size_t>(m), static_cast<std::size_t>(k),
            static_cast<std::size_t>(n), static_cast<int>(repeats), true);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
