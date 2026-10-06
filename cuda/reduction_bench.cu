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
constexpr unsigned kThreads = 256;
constexpr unsigned kMaxBlocks = 65535;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T> struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(std::size_t count) {
        if (count) check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    }
    ~Buffer() { if (ptr) cudaFree(ptr); }
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

template <typename Input>
__global__ void reduce_blocks(const Input* input, double* output, std::size_t n) {
    __shared__ double scratch[kThreads];
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2 + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x * 2;
    double sum = 0.0;
    for (std::size_t i = base; i < n; i += stride) {
        sum += static_cast<double>(input[i]);
        if (i + blockDim.x < n) sum += static_cast<double>(input[i + blockDim.x]);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[blockIdx.x] = scratch[0];
}

unsigned blocks_for(std::size_t n) {
    return static_cast<unsigned>(std::min<std::size_t>((n + 2 * kThreads - 1) / (2 * kThreads), kMaxBlocks));
}

struct Reducer {
    Buffer<double> first;
    Buffer<double> second;

    explicit Reducer(std::size_t n) : first(blocks_for(n)), second(blocks_for(n)) {}

    double* launch(const float* input, std::size_t n) {
        unsigned blocks = blocks_for(n);
        reduce_blocks<<<blocks, kThreads>>>(input, first.ptr, n);
        check(cudaGetLastError());
        double* source = first.ptr;
        double* destination = second.ptr;
        std::size_t count = blocks;
        while (count > 1) {
            blocks = blocks_for(count);
            reduce_blocks<<<blocks, kThreads>>>(source, destination, count);
            check(cudaGetLastError());
            count = blocks;
            std::swap(source, destination);
        }
        return source;
    }
};

void run(std::size_t n, int repeats, bool timing) {
    if (n == 0) {
        if (kernel_lab::reduce_sum({}) != 0.0) throw std::runtime_error("empty oracle failed");
        std::cout << "n=0 PASS (no device allocation/launch)\n";
        return;
    }
    std::vector<float> values(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int numerator = static_cast<int>((i * 37 + 11) % 257) - 128;
        values[i] = static_cast<float>(numerator) / 16.0f;
    }
    const double expected = kernel_lab::reduce_sum(values);
    Buffer<float> device_input(n);
    check(cudaMemcpy(device_input.ptr, values.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    Reducer reducer(n);
    double* result = nullptr;
    for (int i = 0; i < 10; ++i) result = reducer.launch(device_input.ptr, n);
    check(cudaDeviceSynchronize());
    Event start, stop;
    check(cudaEventRecord(start.value));
    for (int i = 0; i < repeats; ++i) result = reducer.launch(device_input.ptr, n);
    check(cudaEventRecord(stop.value));
    check(cudaEventSynchronize(stop.value));
    float elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, start.value, stop.value));
    double observed = 0.0;
    check(cudaMemcpy(&observed, result, sizeof(observed), cudaMemcpyDeviceToHost));
    if (!std::isfinite(observed) || observed != expected)
        throw std::runtime_error("reduction mismatch: expected " + std::to_string(expected) +
                                 ", observed " + std::to_string(observed));
    std::cout << "n=" << n << " PASS sum=" << observed;
    if (timing) {
        std::cout << " warmup=10 repeats=" << repeats
                  << " reduction_ms=" << elapsed_ms / repeats;
    }
    std::cout << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 3) throw std::invalid_argument("usage: reduction_bench [size] [repeats]");
        auto parse = [](const char* text) {
            const std::string value(text);
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("arguments must be unsigned decimal integers");
            return std::stoull(value);
        };
        const auto n = argc > 1 ? parse(argv[1]) : 1048576ULL;
        const auto repeats = argc > 2 ? parse(argv[2]) : 100ULL;
        if (n > 16777216ULL || repeats == 0 || repeats > 10000ULL)
            throw std::invalid_argument("size <= 16777216 and repeats in [1,10000] required");
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
                  << " scope=all-reduction-kernels block=" << kThreads << " accumulation=double\n";
        for (std::size_t edge : {0u, 1u, 2u, 31u, 32u, 33u, 255u, 256u, 257u,
                                 511u, 512u, 513u, 1023u, 1024u, 1025u, 65537u})
            run(edge, 1, false);
        run(static_cast<std::size_t>(n), static_cast<int>(repeats), true);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
