#include "kernel_lab/reference.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
    float* ptr = nullptr;
    explicit Buffer(std::size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(float))); }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Event {
    cudaEvent_t event{};
    Event() { check(cudaEventCreate(&event)); }
    ~Event() { cudaEventDestroy(event); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
__global__ void add(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) out[i] = a[i] + b[i];
}
void run(std::size_t n, int repeats, bool timing) {
    if (n == 0) {
        if (!kernel_lab::vector_add({}, {}).empty()) throw std::runtime_error("empty oracle failed");
        std::cout << "n=0 PASS (no device allocation/launch)\n";
        return;
    }
    std::vector<float> a(n), b(n), out(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 8.f;
        b[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 4.f;
    }
    const auto expected = kernel_lab::vector_add(a, b);
    Buffer da(n), db(n), dc(n);
    const std::size_t bytes = n * sizeof(float);
    check(cudaMemcpy(da.ptr, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.ptr, b.data(), bytes, cudaMemcpyHostToDevice));
    const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 65535));
    auto launch = [&]() {
        add<<<blocks, 256>>>(da.ptr, db.ptr, dc.ptr, n);
        check(cudaGetLastError());
    };
    for (int i = 0; i < 10; ++i) launch();
    check(cudaDeviceSynchronize());
    Event start, stop;
    check(cudaEventRecord(start.event));
    for (int i = 0; i < repeats; ++i) launch();
    check(cudaEventRecord(stop.event));
    check(cudaEventSynchronize(stop.event));
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, start.event, stop.event));
    check(cudaMemcpy(out.data(), dc.ptr, bytes, cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(out[i]) || out[i] != expected[i])
            throw std::runtime_error("vector-add mismatch at " + std::to_string(i));
    std::cout << "n=" << n << " PASS";
    if (timing) {
        const double ms = elapsed / repeats;
        std::cout << " warmup=10 repeats=" << repeats << " kernel_ms=" << ms;
        if (ms > 0) std::cout << " effective_GB_s=" << (3.0 * bytes / (ms * 1e6));
    }
    std::cout << '\n';
}
int main(int argc, char** argv) {
    try {
        if (argc > 3) throw std::invalid_argument("usage: vector_add_bench [size] [repeats]");
        auto parse = [](const char* text) {
            const std::string s(text);
            if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("arguments must be unsigned decimal integers");
            return std::stoull(s);
        };
        const auto n = argc > 1 ? parse(argv[1]) : 1048576ULL;
        const auto repeats = argc > 2 ? parse(argv[2]) : 100ULL;
        if (n > 16777216ULL || repeats == 0 || repeats > 10000ULL)
            throw std::invalid_argument("size <= 16777216 and repeats in [1,10000] required");
        check(cudaSetDevice(0));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0));
        int driver = 0, runtime = 0;
        check(cudaDriverGetVersion(&driver));
        check(cudaRuntimeGetVersion(&runtime));
        std::cout << "GPU=" << prop.name << " cc=" << prop.major << '.' << prop.minor
                  << " SMs=" << prop.multiProcessorCount << " memory_bytes=" << prop.totalGlobalMem
                  << " driver=" << driver << " runtime=" << runtime
                  << " nvcc=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__
                  << " scope=kernel-only block=256\n";
        for (std::size_t edge : {0u,1u,31u,32u,33u,255u,256u,257u,1023u,1024u,1025u,65537u})
            run(edge, 1, false);
        run(static_cast<std::size_t>(n), static_cast<int>(repeats), true);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
