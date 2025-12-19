#pragma once

// Lightweight profiling helpers. Toggle with ENABLE_PROFILING.
#ifndef ENABLE_PROFILING
#define ENABLE_PROFILING 0
#endif

#if ENABLE_PROFILING

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

class ScopedTimer {
public:
    ScopedTimer(const char* label, double* accumulator_ms = nullptr)
        : label_(label), accumulator_ms_(accumulator_ms), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        using namespace std::chrono;
        auto end = steady_clock::now();
        double ms = duration_cast<duration<double, std::milli>>(end - start_).count();
        if (accumulator_ms_) {
            *accumulator_ms_ += ms;
        }
        std::printf("[Profile] %s: %.3f ms\n", label_, ms);
    }

private:
    const char* label_;
    double* accumulator_ms_;
    std::chrono::steady_clock::time_point start_;
};

class CudaTimer {
public:
    CudaTimer(const char* label, float* accumulator_ms = nullptr)
        : label_(label), accumulator_ms_(accumulator_ms), start_(nullptr), stop_(nullptr), last_ms_(0.0f) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        cudaEventRecord(start_);
    }

    void stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&last_ms_, start_, stop_);
        if (accumulator_ms_) {
            *accumulator_ms_ += last_ms_;
        }
        std::printf("[Profile][GPU] %s: %.3f ms\n", label_, last_ms_);
    }

    float elapsedMs() const { return last_ms_; }

private:
    const char* label_;
    float* accumulator_ms_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
    float last_ms_;
};

#else // ENABLE_PROFILING

struct ScopedTimer {
    ScopedTimer(const char*, double* = nullptr) {}
};

struct CudaTimer {
    CudaTimer(const char*, float* = nullptr) {}
    void start() {}
    void stop() {}
    float elapsedMs() const { return 0.0f; }
};

#endif // ENABLE_PROFILING
