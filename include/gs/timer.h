#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace gs {

class Timer {
public:
    explicit Timer(const std::string& name)
        : name_(name), start_(std::chrono::steady_clock::now()) {}

    ~Timer() {
        const auto end = std::chrono::steady_clock::now();
        const auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << "[Timer] " << name_ << " took " << durationMs << " ms" << std::endl;
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
};

} // namespace gs
