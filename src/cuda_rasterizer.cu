#include "cuda_rasterizer.h"
#include "gs/gaussian.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace CudaRasterizer {

Rasterizer::Rasterizer()
    : num_points_(0)
    , sh_dim_(0)
    , d_pos_(nullptr)
    , d_scale_(nullptr)
    , d_rot_(nullptr)
    , d_opacity_(nullptr)
    , d_sh_(nullptr)
{
}

Rasterizer::~Rasterizer() {
    free();
}

void Rasterizer::allocate(int num_points) {
    // Free existing memory if already allocated
    free();

    num_points_ = num_points;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_pos_, num_points * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale_, num_points * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rot_, num_points * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_opacity_, num_points * sizeof(float)));

    printf("✅ Allocated VRAM for %d splats\n", num_points);
}

void Rasterizer::uploadData(
    const float* positions,
    const float* scales,
    const float* rotations,
    const float* opacities,
    const float* sh_coeffs,
    int num_points,
    int sh_dim
) {
    if (num_points_ != num_points) {
        fprintf(stderr, "Error: uploadData called with num_points=%d, but allocated for %d\n",
                num_points, num_points_);
        return;
    }

    sh_dim_ = sh_dim;

    // Allocate SH memory if not already done
    if (d_sh_ == nullptr && sh_dim > 0) {
        CUDA_CHECK(cudaMalloc(&d_sh_, num_points * sh_dim * sizeof(float)));
    }

    // Upload data to GPU
    CUDA_CHECK(cudaMemcpy(d_pos_, positions, num_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_, scales, num_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rot_, rotations, num_points * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacity_, opacities, num_points * sizeof(float), cudaMemcpyHostToDevice));
    
    if (sh_dim > 0 && sh_coeffs != nullptr) {
        CUDA_CHECK(cudaMemcpy(d_sh_, sh_coeffs, num_points * sh_dim * sizeof(float), cudaMemcpyHostToDevice));
    }

    printf("✅ Data uploaded to GPU (%d splats, SH dim=%d)\n", num_points, sh_dim);
}

void Rasterizer::loadFromSplats(const std::vector<gs::GaussianSplat>& splats) {
    int num_points = static_cast<int>(splats.size());
    
    printf("🔄 Converting %d splats to SoA layout for GPU...\n", num_points);
    
    // Convert Structure of Arrays (SoA) for GPU upload
    std::vector<float> positions(num_points * 3);
    std::vector<float> scales(num_points * 3);
    std::vector<float> rotations(num_points * 4);
    std::vector<float> opacities(num_points);
    std::vector<float> sh_coeffs(num_points * 48); // Assume degree 3 SH (48 coeffs)

    for (int i = 0; i < num_points; ++i) {
        const auto& s = splats[i];
        
        // Positions (x, y, z)
        positions[i * 3 + 0] = s.position_ws.x;
        positions[i * 3 + 1] = s.position_ws.y;
        positions[i * 3 + 2] = s.position_ws.z;
        
        // Scales (sx, sy, sz)
        scales[i * 3 + 0] = s.scale.x;
        scales[i * 3 + 1] = s.scale.y;
        scales[i * 3 + 2] = s.scale.z;
        
        // Rotations (quaternion: w, x, y, z)
        rotations[i * 4 + 0] = s.rotation.w;
        rotations[i * 4 + 1] = s.rotation.x;
        rotations[i * 4 + 2] = s.rotation.y;
        rotations[i * 4 + 3] = s.rotation.z;
        
        // Opacity
        opacities[i] = s.opacity;
        
        // SH coefficients (DC + higher order)
        // DC color (3 channels)
        sh_coeffs[i * 48 + 0] = s.dc_color.r;
        sh_coeffs[i * 48 + 1] = s.dc_color.g;
        sh_coeffs[i * 48 + 2] = s.dc_color.b;
        
        // Higher-order SH coefficients
        for (size_t j = 0; j < s.sh_color.coeffs.size() && j < 45; ++j) {
            sh_coeffs[i * 48 + 3 + j] = s.sh_color.coeffs[j];
        }
    }

    // Allocate and upload
    allocate(num_points);
    uploadData(
        positions.data(),
        scales.data(),
        rotations.data(),
        opacities.data(),
        sh_coeffs.data(),
        num_points,
        48
    );
}

void Rasterizer::render() {
    // Placeholder for future rendering implementation
    printf("render() called - not yet implemented\n");
}

void Rasterizer::free() {
    if (d_pos_) {
        CUDA_CHECK(cudaFree(d_pos_));
        d_pos_ = nullptr;
    }
    if (d_scale_) {
        CUDA_CHECK(cudaFree(d_scale_));
        d_scale_ = nullptr;
    }
    if (d_rot_) {
        CUDA_CHECK(cudaFree(d_rot_));
        d_rot_ = nullptr;
    }
    if (d_opacity_) {
        CUDA_CHECK(cudaFree(d_opacity_));
        d_opacity_ = nullptr;
    }
    if (d_sh_) {
        CUDA_CHECK(cudaFree(d_sh_));
        d_sh_ = nullptr;
    }
    
    num_points_ = 0;
    sh_dim_ = 0;
}

} // namespace CudaRasterizer
