#pragma once

#include <vector>

// Forward declaration to avoid including gaussian.h in header
namespace gs {
    struct GaussianSplat;
}

namespace CudaRasterizer {

/**
 * CUDA-based Gaussian Splatting Rasterizer
 * Manages GPU memory and data transfer for Gaussian splats
 */
class Rasterizer {
public:
    Rasterizer();
    ~Rasterizer();

    /**
     * Allocate GPU memory for the specified number of Gaussian splats
     * @param num_points Number of Gaussian splats to allocate memory for
     */
    void allocate(int num_points);

    /**
     * Upload Gaussian splat data to GPU
     * @param positions Flat array of 3D positions (size: num_points * 3)
     * @param scales Flat array of 3D scales (size: num_points * 3)
     * @param rotations Flat array of quaternions (size: num_points * 4)
     * @param opacities Array of opacity values (size: num_points)
     * @param sh_coeffs Flat array of SH coefficients (size: num_points * sh_dim)
     * @param num_points Number of Gaussian splats
     * @param sh_dim Number of SH coefficients per point (typically 48 for degree 3)
     */
    void uploadData(
        const float* positions,
        const float* scales,
        const float* rotations,
        const float* opacities,
        const float* sh_coeffs,
        int num_points,
        int sh_dim
    );

    /**
     * Load and upload Gaussian splats from vector (high-level interface)
     * Automatically handles SoA conversion and GPU upload
     * @param splats Vector of GaussianSplat structures
     */
    void loadFromSplats(const std::vector<gs::GaussianSplat>& splats);

    /**
     * Render the Gaussian splats (placeholder for future implementation)
     */
    void render();

    /**
     * Free all GPU memory
     */
    void free();

private:
    int num_points_;
    int sh_dim_;

    // GPU memory pointers
    float* d_pos_;      // 3D positions (num_points * 3)
    float* d_scale_;    // 3D scales (num_points * 3)
    float* d_rot_;      // Quaternions (num_points * 4)
    float* d_opacity_;  // Opacities (num_points)
    float* d_sh_;       // SH coefficients (num_points * sh_dim)
};

} // namespace CudaRasterizer
