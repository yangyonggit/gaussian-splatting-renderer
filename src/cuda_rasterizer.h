#pragma once

#include <glm/glm.hpp>
#include <vector>

// Forward declaration to avoid including gaussian.h in header
namespace gs {
    struct GaussianSplat;
    struct ScreenSplat;
}

namespace CudaRasterizer {

/**
 * CUDA V1: Minimal Baseline Rasterizer for Gaussian Splatting
 * 
 * Design Philosophy:
 * - NO tiling / binning
 * - NO shared memory optimizations
 * - Sorting done on CPU (back-to-front order)
 * - GPU only does per-pixel splatting with alpha blending
 * - Simple, correct, and easy to debug
 * 
 * Performance: O(W*H*N) - acceptable for debugging and small scenes
 */
class Rasterizer {
public:
    Rasterizer();
    ~Rasterizer();

    /**
     * V1 Render Function - Minimal CUDA Gaussian Splatting
     * 
     * Takes CPU-preprocessed screen-space data and renders to output image.
     * All projection and sorting is done on CPU before calling this function.
     * 
     * @param screen_splats Screen-space splats with 2D positions, conic matrices, colors
     * @param camera_pos Camera position for SH color evaluation
     * @param width Output image width
     * @param height Output image height
     * @param output_image RGB output buffer [W*H*3], allocated by caller
     * @return true on success, false on error
     */
    bool render_cuda(
        const std::vector<gs::ScreenSplat>& screen_splats,
        const glm::vec3& camera_pos,
        int width,
        int height,
        float* output_image
    );

    // Render directly into a device RGBA8 buffer (e.g., mapped GL PBO)
    // out_rgba8_device must point to width*height*4 bytes allocated on device.
    bool render_cuda_to_rgba8_device(
        const std::vector<gs::ScreenSplat>& screen_splats,
        const glm::vec3& camera_pos,
        int width,
        int height,
        unsigned char* out_rgba8_device
    );

    /**
     * Free all GPU memory
     */
    void free();

private:
    // Device memory pointers (allocated dynamically per render call)
    float* d_means2D_;      // Screen positions [N*2]: (sx, sy)
    float* d_conic3D_;      // 2D conic matrices [N*3]: (a, b, c) where conic = [a b; b c]
    float* d_colors_;       // RGB colors [N*3]
    float* d_opacities_;    // Opacity values [N]
    float* d_output_;       // Output image [W*H*3]

    // Cached tile buffers (capacity-managed)
    int* d_tile_splat_list_ = nullptr; // flat list of indices
    int* d_tile_offsets_ = nullptr;    // tile offsets (num_tiles+1)
    size_t tile_splat_list_capacity_ = 0; // number of ints allocated
    size_t tile_offsets_capacity_ = 0;     // number of ints allocated

    int num_splats_;
    int width_, height_;

    // Internal memory management
    bool allocateBuffers(int num_splats, int width, int height);
    void freeBuffers();
    
    // Ensure tile buffers have enough capacity; realloc only when needed
    bool ensureTileBuffers(size_t splat_list_count, size_t offsets_count);
};

} // namespace CudaRasterizer
