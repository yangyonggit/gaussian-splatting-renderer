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

    /**
     * Upload scene-static SH coefficients, positions, and opacities to GPU.
     * Call once after loading PLY. Allocates persistent device buffers.
     * 
     * @param gaussians Vector of GaussianSplat objects
     * @return true on success
     */
    bool uploadSceneData(const std::vector<gs::GaussianSplat>& gaussians);

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
    float* d_means2D_ = nullptr;        // Screen positions [N*2]: (sx, sy)
    float* d_conic3D_ = nullptr;        // 2D conic matrices [N*3]: (a, b, c) where conic = [a b; b c]
    float* d_colors_ = nullptr;         // RGB colors [N*3]
    float* d_opacities_ = nullptr;      // Opacity values [N]
    float* d_output_ = nullptr;         // Output image [W*H*3]

    // Scene-static data (uploaded once, kept resident across frames)
    float* d_pos_ws_ = nullptr;         // World positions [N*3]: (x,y,z) interleaved
    float* d_sh_coeffs_ = nullptr;      // SH coefficients [N*27]: (R_0..8, G_0..8, B_0..8) - Degree 2
    float* d_dc_colors_ = nullptr;      // DC (0-order SH) colors [N*3]
    size_t scene_splat_capacity_ = 0;   // Capacity for scene data
    size_t scene_num_splats_ = 0;       // Number of splats currently loaded

    // Cached tile buffers (capacity-managed)
    int* d_tile_splat_list_ = nullptr; // flat list of indices
    int* d_tile_offsets_ = nullptr;    // tile offsets (num_tiles+1)
    size_t tile_splat_list_capacity_ = 0; // number of ints allocated
    size_t tile_offsets_capacity_ = 0;     // number of ints allocated

    // Per-frame buffers (capacity-managed device)
    int* d_gaussian_ids_ = nullptr;     // gaussian index for each screen splat [N]
    size_t frame_buffer_capacity_ = 0;  // capacity of d_means2D_, d_conic3D_, d_colors_, d_opacities_, d_gaussian_ids_

    // Persistent host buffers (reused each frame, no realloc)
    std::vector<float> h_means2D_;       // [N*2] screen positions
    std::vector<float> h_conic3D_;       // [N*3] conic matrices
    std::vector<float> h_opacities_;     // [N] opacity values
    std::vector<int> h_gaussian_ids_;    // [N] gaussian indices
    size_t host_buffer_capacity_ = 0;    // tracked capacity to avoid reallocs

    int num_splats_;
    int width_, height_;

    // Scene reference (saved from uploadSceneData)
    const std::vector<gs::GaussianSplat>* gaussians_ptr_ = nullptr;

    // Internal memory management
    bool ensureFrameBuffers(int num_splats, int width, int height);
    void freeFrameBuffers();
    bool ensureHostBuffers(int num_splats);  // Reuse persistent host vectors
    
    // Ensure tile buffers have enough capacity; realloc only when needed
    bool ensureTileBuffers(size_t splat_list_count, size_t offsets_count);
};

} // namespace CudaRasterizer
