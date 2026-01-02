#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

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

    // V2: Full GPU preprocess (project + ellipse params) then render into device RGBA8.
    // Uses scene-static buffers uploaded via uploadSceneData().
    bool render_cuda_to_rgba8_device_v2(
        const glm::mat4& view,
        const glm::mat4& proj,
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
    float* d_radii_px_ = nullptr;       // Conservative radius in pixels [N]
    float* d_depths_ = nullptr;         // Depth values for sorting [N]
    float* d_output_ = nullptr;         // Output image [W*H*3]

    // Scene-static data (uploaded once, kept resident across frames)
    float* d_pos_ws_ = nullptr;         // World positions [N*3]: (x,y,z) interleaved
    float* d_scale_ = nullptr;          // World-space axis scales [N*3]
    float* d_rotation_ = nullptr;       // Rotation quaternion [N*4] as (w,x,y,z)
    float* d_opacity_scene_ = nullptr;  // Opacity [N]
    float* d_sh_coeffs_ = nullptr;      // SH coefficients [N*27]: (R_0..8, G_0..8, B_0..8) - Degree 2
    float* d_dc_colors_ = nullptr;      // DC (0-order SH) colors [N*3]
    size_t scene_splat_capacity_ = 0;   // Capacity for scene data
    size_t scene_num_splats_ = 0;       // Number of splats currently loaded

    // Cached tile buffers (capacity-managed)
    int* d_tile_splat_list_ = nullptr;  // flat list of splat indices (sorted)
    int* d_tile_offsets_ = nullptr;     // tile offsets (num_tiles+1)
    size_t tile_splat_list_capacity_ = 0;
    size_t tile_offsets_capacity_ = 0;

    // GPU tile binning & sorting buffers (capacity-managed)
    int* d_num_tiles_touched_ = nullptr;     // [N] count of tiles per splat
    int* d_dup_offsets_ = nullptr;           // [N+1] exclusive scan of num_tiles_touched
    uint64_t* d_keys_ = nullptr;             // [total_dup] sort keys (tile_id, depth)
    uint64_t* d_keys_sorted_ = nullptr;      // [total_dup] sorted keys
    int* d_values_ = nullptr;                // [total_dup] splat indices
    int* d_values_sorted_ = nullptr;         // [total_dup] sorted splat indices
    uint32_t* d_tile_ids_sorted_ = nullptr;  // [total_dup] tile ids extracted from sorted keys
    uint32_t* d_unique_tile_ids_ = nullptr;  // [num_runs] unique tile ids after RLE
    int* d_run_lengths_ = nullptr;           // [num_runs] run lengths from RLE
    int* d_run_offsets_ = nullptr;           // [num_runs] exclusive scan of run lengths
    int* d_num_runs_device_ = nullptr;       // single int on device holding num_runs
    void* d_cub_temp_ = nullptr;             // CUB temporary storage
    size_t sort_buffer_capacity_ = 0;        // capacity for sort arrays
    size_t run_buffer_capacity_ = 0;         // capacity for RLE buffers (>= total_dup)
    size_t per_splat_capacity_ = 0;          // capacity for per-splat count/offset buffers
    size_t cub_temp_bytes_ = 0;              // CUB temp storage size

    // Per-frame buffers (capacity-managed device)
    int* d_gaussian_ids_ = nullptr;     // gaussian index for each screen splat [N]
    size_t frame_buffer_capacity_ = 0;  // capacity of d_means2D_, d_conic3D_, d_colors_, d_opacities_, d_gaussian_ids_

    // Persistent host buffers (reused each frame, no realloc)
    std::vector<float> h_means2D_;       // [N*2] screen positions
    std::vector<float> h_conic3D_;       // [N*3] conic matrices
    std::vector<float> h_opacities_;     // [N] opacity values
    std::vector<float> h_radii_px_;      // [N] radius values
    std::vector<float> h_depths_;        // [N] depth values
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
    bool ensureTileBuffers(size_t splat_list_count, size_t offsets_count);
    bool ensureSortBuffers(int num_splats, size_t total_duplicates); // GPU sort + RLE buffers
    bool buildTileBinning(int num_splats, int num_tiles_x, int num_tiles_y, int num_tiles, size_t& total_duplicates);
    bool debugValidateTileOffsets(int num_tiles, size_t total_duplicates);
};

} // namespace CudaRasterizer
