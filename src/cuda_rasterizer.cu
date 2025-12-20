#include "cuda_rasterizer.h"
#include "gs/gaussian.h"
#include "gs/screen_splat.h"
#include "gs/sh_color.h"
#include "gs/profiler.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstring>
#include <cmath>
#include <limits>
#include <glm/glm.hpp>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>

// CUDA error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return false; \
        } \
    } while (0)

#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#if defined(ENABLE_PROFILING) || !defined(NDEBUG)
#define CUDA_DEBUG_SYNC() CUDA_CHECK(cudaDeviceSynchronize())
#else
#define CUDA_DEBUG_SYNC()
#endif

namespace CudaRasterizer {

// ============================================================
// Utility Functions: Float-to-Ordered-Uint Conversion
// ============================================================

// Device/host float to ordered uint32 (radix-sortable, IEEE order preserved)
__device__ inline uint32_t floatToOrderedUintDevice(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
}

__host__ inline uint32_t floatToOrderedUintHost(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(uint32_t));
    return (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
}

__device__ __host__ inline uint32_t floatToOrderedUint(float f) {
#ifdef __CUDA_ARCH__
    return floatToOrderedUintDevice(f);
#else
    return floatToOrderedUintHost(f);
#endif
}

__device__ __host__ inline float sanitizeDepth(float depth) {
#ifdef __CUDA_ARCH__
    return isnan(depth) ? __int_as_float(0x7f800000) : depth; // +inf on device
#else
    return std::isnan(depth) ? std::numeric_limits<float>::infinity() : depth;
#endif
}

/**
 * Pack (tile_id, depth_descending) into 64-bit sort key.
 * Depth is assumed to be view-space or camera-space depth where larger means farther.
 * NaN is pushed to the far plane to keep keys ordered.
 */
__device__ inline uint64_t packTileDepthKey(uint32_t tile_id, float depth) {
    float depth_sanitized = sanitizeDepth(depth);
    uint32_t depth_ordered = floatToOrderedUint(depth_sanitized);
    uint32_t depth_desc = 0xFFFFFFFFu - depth_ordered;  // Descending = farther first
    return (uint64_t(tile_id) << 32) | uint64_t(depth_desc);
}

// ============================================================
// CUDA Kernels: GPU Tile Binning & Sorting
// ============================================================

/**
 * Phase 1: Compute how many tiles each splat touches
 * Output: num_tiles_touched[idx] = count of tiles covered by splat idx
 */
__global__ void computeTileTouchCountKernel(
    const float* __restrict__ means2D,      // [N*2] screen positions
    const float* __restrict__ radii_px,     // [N] conservative radius
    int num_splats,
    int num_tiles_x,
    int num_tiles_y,
    int* __restrict__ num_tiles_touched     // [N] output counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_splats) return;

    float sx = means2D[idx * 2 + 0];
    float sy = means2D[idx * 2 + 1];
    float radius = radii_px[idx];

    // Compute tile bounding box
    int tile_x_min = max(0, (int)((sx - radius) / 16.0f));
    int tile_x_max = min(num_tiles_x - 1, (int)((sx + radius) / 16.0f));
    int tile_y_min = max(0, (int)((sy - radius) / 16.0f));
    int tile_y_max = min(num_tiles_y - 1, (int)((sy + radius) / 16.0f));

    // Clamp to valid range (safety)
    if (tile_x_max < tile_x_min || tile_y_max < tile_y_min) {
        num_tiles_touched[idx] = 0;
        return;
    }

    int count = (tile_x_max - tile_x_min + 1) * (tile_y_max - tile_y_min + 1);
    num_tiles_touched[idx] = count;
}

/**
 * Phase 2: Emit duplicate keys/values for each splat-tile pair
 * Each splat writes (tile_id, depth) keys and (splat_idx) values
 * Output arrays are indexed by dup_offsets[idx] + local_offset
 */
__global__ void emitDuplicateKeysKernel(
    const float* __restrict__ means2D,
    const float* __restrict__ radii_px,
    const float* __restrict__ depths,       // [N] depth values for sorting
    int num_splats,
    int num_tiles_x,
    int num_tiles_y,
    const int* __restrict__ dup_offsets,    // [N] exclusive scan of num_tiles_touched
    uint64_t* __restrict__ out_keys,        // [total_duplicates]
    int* __restrict__ out_values            // [total_duplicates]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_splats) return;

    float sx = means2D[idx * 2 + 0];
    float sy = means2D[idx * 2 + 1];
    float radius = radii_px[idx];
    float depth = depths[idx];

    int tile_x_min = max(0, (int)((sx - radius) / 16.0f));
    int tile_x_max = min(num_tiles_x - 1, (int)((sx + radius) / 16.0f));
    int tile_y_min = max(0, (int)((sy - radius) / 16.0f));
    int tile_y_max = min(num_tiles_y - 1, (int)((sy + radius) / 16.0f));

    if (tile_x_max < tile_x_min || tile_y_max < tile_y_min) return;

    int write_base = dup_offsets[idx];
    int local_k = 0;

    // Emit one entry per touched tile
    for (int ty = tile_y_min; ty <= tile_y_max; ++ty) {
        for (int tx = tile_x_min; tx <= tile_x_max; ++tx) {
            int tile_id = ty * num_tiles_x + tx;
            uint64_t key = packTileDepthKey(tile_id, depth);
            
            out_keys[write_base + local_k] = key;
            out_values[write_base + local_k] = idx;
            ++local_k;
        }
    }
}

// Extract high-32-bit tile ids from sorted keys
__global__ void extractTileIdsKernel(const uint64_t* __restrict__ sorted_keys,
                                     uint32_t* __restrict__ tile_ids,
                                     int total_duplicates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_duplicates) {
        tile_ids[idx] = static_cast<uint32_t>(sorted_keys[idx] >> 32);
    }
}

// Scatter run offsets to tile_offsets for tiles that appear
__global__ void scatterTileOffsetsKernel(const uint32_t* __restrict__ unique_tile_ids,
                                         const int* __restrict__ run_offsets,
                                         int num_runs,
                                         int* __restrict__ tile_offsets) {
    int run = blockIdx.x * blockDim.x + threadIdx.x;
    if (run < num_runs) {
        uint32_t tile = unique_tile_ids[run];
        tile_offsets[tile] = run_offsets[run];
    }
}

// Set the sentinel tail offset
__global__ void setTileOffsetsTailKernel(int* tile_offsets, int num_tiles, int total_duplicates) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        tile_offsets[num_tiles] = total_duplicates;
    }
}

// Fill gaps so that empty tiles get start=end of next valid tile
__global__ void fillTileOffsetGapsKernel(int* tile_offsets, int num_tiles) {
    // Serial backward sweep is acceptable: num_tiles is moderate (< few 10k)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int t = num_tiles - 1; t >= 0; --t) {
            int next = tile_offsets[t + 1];
            int cur = tile_offsets[t];
            if (cur < 0) {
                tile_offsets[t] = next;
            }
        }
    }
}

// ============================================================
// CUDA Kernel: Naive Per-Pixel Gaussian Splatting
// ============================================================

// ============================================================
// CUDA Kernel: Tile-based Gaussian Splatting (V2)
// ============================================================

/**
 * Tile-based Gaussian Splat Kernel (V2 - Compact Tile Lists)
 * 
 * Each block processes a 16x16 tile using pre-computed tile splat lists.
 * This eliminates the need to iterate through all 950K+ splats per pixel.
 */

#define TILE_SIZE 16

__global__ void splatKernel(
    const float* means2D,
    const float* conic3D,
    const float* colors,
    const float* opacities,
    const int* tile_splat_list,       // Compact list of splat indices for this tile
    const int* tile_offsets,          // Offset into tile_splat_list for each tile
    int num_tiles_x,
    int width,
    int height,
    float* output
) {
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int px = tile_x * TILE_SIZE + thread_x;
    int py = tile_y * TILE_SIZE + thread_y;

    if (px >= width || py >= height) return;

    // Get tile splat list
    int tile_id = tile_y * num_tiles_x + tile_x;
    int splat_start = tile_offsets[tile_id];
    int splat_end = tile_offsets[tile_id + 1];

    // Pixel center in continuous coordinates
    float x = px + 0.5f;
    float y = py + 0.5f;

    // Initialize pixel color (background)
    const float bgColor = 30.0f / 255.0f;
    float C_r = bgColor;
    float C_g = bgColor;
    float C_b = bgColor;

    // Process only splats relevant to this tile
    for (int i = splat_start; i < splat_end; ++i) {
        int idx = tile_splat_list[i];

        float sx = means2D[idx * 2 + 0];
        float sy = means2D[idx * 2 + 1];

        float conic_a = conic3D[idx * 3 + 0];
        float conic_b = conic3D[idx * 3 + 1];
        float conic_c = conic3D[idx * 3 + 2];

        float color_r = colors[idx * 3 + 0];
        float color_g = colors[idx * 3 + 1];
        float color_b = colors[idx * 3 + 2];

        float opacity = opacities[idx];

        // Compute offset from gaussian center
        float dx = x - sx;
        float dy = y - sy;

        // Evaluate 2D Gaussian weight
        float power = -0.5f * (conic_a * dx * dx + 2.0f * conic_b * dx * dy + conic_c * dy * dy);

        if (power < -4.5f) continue;
        if (power > 0.0f) power = 0.0f;

        float w = expf(power);
        float alpha = opacity * w;
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);

        if (alpha < 1e-4f) continue;

        // Alpha compositing
        float src_a = alpha;
        float src_r = color_r * src_a;
        float src_g = color_g * src_a;
        float src_b = color_b * src_a;

        float one_minus_alpha = 1.0f - src_a;
        C_r = src_r + C_r * one_minus_alpha;
        C_g = src_g + C_g * one_minus_alpha;
        C_b = src_b + C_b * one_minus_alpha;

        if (one_minus_alpha < 1e-4f) break;
    }

    // Write final pixel color
    int pixel_idx = py * width + px;
    output[pixel_idx * 3 + 0] = C_r;
    output[pixel_idx * 3 + 1] = C_g;
    output[pixel_idx * 3 + 2] = C_b;
}

// Pack float RGB [0,1] into RGBA8 buffer on device
__global__ void packToRGBA8(const float* __restrict__ src_rgb,
                            unsigned char* __restrict__ dst_rgba8,
                            int width,
                            int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    float r = src_rgb[idx * 3 + 0];
    float g = src_rgb[idx * 3 + 1];
    float b = src_rgb[idx * 3 + 2];
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);
    int o = idx * 4;
    dst_rgba8[o + 0] = static_cast<unsigned char>(r * 255.0f + 0.5f);
    dst_rgba8[o + 1] = static_cast<unsigned char>(g * 255.0f + 0.5f);
    dst_rgba8[o + 2] = static_cast<unsigned char>(b * 255.0f + 0.5f);
    dst_rgba8[o + 3] = 255;
}

// ============================================================
// CUDA Kernel: GPU-side SH Color Evaluation
// ============================================================

// Evaluate 9 real SH basis functions (up to l = 2)
// Matches CPU-side evalSH9 from sh_color.cpp
__device__ inline void evalSH9_device(const glm::vec3& dir, float* sh)
{
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    sh[0] = 0.28209479177387814f;
    sh[1] = -0.4886025119029199f * y;
    sh[2] =  0.4886025119029199f * z;
    sh[3] = -0.4886025119029199f * x;
    sh[4] =  1.0925484305920792f * x * y;
    sh[5] = -1.0925484305920792f * y * z;
    sh[6] =  0.31539156525252005f * (3.0f * z * z - 1.0f);
    sh[7] = -1.0925484305920792f * x * z;
    sh[8] =  0.5462742152960396f * (x * x - y * y);
}

// Evaluate SH color for all splats given camera position
// One thread per splat
__global__ void evalSHColorKernel(
    const int* __restrict__ gaussian_ids,       // [N] gaussian index for each screen splat
    const float* __restrict__ pos_ws_xyz,       // [num_gaussians*3] world-space positions
    const float* __restrict__ dc_colors,        // [num_gaussians*3] DC colors
    const float* __restrict__ sh_coeffs,        // [num_gaussians*27] SH coefficients
    float cam_x, float cam_y, float cam_z,      // Camera position components
    int num_splats,
    float* __restrict__ out_colors              // [num_splats*3] output RGB colors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_splats) return;

    // Map screen splat to source gaussian
    int gid = gaussian_ids[idx];

    // Compute view direction (extract position as float3)
    float pos_x = pos_ws_xyz[gid * 3 + 0];
    float pos_y = pos_ws_xyz[gid * 3 + 1];
    float pos_z = pos_ws_xyz[gid * 3 + 2];
    
    float view_x = cam_x - pos_x;
    float view_y = cam_y - pos_y;
    float view_z = cam_z - pos_z;
    
    // Normalize view direction
    float len_sq = view_x * view_x + view_y * view_y + view_z * view_z;
    float len = sqrtf(len_sq);
    if (len < 1e-6f) len = 1e-6f;
    float inv_len = 1.0f / len;
    view_x *= inv_len;
    view_y *= inv_len;
    view_z *= inv_len;

    // Evaluate SH basis functions with float view direction (l≤2, 9 basis)
    float sh[9];
    float x = view_x, y = view_y, z = view_z;
    sh[0] = 0.28209479177387814f;
    sh[1] = -0.4886025119029199f * y;
    sh[2] =  0.4886025119029199f * z;
    sh[3] = -0.4886025119029199f * x;
    sh[4] =  1.0925484305920792f * x * y;
    sh[5] = -1.0925484305920792f * y * z;
    sh[6] =  0.31539156525252005f * (3.0f * z * z - 1.0f);
    sh[7] = -1.0925484305920792f * x * z;
    sh[8] =  0.5462742152960396f * (x * x - y * y);

    // Start with DC color
    float r = dc_colors[gid * 3 + 0];
    float g = dc_colors[gid * 3 + 1];
    float b = dc_colors[gid * 3 + 2];

    // Add higher-order SH contributions (indices 1..8, skip DC at 0)
    #pragma unroll 8
    for (int basis = 1; basis < 9; ++basis) {
        float w = sh[basis];
        r += sh_coeffs[gid * 27 + 0 * 9 + basis] * w;      // R coefficients at [0..8]
        g += sh_coeffs[gid * 27 + 1 * 9 + basis] * w;      // G coefficients at [9..17]
        b += sh_coeffs[gid * 27 + 2 * 9 + basis] * w;      // B coefficients at [18..26]
    }

    // Clamp to [0, 1]
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);

    // Write output
    out_colors[idx * 3 + 0] = r;
    out_colors[idx * 3 + 1] = g;
    out_colors[idx * 3 + 2] = b;
}

// ============================================================
// Host Code: Memory Management and Rendering
// ============================================================

Rasterizer::Rasterizer()
    : d_means2D_(nullptr)
    , d_conic3D_(nullptr)
    , d_colors_(nullptr)
    , d_opacities_(nullptr)
    , d_radii_px_(nullptr)
    , d_depths_(nullptr)
    , d_output_(nullptr)
    , d_tile_splat_list_(nullptr)
    , d_tile_offsets_(nullptr)
    , d_num_tiles_touched_(nullptr)
    , d_dup_offsets_(nullptr)
    , d_keys_(nullptr)
    , d_keys_sorted_(nullptr)
    , d_values_(nullptr)
    , d_values_sorted_(nullptr)
    , d_cub_temp_(nullptr)
    , tile_splat_list_capacity_(0)
    , tile_offsets_capacity_(0)
    , sort_buffer_capacity_(0)
    , cub_temp_bytes_(0)
    , num_splats_(0)
    , width_(0)
    , height_(0)
{
}

Rasterizer::~Rasterizer() {
    free();
}

bool Rasterizer::ensureFrameBuffers(int num_splats, int width, int height) {
    // Only reallocate if capacity grows
    if (num_splats <= frame_buffer_capacity_ && width_ == width && height_ == height) {
        num_splats_ = num_splats;
        return true;
    }

    num_splats_ = num_splats;
    width_ = width;
    height_ = height;
    frame_buffer_capacity_ = num_splats;

    // Allocate device memory for frame buffers
    if (d_means2D_) cudaFree(d_means2D_);
    if (d_conic3D_) cudaFree(d_conic3D_);
    if (d_colors_) cudaFree(d_colors_);
    if (d_opacities_) cudaFree(d_opacities_);
    if (d_radii_px_) cudaFree(d_radii_px_);
    if (d_depths_) cudaFree(d_depths_);
    if (d_output_) cudaFree(d_output_);
    if (d_gaussian_ids_) cudaFree(d_gaussian_ids_);

    CUDA_CHECK(cudaMalloc(&d_means2D_, num_splats * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conic3D_, num_splats * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colors_, num_splats * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_opacities_, num_splats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radii_px_, num_splats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depths_, num_splats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_, width * height * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gaussian_ids_, num_splats * sizeof(int)));

    printf("✅ Ensured frame buffers: %d splats, %dx%d image\n", num_splats, width, height);
    return true;
}

void Rasterizer::freeFrameBuffers() {
    if (d_means2D_) { cudaFree(d_means2D_); d_means2D_ = nullptr; }
    if (d_conic3D_) { cudaFree(d_conic3D_); d_conic3D_ = nullptr; }
    if (d_colors_) { cudaFree(d_colors_); d_colors_ = nullptr; }
    if (d_opacities_) { cudaFree(d_opacities_); d_opacities_ = nullptr; }
    if (d_radii_px_) { cudaFree(d_radii_px_); d_radii_px_ = nullptr; }
    if (d_depths_) { cudaFree(d_depths_); d_depths_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    if (d_gaussian_ids_) { cudaFree(d_gaussian_ids_); d_gaussian_ids_ = nullptr; }
    if (d_tile_splat_list_) { cudaFree(d_tile_splat_list_); d_tile_splat_list_ = nullptr; tile_splat_list_capacity_ = 0; }
    if (d_tile_offsets_) { cudaFree(d_tile_offsets_); d_tile_offsets_ = nullptr; tile_offsets_capacity_ = 0; }
    if (d_num_tiles_touched_) { cudaFree(d_num_tiles_touched_); d_num_tiles_touched_ = nullptr; }
    if (d_dup_offsets_) { cudaFree(d_dup_offsets_); d_dup_offsets_ = nullptr; }
    if (d_keys_) { cudaFree(d_keys_); d_keys_ = nullptr; }
    if (d_keys_sorted_) { cudaFree(d_keys_sorted_); d_keys_sorted_ = nullptr; }
    if (d_values_) { cudaFree(d_values_); d_values_ = nullptr; }
    if (d_values_sorted_) { cudaFree(d_values_sorted_); d_values_sorted_ = nullptr; }
    if (d_tile_ids_sorted_) { cudaFree(d_tile_ids_sorted_); d_tile_ids_sorted_ = nullptr; }
    if (d_unique_tile_ids_) { cudaFree(d_unique_tile_ids_); d_unique_tile_ids_ = nullptr; }
    if (d_run_lengths_) { cudaFree(d_run_lengths_); d_run_lengths_ = nullptr; }
    if (d_run_offsets_) { cudaFree(d_run_offsets_); d_run_offsets_ = nullptr; }
    if (d_num_runs_device_) { cudaFree(d_num_runs_device_); d_num_runs_device_ = nullptr; }
    if (d_cub_temp_) { cudaFree(d_cub_temp_); d_cub_temp_ = nullptr; cub_temp_bytes_ = 0; }
    frame_buffer_capacity_ = 0;
    sort_buffer_capacity_ = 0;
    run_buffer_capacity_ = 0;
    per_splat_capacity_ = 0;
    
    // Host buffers: shrink to zero but keep capacity
    h_means2D_.clear();
    h_conic3D_.clear();
    h_opacities_.clear();
    h_radii_px_.clear();
    h_depths_.clear();
    h_gaussian_ids_.clear();
}

void Rasterizer::free() {
    freeFrameBuffers();
    // Free scene-static data
    if (d_pos_ws_) { cudaFree(d_pos_ws_); d_pos_ws_ = nullptr; }
    if (d_sh_coeffs_) { cudaFree(d_sh_coeffs_); d_sh_coeffs_ = nullptr; }
    if (d_dc_colors_) { cudaFree(d_dc_colors_); d_dc_colors_ = nullptr; }
    scene_splat_capacity_ = 0;
    scene_num_splats_ = 0;
    gaussians_ptr_ = nullptr;
}

bool Rasterizer::ensureHostBuffers(int num_splats) {
    // Ensure host vectors have capacity but don't reallocate if sufficient
    if (static_cast<int>(host_buffer_capacity_) < num_splats) {
        // Allocate with some margin to avoid repeated reallocations
        size_t new_capacity = static_cast<size_t>(num_splats) * 1.1 + 1000;
        h_means2D_.reserve(new_capacity * 2);
        h_conic3D_.reserve(new_capacity * 3);
        h_opacities_.reserve(new_capacity);
        h_radii_px_.reserve(new_capacity);
        h_depths_.reserve(new_capacity);
        h_gaussian_ids_.reserve(new_capacity);
        host_buffer_capacity_ = new_capacity;
    }
    // Resize to actual count (but capacity won't shrink)
    h_means2D_.resize(num_splats * 2);
    h_conic3D_.resize(num_splats * 3);
    h_opacities_.resize(num_splats);
    h_radii_px_.resize(num_splats);
    h_depths_.resize(num_splats);
    h_gaussian_ids_.resize(num_splats);
    return true;
}

bool Rasterizer::ensureTileBuffers(size_t splat_list_count, size_t offsets_count) {
    if (splat_list_count > tile_splat_list_capacity_) {
        if (d_tile_splat_list_) cudaFree(d_tile_splat_list_);
        CUDA_CHECK(cudaMalloc(&d_tile_splat_list_, splat_list_count * sizeof(int)));
        tile_splat_list_capacity_ = splat_list_count;
    }
    if (offsets_count > tile_offsets_capacity_) {
        if (d_tile_offsets_) cudaFree(d_tile_offsets_);
        CUDA_CHECK(cudaMalloc(&d_tile_offsets_, offsets_count * sizeof(int)));
        tile_offsets_capacity_ = offsets_count;
    }
    return true;
}

bool Rasterizer::ensureSortBuffers(int num_splats, size_t total_duplicates) {
    size_t required_splats = static_cast<size_t>(num_splats);

    // Per-splat buffers (counts + offsets)
    if (required_splats > per_splat_capacity_) {
        if (d_num_tiles_touched_) cudaFree(d_num_tiles_touched_);
        if (d_dup_offsets_) cudaFree(d_dup_offsets_);
        CUDA_CHECK(cudaMalloc(&d_num_tiles_touched_, required_splats * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dup_offsets_, (required_splats + 1) * sizeof(int)));
        per_splat_capacity_ = required_splats;
    } else {
        if (!d_num_tiles_touched_) CUDA_CHECK(cudaMalloc(&d_num_tiles_touched_, per_splat_capacity_ * sizeof(int)));
        if (!d_dup_offsets_) CUDA_CHECK(cudaMalloc(&d_dup_offsets_, (per_splat_capacity_ + 1) * sizeof(int)));
    }

    // Duplicate arrays for sort
    if (total_duplicates > sort_buffer_capacity_) {
        if (d_keys_) cudaFree(d_keys_);
        if (d_keys_sorted_) cudaFree(d_keys_sorted_);
        if (d_values_) cudaFree(d_values_);
        if (d_values_sorted_) cudaFree(d_values_sorted_);

        CUDA_CHECK(cudaMalloc(&d_keys_, total_duplicates * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_sorted_, total_duplicates * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_values_, total_duplicates * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_values_sorted_, total_duplicates * sizeof(int)));

        sort_buffer_capacity_ = total_duplicates;
    } else {
        if (!d_keys_) CUDA_CHECK(cudaMalloc(&d_keys_, sort_buffer_capacity_ * sizeof(uint64_t)));
        if (!d_keys_sorted_) CUDA_CHECK(cudaMalloc(&d_keys_sorted_, sort_buffer_capacity_ * sizeof(uint64_t)));
        if (!d_values_) CUDA_CHECK(cudaMalloc(&d_values_, sort_buffer_capacity_ * sizeof(int)));
        if (!d_values_sorted_) CUDA_CHECK(cudaMalloc(&d_values_sorted_, sort_buffer_capacity_ * sizeof(int)));
    }

    // RLE buffers (size <= total_duplicates)
    size_t rle_needed = total_duplicates;
    if (rle_needed > run_buffer_capacity_) {
        if (d_tile_ids_sorted_) cudaFree(d_tile_ids_sorted_);
        if (d_unique_tile_ids_) cudaFree(d_unique_tile_ids_);
        if (d_run_lengths_) cudaFree(d_run_lengths_);
        if (d_run_offsets_) cudaFree(d_run_offsets_);

        if (rle_needed == 0) rle_needed = 1; // allocate minimal to avoid nullptr
        CUDA_CHECK(cudaMalloc(&d_tile_ids_sorted_, rle_needed * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_unique_tile_ids_, rle_needed * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_run_lengths_, rle_needed * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_run_offsets_, rle_needed * sizeof(int)));

        run_buffer_capacity_ = rle_needed;
    } else {
        if (!d_tile_ids_sorted_) CUDA_CHECK(cudaMalloc(&d_tile_ids_sorted_, run_buffer_capacity_ * sizeof(uint32_t)));
        if (!d_unique_tile_ids_) CUDA_CHECK(cudaMalloc(&d_unique_tile_ids_, run_buffer_capacity_ * sizeof(uint32_t)));
        if (!d_run_lengths_) CUDA_CHECK(cudaMalloc(&d_run_lengths_, run_buffer_capacity_ * sizeof(int)));
        if (!d_run_offsets_) CUDA_CHECK(cudaMalloc(&d_run_offsets_, run_buffer_capacity_ * sizeof(int)));
    }

    if (!d_num_runs_device_) {
        CUDA_CHECK(cudaMalloc(&d_num_runs_device_, sizeof(int)));
    }

    return true;
}

bool Rasterizer::debugValidateTileOffsets(int num_tiles, size_t total_duplicates) {
#if defined(ENABLE_PROFILING) || !defined(NDEBUG)
    std::vector<int> h_offsets(static_cast<size_t>(num_tiles) + 1);
    CUDA_CHECK(cudaMemcpy(h_offsets.data(), d_tile_offsets_, (num_tiles + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    for (int t = 0; t < num_tiles; ++t) {
        if (h_offsets[t] < 0) {
            fprintf(stderr, "[validate] tile %d has negative offset %d\n", t, h_offsets[t]);
            return false;
        }
        if (h_offsets[t] > h_offsets[t + 1]) {
            fprintf(stderr, "[validate] tile_offsets not monotonic at tile %d: %d > %d\n", t, h_offsets[t], h_offsets[t + 1]);
            return false;
        }
    }

    if (static_cast<size_t>(h_offsets[num_tiles]) != total_duplicates) {
        fprintf(stderr, "[validate] tail mismatch: tile_offsets[%d]=%d, expected %zu\n", num_tiles, h_offsets[num_tiles], total_duplicates);
        return false;
    }
#else
    (void)num_tiles;
    (void)total_duplicates;
#endif
    return true;
}

bool Rasterizer::buildTileBinning(int num_splats, int num_tiles_x, int num_tiles_y, int num_tiles, size_t& total_duplicates) {
    dim3 block256(256);
    dim3 grid256((num_splats + 255) / 256);

    // Ensure per-splat buffers exist (duplicates capacity will be resized after total_duplicates is known)
    if (!ensureSortBuffers(num_splats, 1)) return false;

    // Phase 1: tile touch counts
    computeTileTouchCountKernel<<<grid256, block256>>>(
        d_means2D_, d_radii_px_, num_splats, num_tiles_x, num_tiles_y, d_num_tiles_touched_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_DEBUG_SYNC();

    // Phase 2: exclusive scan for duplicate offsets
    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, d_num_tiles_touched_, d_dup_offsets_, num_splats);

    auto ensureCubTemp = [&](size_t bytes) -> bool {
        if (bytes > cub_temp_bytes_) {
            if (d_cub_temp_) cudaFree(d_cub_temp_);
            cudaError_t err = cudaMalloc(&d_cub_temp_, bytes);
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaMalloc for CUB temp failed: %s\n", cudaGetErrorString(err));
                return false;
            }
            cub_temp_bytes_ = bytes;
        }
        return true;
    };

    if (!ensureCubTemp(scan_temp_bytes)) return false;
    cub::DeviceScan::ExclusiveSum(d_cub_temp_, scan_temp_bytes, d_num_tiles_touched_, d_dup_offsets_, num_splats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_DEBUG_SYNC();

    // Phase 3: compute total_duplicates safely
    int last_offset = 0;
    int last_count = 0;
    CUDA_CHECK(cudaMemcpy(&last_offset, d_dup_offsets_ + num_splats - 1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_count, d_num_tiles_touched_ + num_splats - 1, sizeof(int), cudaMemcpyDeviceToHost));

    if (last_offset < 0 || last_count < 0) {
        fprintf(stderr, "Invalid scan results: last_offset=%d, last_count=%d\n", last_offset, last_count);
        return false;
    }

    size_t total = static_cast<size_t>(last_offset) + static_cast<size_t>(last_count);
    if (total < static_cast<size_t>(last_offset)) {
        fprintf(stderr, "Overflow computing total_duplicates\n");
        return false;
    }

    if (total > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "total_duplicates=%zu exceeds int range\n", total);
        return false;
    }

    const size_t max_per_splat = 1024; // conservative upper bound
    size_t hard_cap = static_cast<size_t>(num_splats) * max_per_splat;
    if (total > hard_cap) {
        fprintf(stderr, "total_duplicates=%zu exceeds conservative cap=%zu; aborting to avoid OOM/overflow\n", total, hard_cap);
        return false;
    }

    total_duplicates = total;

    // Phase 4: allocate buffers based on total_duplicates
    if (!ensureSortBuffers(num_splats, total_duplicates)) return false;
    if (!ensureTileBuffers(total_duplicates, static_cast<size_t>(num_tiles) + 1)) return false;

    if (total_duplicates == 0) {
        CUDA_CHECK(cudaMemset(d_tile_offsets_, 0, (num_tiles + 1) * sizeof(int)));
        return true;
    }

    // Phase 5: emit duplicate keys/values
    emitDuplicateKeysKernel<<<grid256, block256>>>(
        d_means2D_, d_radii_px_, d_depths_, num_splats, num_tiles_x, num_tiles_y,
        d_dup_offsets_, d_keys_, d_values_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_DEBUG_SYNC();

    // Phase 6: radix sort (tile_id primary, depth secondary)
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_temp_bytes, d_keys_, d_keys_sorted_, d_values_, d_values_sorted_, total_duplicates);
    if (!ensureCubTemp(sort_temp_bytes)) return false;
    cub::DeviceRadixSort::SortPairs(d_cub_temp_, sort_temp_bytes, d_keys_, d_keys_sorted_, d_values_, d_values_sorted_, total_duplicates);
    CUDA_CHECK(cudaGetLastError());
    CUDA_DEBUG_SYNC();

    // Phase 7: extract tile ids for RLE
    dim3 gridDup((static_cast<int>(total_duplicates) + 255) / 256);
    extractTileIdsKernel<<<gridDup, block256>>>(d_keys_sorted_, d_tile_ids_sorted_, static_cast<int>(total_duplicates));
    CUDA_CHECK(cudaGetLastError());
    CUDA_DEBUG_SYNC();

    // Phase 8: run-length encode tile ids
    size_t rle_temp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, rle_temp_bytes,
        d_tile_ids_sorted_, d_unique_tile_ids_, d_run_lengths_, d_num_runs_device_, static_cast<int>(total_duplicates));
    if (!ensureCubTemp(rle_temp_bytes)) return false;
    cub::DeviceRunLengthEncode::Encode(d_cub_temp_, rle_temp_bytes,
        d_tile_ids_sorted_, d_unique_tile_ids_, d_run_lengths_, d_num_runs_device_, static_cast<int>(total_duplicates));
    CUDA_CHECK(cudaGetLastError());
    CUDA_DEBUG_SYNC();

    int h_num_runs = 0;
    CUDA_CHECK(cudaMemcpy(&h_num_runs, d_num_runs_device_, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_num_runs <= 0) {
        fprintf(stderr, "Run-length encode produced zero runs while total_duplicates=%zu\n", total_duplicates);
        return false;
    }

    // Phase 9: exclusive scan of run lengths -> run offsets
    size_t run_scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, run_scan_temp_bytes, d_run_lengths_, d_run_offsets_, h_num_runs);
    if (!ensureCubTemp(run_scan_temp_bytes)) return false;
    cub::DeviceScan::ExclusiveSum(d_cub_temp_, run_scan_temp_bytes, d_run_lengths_, d_run_offsets_, h_num_runs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_DEBUG_SYNC();

    // Phase 10: scatter run offsets into tile_offsets
    CUDA_CHECK(cudaMemset(d_tile_offsets_, 0xFF, (num_tiles + 1) * sizeof(int))); // set to -1
    dim3 gridRuns((h_num_runs + 255) / 256);
    scatterTileOffsetsKernel<<<gridRuns, block256>>>(d_unique_tile_ids_, d_run_offsets_, h_num_runs, d_tile_offsets_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_DEBUG_SYNC();

    setTileOffsetsTailKernel<<<1, 1>>>(d_tile_offsets_, num_tiles, static_cast<int>(total_duplicates));
    CUDA_CHECK(cudaGetLastError());
    CUDA_DEBUG_SYNC();

    fillTileOffsetGapsKernel<<<1, 1>>>(d_tile_offsets_, num_tiles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_DEBUG_SYNC();

    // Copy sorted splat indices into tile_splat_list
    CUDA_CHECK(cudaMemcpy(d_tile_splat_list_, d_values_sorted_, total_duplicates * sizeof(int), cudaMemcpyDeviceToDevice));

    return debugValidateTileOffsets(num_tiles, total_duplicates);
}

bool Rasterizer::uploadSceneData(const std::vector<gs::GaussianSplat>& gaussians) {
    int num = static_cast<int>(gaussians.size());
    if (num == 0) {
        fprintf(stderr, "Error: No gaussians to upload\n");
        return false;
    }

    // Save reference for render-time lookup
    gaussians_ptr_ = &gaussians;

    printf("📤 Uploading scene data (%d gaussians) to GPU...\n", num);
    printf("  [DEBUG] sizeof(glm::vec3) = %zu bytes\n", sizeof(glm::vec3));

    // Prepare host buffers - use tightly packed float arrays to avoid GLM alignment issues
    std::vector<float> h_pos_ws(num * 3);  // Tightly packed x,y,z floats
    std::vector<float> h_dc_colors(num * 3);
    std::vector<float> h_sh_coeffs(num * 27);  // 3 channels * 9 basis functions (Degree 2)

    for (int i = 0; i < num; ++i) {
        const gs::GaussianSplat& g = gaussians[i];
        
        // Unpack position to avoid GLM padding/alignment issues
        h_pos_ws[i * 3 + 0] = g.position_ws.x;
        h_pos_ws[i * 3 + 1] = g.position_ws.y;
        h_pos_ws[i * 3 + 2] = g.position_ws.z;
        
        h_dc_colors[i * 3 + 0] = g.dc_color.r;
        h_dc_colors[i * 3 + 1] = g.dc_color.g;
        h_dc_colors[i * 3 + 2] = g.dc_color.b;

        // Pack SH coefficients: [R_0..8, G_0..8, B_0..8]
        // Note: original data layout is [R0..Rn, G0..Gn, B0..Bn] where n = original n_basis
        int original_n_basis = static_cast<int>(g.sh_color.coeffs.size()) / 3;
        int used_basis = std::min(original_n_basis, 9);
        
        for (int b = 0; b < 9; ++b) {
            if (b < used_basis) {
                // Use original_n_basis for indexing into source data
                h_sh_coeffs[i * 27 + 0 * 9 + b] = g.sh_color.coeffs[0 * original_n_basis + b]; // R
                h_sh_coeffs[i * 27 + 1 * 9 + b] = g.sh_color.coeffs[1 * original_n_basis + b]; // G
                h_sh_coeffs[i * 27 + 2 * 9 + b] = g.sh_color.coeffs[2 * original_n_basis + b]; // B
            } else {
                h_sh_coeffs[i * 27 + 0 * 9 + b] = 0.0f;
                h_sh_coeffs[i * 27 + 1 * 9 + b] = 0.0f;
                h_sh_coeffs[i * 27 + 2 * 9 + b] = 0.0f;
            }
        }
    }

    // Allocate/reallocate device memory only if capacity grows
    if (num > scene_splat_capacity_) {
        if (d_pos_ws_) cudaFree(d_pos_ws_);
        if (d_sh_coeffs_) cudaFree(d_sh_coeffs_);
        if (d_dc_colors_) cudaFree(d_dc_colors_);

        CUDA_CHECK(cudaMalloc(&d_pos_ws_, num * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sh_coeffs_, num * 27 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dc_colors_, num * 3 * sizeof(float)));
        scene_splat_capacity_ = num;
    }

    // Upload to device
    printf("  Uploading pos_ws: %zu bytes\n", num * 3 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_pos_ws_, h_pos_ws.data(), num * 3 * sizeof(float), cudaMemcpyHostToDevice));
    printf("  Uploading sh_coeffs: %zu bytes\n", num * 27 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_sh_coeffs_, h_sh_coeffs.data(), num * 27 * sizeof(float), cudaMemcpyHostToDevice));
    printf("  Uploading dc_colors: %zu bytes\n", num * 3 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_dc_colors_, h_dc_colors.data(), num * 3 * sizeof(float), cudaMemcpyHostToDevice));

    scene_num_splats_ = num;
    printf("✅ Scene data uploaded: %d gaussians\n  Pointers: pos_ws=%p, sh_coeffs=%p, dc_colors=%p\n", 
           num, d_pos_ws_, d_sh_coeffs_, d_dc_colors_);
    return true;
}

bool Rasterizer::render_cuda(
    const std::vector<gs::ScreenSplat>& screen_splats,
    const glm::vec3& camera_pos,
    int width,
    int height,
    float* output_image
) {
    int num_splats = static_cast<int>(screen_splats.size());
    
    if (num_splats == 0) {
        fprintf(stderr, "Error: No splats to render!\n");
        return false;
    }

    if (!gaussians_ptr_) {
        fprintf(stderr, "Error: Scene data not uploaded. Call uploadSceneData() first.\n");
        return false;
    }

    printf("🚀 CUDA V2 Tile-based Render: %d splats, %dx%d image\n", num_splats, width, height);

    double t_sh_ms = 0.0;
    double t_tile_ms = 0.0;
    double t_h2d_ms = 0.0;
    double t_d2h_ms = 0.0;
    float gpu_total_ms = 0.0f;

    // Ensure frame buffers have capacity (realloc only if needed)
    if (!ensureFrameBuffers(num_splats, width, height)) {
        fprintf(stderr, "❌ Failed to ensure frame buffers\n");
        return false;
    }

    // Prepare host arrays for upload (convert ScreenSplat to SoA format)
    printf("🔄 Preparing %d splats for upload...\n", num_splats);
    fflush(stdout);
    
    // Reuse persistent host buffers (no malloc/free per frame)
    if (!ensureHostBuffers(num_splats)) {
        fprintf(stderr, "❌ Failed to ensure host buffers\n");
        return false;
    }

    {
        ScopedTimer timer("pack_screen_data", &t_sh_ms);
        for (int i = 0; i < num_splats; ++i) {
            if (i % 100000 == 0) {
                printf("  Progress: %d / %d splats (%.1f%%)...\n", i, num_splats, 100.0f * i / num_splats);
                fflush(stdout);
            }
            
            const gs::ScreenSplat& sp = screen_splats[i];
            
            if (!sp.src) {
                fprintf(stderr, "Error: Null source pointer at splat %d\n", i);
                return false;
            }
            
            h_means2D_[i * 2 + 0] = sp.sx;
            h_means2D_[i * 2 + 1] = sp.sy;
            
            h_conic3D_[i * 3 + 0] = sp.cov_inv[0][0];
            h_conic3D_[i * 3 + 1] = sp.cov_inv[0][1];
            h_conic3D_[i * 3 + 2] = sp.cov_inv[1][1];
            
            h_opacities_[i] = sp.src->opacity;
            h_radii_px_[i] = sp.radius_px;         // Conservative radius for tile binning
            h_depths_[i] = sp.depth;                // Depth for back-to-front sorting
            
            // Use cached gaussian_id from ScreenSplat projection
            // Already assigned in cpu_rasterizer.cpp during projection
            h_gaussian_ids_[i] = sp.gaussian_id;
            
            if (sp.gaussian_id < 0) {
                fprintf(stderr, "Error: gaussian_id not set in splat %d\n", i);
                return false;
            }
        }
    }

    printf("✅ Screen data packed. Computing SH colors on GPU...\n");
    fflush(stdout);

    // Upload data to GPU
    printf("📤 Uploading data to GPU...\n");
    fflush(stdout);

    {
        ScopedTimer timer("upload_h2d", &t_h2d_ms);
        CUDA_CHECK(cudaMemcpy(d_means2D_, h_means2D_.data(), 
                             num_splats * 2 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conic3D_, h_conic3D_.data(), 
                             num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_opacities_, h_opacities_.data(), 
                             num_splats * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_radii_px_, h_radii_px_.data(),
                             num_splats * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_depths_, h_depths_.data(),
                             num_splats * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gaussian_ids_, h_gaussian_ids_.data(),
                             num_splats * sizeof(int), cudaMemcpyHostToDevice));
    }

    printf("✅ Uploaded data to GPU\n");
    fflush(stdout);

    // Launch GPU SH evaluation kernel
    {
        ScopedTimer timer("gpu_sh_eval", &t_sh_ms);
        dim3 block(256);
        dim3 grid((num_splats + 255) / 256);
        
        printf("  Launching evalSHColorKernel: grid=(%u, 1, 1), block=(%u, 1, 1), num_splats=%d\n",
               grid.x, block.x, num_splats);
        
        evalSHColorKernel<<<grid, block>>>(
            d_gaussian_ids_,
            d_pos_ws_,
            d_dc_colors_,
            d_sh_coeffs_,
            camera_pos.x, camera_pos.y, camera_pos.z,
            num_splats,
            d_colors_
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // D2H copy will implicitly synchronize; no explicit sync needed here
        printf("  ✓ evalSHColorKernel enqueued (will sync via D2H copy)\n");
        fflush(stdout);
    }

    printf("✅ SH evaluation complete (GPU)\n");
    fflush(stdout);

    // ============================================================
    // GPU Tile Binning & Sorting Pipeline (Complete GPU-side)
    // ============================================================
    printf("🔨 GPU Tile Binning & Sorting (back-to-front)...\n");
    fflush(stdout);

    int num_tiles_x = (width + 15) / 16;
    int num_tiles_y = (height + 15) / 16;
    int num_tiles = num_tiles_x * num_tiles_y;
    size_t total_duplicates = 0;

    {
        ScopedTimer timer("gpu_tile_binning_sort", &t_tile_ms);
        if (!buildTileBinning(num_splats, num_tiles_x, num_tiles_y, num_tiles, total_duplicates)) {
            fprintf(stderr, "Tile binning failed\n");
            return false;
        }
    }

    printf("✅ GPU tile binning & sorting complete: %zu duplicates, %d tiles\n",
           total_duplicates, num_tiles);
    fflush(stdout);

    // Launch tile-based rendering kernel (unchanged)
    dim3 block(16, 16);
    dim3 grid(num_tiles_x, num_tiles_y);

    printf("🔧 Launching tile-based kernel: grid(%d, %d), block(%d, %d)\n",
           grid.x, grid.y, block.x, block.y);
    fflush(stdout);

    CudaTimer kernel_timer("raster_kernel", &gpu_total_ms);
    kernel_timer.start();

    splatKernel<<<grid, block>>>(
        d_means2D_,
        d_conic3D_,
        d_colors_,
        d_opacities_,
        d_tile_splat_list_,
        d_tile_offsets_,
        num_tiles_x,
        width,
        height,
        d_output_
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
        return false;
    }

    // No explicit sync; host copy will synchronize
    {
        ScopedTimer timer("download_d2h", &t_d2h_ms);
        CUDA_CHECK(cudaMemcpy(output_image, d_output_, 
                             width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Stop timer after copy (implies kernel completion)
    kernel_timer.stop();

#if ENABLE_PROFILING
    double cpu_total_ms = t_sh_ms + t_tile_ms + t_h2d_ms + t_d2h_ms;
    std::printf("[Profile] stats: splats_total=%d, tile_refs=%zu, res=%dx%d\n",
                num_splats, total_duplicates, width, height);
    std::printf("[Profile] total_cpu_render: %.3f ms | total_gpu: %.3f ms\n",
                cpu_total_ms, static_cast<double>(gpu_total_ms));
#endif

    // Tile buffers are cached across frames; do not free here

    return true;
}

bool Rasterizer::render_cuda_to_rgba8_device(
    const std::vector<gs::ScreenSplat>& screen_splats,
    const glm::vec3& camera_pos,
    int width,
    int height,
    unsigned char* out_rgba8_device
) {
    // Reuse the existing pipeline through tile building and kernel, but pack to device
    int num_splats = static_cast<int>(screen_splats.size());
    if (num_splats == 0) {
        fprintf(stderr, "Error: No splats to render!\n");
        return false;
    }

    if (!gaussians_ptr_) {
        fprintf(stderr, "Error: Scene data not uploaded. Call uploadSceneData() first.\n");
        return false;
    }

    // Ensure frame buffers have capacity (realloc only if needed)
    if (!ensureFrameBuffers(num_splats, width, height)) return false;

    // Reuse persistent host buffers
    if (!ensureHostBuffers(num_splats)) return false;

    for (int i = 0; i < num_splats; ++i) {
        const gs::ScreenSplat& sp = screen_splats[i];
        if (!sp.src) { fprintf(stderr, "Null src at %d\n", i); return false; }
        
        h_means2D_[i * 2 + 0] = sp.sx;
        h_means2D_[i * 2 + 1] = sp.sy;
        h_conic3D_[i * 3 + 0] = sp.cov_inv[0][0];
        h_conic3D_[i * 3 + 1] = sp.cov_inv[0][1];
        h_conic3D_[i * 3 + 2] = sp.cov_inv[1][1];
        h_opacities_[i] = sp.src->opacity;
        h_radii_px_[i] = sp.radius_px;
        h_depths_[i] = sp.depth;
        
        // Use cached gaussian_id from ScreenSplat projection
        h_gaussian_ids_[i] = sp.gaussian_id;
        
        if (sp.gaussian_id < 0) {
            fprintf(stderr, "Error: gaussian_id not set in splat %d\n", i);
            return false;
        }
    }

    // Upload to GPU
    CUDA_CHECK(cudaMemcpy(d_means2D_, h_means2D_.data(), num_splats * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conic3D_, h_conic3D_.data(), num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacities_, h_opacities_.data(), num_splats * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_radii_px_, h_radii_px_.data(), num_splats * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_depths_, h_depths_.data(), num_splats * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gaussian_ids_, h_gaussian_ids_.data(), num_splats * sizeof(int), cudaMemcpyHostToDevice));

    // Launch GPU SH evaluation kernel
    {
        dim3 block(256);
        dim3 grid((num_splats + 255) / 256);
        
        printf("  [PBO] Launching evalSHColorKernel: grid=(%u, 1, 1), block=(%u, 1, 1), num_splats=%d\n",
               grid.x, block.x, num_splats);
        
        evalSHColorKernel<<<grid, block>>>(
            d_gaussian_ids_,
            d_pos_ws_,
            d_dc_colors_,
            d_sh_coeffs_,
            camera_pos.x, camera_pos.y, camera_pos.z,
            num_splats,
            d_colors_
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel launch error (PBO): %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Stream ordering ensures kernel before tile kernel; OpenGL unmap provides fence
        printf("  ✓ [PBO] evalSHColorKernel enqueued (stream-ordered execution)\n");
        fflush(stdout);
    }

    int num_tiles_x = (width + 15) / 16;
    int num_tiles_y = (height + 15) / 16;
    int num_tiles = num_tiles_x * num_tiles_y;
    size_t total_duplicates = 0;

    // GPU tile binning & sorting (same as render_cuda)
    {
        if (!buildTileBinning(num_splats, num_tiles_x, num_tiles_y, num_tiles, total_duplicates)) {
            fprintf(stderr, "Tile binning failed (PBO path)\n");
            return false;
        }
    }

    // Render
    dim3 block(16, 16);
    dim3 grid(num_tiles_x, num_tiles_y);
    splatKernel<<<grid, block>>>(
        d_means2D_, d_conic3D_, d_colors_, d_opacities_,
        d_tile_splat_list_, d_tile_offsets_,
        num_tiles_x,
        width, height,
        d_output_);

    // Pack to RGBA8 directly into provided device buffer (PBO)
    dim3 p(16, 16);
    dim3 g((width + 15)/16, (height + 15)/16);
    packToRGBA8<<<g, p>>>(d_output_, out_rgba8_device, width, height);

    return true;
}

} // namespace CudaRasterizer
