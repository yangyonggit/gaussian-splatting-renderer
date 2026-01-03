#include "cuda_rasterizer.h"
#include "gs/gaussian.h"
#include "gs/screen_splat.h"
#include "gs/sh_color.h"
#include "gs/profiler.h"
#include "gs/nvtx_helper.h"
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

// Lightweight matrix struct to avoid relying on GLM mat operations in device code.
// Layout matches GLM column-major: m[col*4 + row]
struct Mat4f {
    float m[16];
};

__host__ inline Mat4f toMat4f(const glm::mat4& M) {
    Mat4f out{};
    const float* p = &M[0][0];
    std::memcpy(out.m, p, 16 * sizeof(float));
    return out;
}

__device__ __forceinline__ float4 mulMat4Vec4(const Mat4f& M, const float4& v) {
    return make_float4(
        M.m[0] * v.x + M.m[4] * v.y + M.m[8]  * v.z + M.m[12] * v.w,
        M.m[1] * v.x + M.m[5] * v.y + M.m[9]  * v.z + M.m[13] * v.w,
        M.m[2] * v.x + M.m[6] * v.y + M.m[10] * v.z + M.m[14] * v.w,
        M.m[3] * v.x + M.m[7] * v.y + M.m[11] * v.z + M.m[15] * v.w
    );
}

__device__ __forceinline__ float2 ndcToPixelEllipse(float ndc_x, float ndc_y, int width, int height) {
    // Must match gs::projectToScreenEllipse's ndcToPixel() exactly.
    float sx = (1.0f - (ndc_x * 0.5f + 0.5f)) * static_cast<float>(width);
    float sy = (ndc_y * 0.5f + 0.5f) * static_cast<float>(height);
    return make_float2(sx, sy);
}

__device__ __forceinline__ float clampf(float x, float a, float b) {
    return fminf(fmaxf(x, a), b);
}

__device__ __forceinline__ void quatToMat3Cols(float qw, float qx, float qy, float qz,
                                               float3& c0, float3& c1, float3& c2) {
    // Quaternion (w,x,y,z) to rotation matrix columns, consistent with GLM mat3_cast.
    float xx = qx * qx;
    float yy = qy * qy;
    float zz = qz * qz;
    float xy = qx * qy;
    float xz = qx * qz;
    float yz = qy * qz;
    float wx = qw * qx;
    float wy = qw * qy;
    float wz = qw * qz;

    // Columns of the rotation matrix
    c0 = make_float3(1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz),         2.0f * (xz - wy));
    c1 = make_float3(2.0f * (xy - wz),         1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx));
    c2 = make_float3(2.0f * (xz + wy),         2.0f * (yz - wx),         1.0f - 2.0f * (xx + yy));
}

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
//#define CUDA_DEBUG_SYNC() CUDA_CHECK(cudaDeviceSynchronize())
#define CUDA_DEBUG_SYNC()
#else
#define CUDA_DEBUG_SYNC()
#endif

namespace CudaRasterizer {

// ============================================================
// Tile binning capacity policy (conservative pre-allocation)
// ============================================================
static constexpr int kTileSizePx = 16;
static constexpr int kDuplicateReserveFactor = 32; // preallocate num_splats * 16 duplicates

// To guarantee duplicates never exceed capacity without host-side total_duplicates,
// clamp per-splat tile footprint by limiting radius.
// Worst-case tile span per axis is approximately (2r / tile + 1).
// For kDuplicateReserveFactor=16 => 4x4 tiles. Solve (2r/16 + 1) <= 4 => r <= 24.
static constexpr float kMaxRadiusPxForTileBudget = 512.0f;

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
    radius = fminf(radius, kMaxRadiusPxForTileBudget);

    // Cull invalid/disabled splats early.
    if (!(radius > 0.0f) || !isfinite(sx) || !isfinite(sy)) {
        num_tiles_touched[idx] = 0;
        return;
    }

    // Compute tile bounding box
    int tile_x_min = max(0, (int)((sx - radius) / float(kTileSizePx)));
    int tile_x_max = min(num_tiles_x - 1, (int)((sx + radius) / float(kTileSizePx)));
    int tile_y_min = max(0, (int)((sy - radius) / float(kTileSizePx)));
    int tile_y_max = min(num_tiles_y - 1, (int)((sy + radius) / float(kTileSizePx)));

    // Clamp to valid range (safety)
    if (tile_x_max < tile_x_min || tile_y_max < tile_y_min) {
        num_tiles_touched[idx] = 0;
        return;
    }

    int count = (tile_x_max - tile_x_min + 1) * (tile_y_max - tile_y_min + 1);
    num_tiles_touched[idx] = count;
}

// ============================================================
// CUDA Kernel: GPU Preprocess (project + ellipse params)
// Mirrors gs::projectToScreenEllipse() logic.
// ============================================================

__global__ void preprocessKernel(
    const float* __restrict__ pos_ws_xyz,      // [N*3]
    const float* __restrict__ scales_xyz,      // [N*3]
    const float* __restrict__ rotations_wxyz,  // [N*4] (w,x,y,z)
    const float* __restrict__ opacity_scene,   // [N]
    int num_splats,
    Mat4f view,
    Mat4f proj,
    int width,
    int height,
    float* __restrict__ means2D,               // [N*2]
    float* __restrict__ conic3D,               // [N*3] (inv00, inv01, inv11)
    float* __restrict__ opacities,             // [N]
    float* __restrict__ radii_px,              // [N] (<=0 means culled)
    float* __restrict__ depths,                // [N]
    int* __restrict__ gaussian_ids             // [N]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_splats) return;

    gaussian_ids[idx] = idx;
    float opacity = opacity_scene ? opacity_scene[idx] : 1.0f;
    opacities[idx] = opacity;

    float px = pos_ws_xyz[idx * 3 + 0];
    float py = pos_ws_xyz[idx * 3 + 1];
    float pz = pos_ws_xyz[idx * 3 + 2];

    float sx3 = scales_xyz[idx * 3 + 0];
    float sy3 = scales_xyz[idx * 3 + 1];
    float sz3 = scales_xyz[idx * 3 + 2];

    float qw = rotations_wxyz[idx * 4 + 0];
    float qx = rotations_wxyz[idx * 4 + 1];
    float qy = rotations_wxyz[idx * 4 + 2];
    float qz = rotations_wxyz[idx * 4 + 3];

    // Center: clip = proj * view * [p,1]
    float4 center_ws = make_float4(px, py, pz, 1.0f);
    float4 center_vs = mulMat4Vec4(view, center_ws);
    float4 center_cs = mulMat4Vec4(proj, center_vs);

    // Match CPU: if center_cs.w <= 0 -> behind camera
    if (!(center_cs.w > 0.0f) || !isfinite(center_cs.w)) {
        means2D[idx * 2 + 0] = 0.0f;
        means2D[idx * 2 + 1] = 0.0f;
        conic3D[idx * 3 + 0] = 0.0f;
        conic3D[idx * 3 + 1] = 0.0f;
        conic3D[idx * 3 + 2] = 0.0f;
        radii_px[idx] = -1.0f;
        depths[idx] = 0.0f;
        opacities[idx] = 0.0f;
        return;
    }

    float inv_w = 1.0f / center_cs.w;
    float ndc_x = center_cs.x * inv_w;
    float ndc_y = center_cs.y * inv_w;
    float ndc_z = center_cs.z * inv_w;

    // Match CPU's optional Z reject
    if (ndc_z < -1.5f || ndc_z > 1.5f || !isfinite(ndc_z)) {
        means2D[idx * 2 + 0] = 0.0f;
        means2D[idx * 2 + 1] = 0.0f;
        conic3D[idx * 3 + 0] = 0.0f;
        conic3D[idx * 3 + 1] = 0.0f;
        conic3D[idx * 3 + 2] = 0.0f;
        radii_px[idx] = -1.0f;
        depths[idx] = 0.0f;
        opacities[idx] = 0.0f;
        return;
    }

    float2 center_px = ndcToPixelEllipse(ndc_x, ndc_y, width, height);

    // Rotation columns
    float3 c0, c1, c2;
    quatToMat3Cols(qw, qx, qy, qz, c0, c1, c2);

    // Principal axes in world space (columns scaled)
    float3 axes_ws[3];
    axes_ws[0] = make_float3(c0.x * sx3, c0.y * sx3, c0.z * sx3);
    axes_ws[1] = make_float3(c1.x * sy3, c1.y * sy3, c1.z * sy3);
    axes_ws[2] = make_float3(c2.x * sz3, c2.y * sz3, c2.z * sz3);

    float2 axes_2d[3];
    float max_len2 = 0.0f;

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        float4 end_ws = make_float4(px + axes_ws[i].x, py + axes_ws[i].y, pz + axes_ws[i].z, 1.0f);
        float4 end_vs = mulMat4Vec4(view, end_ws);
        float4 end_cs = mulMat4Vec4(proj, end_vs);

        // Match CPU: if end_cs.w <= 0, push w to small positive to avoid div0
        if (!(end_cs.w > 0.0f) || !isfinite(end_cs.w)) {
            end_cs.w = 1e-3f;
        }

        float inv_end_w = 1.0f / end_cs.w;
        float end_ndc_x = end_cs.x * inv_end_w;
        float end_ndc_y = end_cs.y * inv_end_w;
        float2 end_px = ndcToPixelEllipse(end_ndc_x, end_ndc_y, width, height);

        float2 a = make_float2(end_px.x - center_px.x, end_px.y - center_px.y);
        axes_2d[i] = a;
        float len2 = a.x * a.x + a.y * a.y;
        if (len2 > max_len2) max_len2 = len2;
    }

    if (!(max_len2 > 1e-8f) || !isfinite(max_len2)) {
        means2D[idx * 2 + 0] = 0.0f;
        means2D[idx * 2 + 1] = 0.0f;
        conic3D[idx * 3 + 0] = 0.0f;
        conic3D[idx * 3 + 1] = 0.0f;
        conic3D[idx * 3 + 2] = 0.0f;
        radii_px[idx] = -1.0f;
        depths[idx] = 0.0f;
        opacities[idx] = 0.0f;
        return;
    }

    // Build 2D covariance as sum outer products (pixels)
    float cov00 = 0.0f;
    float cov01 = 0.0f;
    float cov11 = 0.0f;

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        float ax = axes_2d[i].x;
        float ay = axes_2d[i].y;
        cov00 += ax * ax;
        cov01 += ax * ay;
        cov11 += ay * ay;
    }

    // Regularization: eps = 1e-3 * trace + 1e-6
    float trace = cov00 + cov11;
    float eps = 1e-3f * trace + 1e-6f;
    cov00 += eps;
    cov11 += eps;

    float det = cov00 * cov11 - cov01 * cov01;
    if (!(det > 0.0f) || !isfinite(det)) {
        means2D[idx * 2 + 0] = 0.0f;
        means2D[idx * 2 + 1] = 0.0f;
        conic3D[idx * 3 + 0] = 0.0f;
        conic3D[idx * 3 + 1] = 0.0f;
        conic3D[idx * 3 + 2] = 0.0f;
        radii_px[idx] = -1.0f;
        depths[idx] = 0.0f;
        opacities[idx] = 0.0f;
        return;
    }

    float inv_det = 1.0f / det;
    float inv00 = cov11 * inv_det;
    float inv01 = -cov01 * inv_det;
    float inv11 = cov00 * inv_det;

    float max_len = sqrtf(max_len2);
    float radius = max_len * 3.0f;
    // Capacity-aware clamp: ensures per-splat duplicates stay within pre-allocated budget.
    // This prevents out-of-bounds writes later in tile binning.
    radius = clampf(radius, 1.0f, kMaxRadiusPxForTileBudget);

    means2D[idx * 2 + 0] = center_px.x;
    means2D[idx * 2 + 1] = center_px.y;
    conic3D[idx * 3 + 0] = inv00;
    conic3D[idx * 3 + 1] = inv01;
    conic3D[idx * 3 + 2] = inv11;
    radii_px[idx] = radius;
    depths[idx] = ndc_z * 0.5f + 0.5f;
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
    radius = fminf(radius, kMaxRadiusPxForTileBudget);
    float depth = depths[idx];

    int tile_x_min = max(0, (int)((sx - radius) / float(kTileSizePx)));
    int tile_x_max = min(num_tiles_x - 1, (int)((sx + radius) / float(kTileSizePx)));
    int tile_y_min = max(0, (int)((sy - radius) / float(kTileSizePx)));
    int tile_y_max = min(num_tiles_y - 1, (int)((sy + radius) / float(kTileSizePx)));

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
                                         const int* __restrict__ run_lengths,
                                         int run_capacity,
                                         int num_tiles,
                                         int* __restrict__ tile_offsets) {
    int run = blockIdx.x * blockDim.x + threadIdx.x;
    if (run < run_capacity) {
        int len = run_lengths[run];
        if (len > 0) {
            uint32_t tile = unique_tile_ids[run];
            if (tile < static_cast<uint32_t>(num_tiles)) {
                tile_offsets[tile] = run_offsets[run];
            }
        }
    }
}

// Set the sentinel tail offset
__global__ void setTileOffsetsTailKernel(int* tile_offsets, int num_tiles, int total_duplicates) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        tile_offsets[num_tiles] = total_duplicates;
    }
}

// Set tile_offsets[num_tiles] from device-side computed total duplicates.
__global__ void setTileOffsetsTailFromDeviceKernel(int* tile_offsets, int num_tiles, const int* total_duplicates_device) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        tile_offsets[num_tiles] = total_duplicates_device ? total_duplicates_device[0] : 0;
    }
}

// Initialize duplicate buffers with sentinel tile_id=0xFFFFFFFF so unused entries sort to the end.
__global__ void initDuplicateBuffersKernel(uint64_t* keys, int* values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        keys[idx] = packTileDepthKey(0xFFFFFFFFu, 0.0f);
        values[idx] = 0;
    }
}

// Count valid duplicates after sort (tile_id != 0xFFFFFFFF).
__global__ void countValidDuplicatesKernel(const uint32_t* tile_ids_sorted, int n, int* out_count) {
    __shared__ int s_sum;
    if (threadIdx.x == 0) s_sum = 0;
    __syncthreads();

    int local = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        local += (tile_ids_sorted[i] != 0xFFFFFFFFu);
    }

    atomicAdd(&s_sum, local);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(out_count, s_sum);
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
__device__ __forceinline__ void evalSH9_device(const glm::vec3& dir, float* sh)
{
    const float x = dir.x;
    const float y = dir.y;
    const float z = dir.z;

    constexpr float c0 = 0.28209479177387814f;
    constexpr float c1 = 0.4886025119029199f;
    constexpr float c2 = 1.0925484305920792f;
    constexpr float c3 = 0.31539156525252005f;
    constexpr float c4 = 0.5462742152960396f;

    sh[0] = c0;
    sh[1] = -c1 * y;
    sh[2] =  c1 * z;
    sh[3] = -c1 * x;
    sh[4] =  c2 * x * y;
    sh[5] = -c2 * y * z;
    sh[6] =  c3 * (3.0f * z * z - 1.0f);
    sh[7] = -c2 * x * z;
    sh[8] =  c4 * (x * x - y * y);
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
    float inv_len = rsqrtf(len_sq + 1e-8f);
    view_x *= inv_len;
    view_y *= inv_len;
    view_z *= inv_len;

    // Evaluate SH basis functions with float view direction (l≤2, 9 basis)
    float sh[9];
    evalSH9_device(glm::vec3(view_x, view_y, view_z), sh);

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
    if (d_total_duplicates_device_) { cudaFree(d_total_duplicates_device_); d_total_duplicates_device_ = nullptr; }
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
    if (d_scale_) { cudaFree(d_scale_); d_scale_ = nullptr; }
    if (d_rotation_) { cudaFree(d_rotation_); d_rotation_ = nullptr; }
    if (d_opacity_scene_) { cudaFree(d_opacity_scene_); d_opacity_scene_ = nullptr; }
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

    // Conservative fixed-capacity policy: do not rely on per-frame computed total_duplicates.
    // total_duplicates is treated as a lower bound (e.g., callers may pass 0 or 1).
    size_t reserved_duplicates = static_cast<size_t>(num_splats) * static_cast<size_t>(kDuplicateReserveFactor);
    if (reserved_duplicates == 0) reserved_duplicates = 1;
    size_t duplicate_capacity = (total_duplicates > reserved_duplicates) ? total_duplicates : reserved_duplicates;

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

    // Duplicate arrays for sort (capacity-managed)
    if (duplicate_capacity > sort_buffer_capacity_) {
        if (d_keys_) cudaFree(d_keys_);
        if (d_keys_sorted_) cudaFree(d_keys_sorted_);
        if (d_values_) cudaFree(d_values_);
        if (d_values_sorted_) cudaFree(d_values_sorted_);

        CUDA_CHECK(cudaMalloc(&d_keys_, duplicate_capacity * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_sorted_, duplicate_capacity * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_values_, duplicate_capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_values_sorted_, duplicate_capacity * sizeof(int)));

        sort_buffer_capacity_ = duplicate_capacity;
    } else {
        if (!d_keys_) CUDA_CHECK(cudaMalloc(&d_keys_, sort_buffer_capacity_ * sizeof(uint64_t)));
        if (!d_keys_sorted_) CUDA_CHECK(cudaMalloc(&d_keys_sorted_, sort_buffer_capacity_ * sizeof(uint64_t)));
        if (!d_values_) CUDA_CHECK(cudaMalloc(&d_values_, sort_buffer_capacity_ * sizeof(int)));
        if (!d_values_sorted_) CUDA_CHECK(cudaMalloc(&d_values_sorted_, sort_buffer_capacity_ * sizeof(int)));
    }

    // RLE buffers (size <= duplicate_capacity)
    size_t rle_needed = duplicate_capacity;
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

    if (!d_num_runs_device_) CUDA_CHECK(cudaMalloc(&d_num_runs_device_, sizeof(int)));
    if (!d_total_duplicates_device_) CUDA_CHECK(cudaMalloc(&d_total_duplicates_device_, sizeof(int)));

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
    gs::NvtxRange nvtx_range("Build_Tile_Binning");
    
    dim3 block256(256);
    dim3 grid256((num_splats + 255) / 256);

    // Conservative pre-allocation: avoid any per-frame D2H sync to compute total_duplicates.
    // We will build a fixed-capacity duplicate list and use sentinel keys for unused entries.
    const size_t duplicate_capacity = std::max<size_t>(1, static_cast<size_t>(num_splats) * static_cast<size_t>(kDuplicateReserveFactor));

    // Ensure buffers exist at conservative capacity.
    if (!ensureSortBuffers(num_splats, duplicate_capacity)) return false;
    if (!ensureTileBuffers(duplicate_capacity, static_cast<size_t>(num_tiles) + 1)) return false;

    // Caller-visible total_duplicates is produced asynchronously on device (no D2H sync here).
    total_duplicates = 0;

    cudaStream_t stream = 0; // default stream; all work remains async from the CPU perspective.

    // Define the CUB temp buffer allocation helper (used across multiple phases)
    auto ensureCubTemp = [&](size_t bytes) -> bool {
        if (bytes > cub_temp_bytes_) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
            if (d_cub_temp_) CUDA_CHECK(cudaFreeAsync(d_cub_temp_, stream));
            cudaError_t err = cudaMallocAsync(&d_cub_temp_, bytes, stream);
#else
            if (d_cub_temp_) cudaFree(d_cub_temp_);
            cudaError_t err = cudaMalloc(&d_cub_temp_, bytes);
#endif
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaMalloc for CUB temp failed: %s\n", cudaGetErrorString(err));
                return false;
            }
            cub_temp_bytes_ = bytes;
        }
        return true;
    };

    // Phase 0: initialize duplicate buffers with sentinel values so unused entries are safe.
    {
        gs::NvtxRange nvtx_phase0("Init_Duplicate_Buffers");
        int n = static_cast<int>(duplicate_capacity);
        dim3 gridInit((n + 255) / 256);
        initDuplicateBuffersKernel<<<gridInit, block256, 0, stream>>>(d_keys_, d_values_, n);
        CUDA_CHECK(cudaGetLastError());
    }

    // Phase 1: tile touch counts
    {
        gs::NvtxRange nvtx_phase1("Compute_Tile_Touch_Count");
        computeTileTouchCountKernel<<<grid256, block256, 0, stream>>>(
            d_means2D_, d_radii_px_, num_splats, num_tiles_x, num_tiles_y, d_num_tiles_touched_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_DEBUG_SYNC();
    }

    // Phase 2: exclusive scan for duplicate offsets
    {
        gs::NvtxRange nvtx_phase2("Exclusive_Scan_Duplicates");
        size_t scan_temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, d_num_tiles_touched_, d_dup_offsets_, num_splats, stream);

        if (!ensureCubTemp(scan_temp_bytes)) return false;
        cub::DeviceScan::ExclusiveSum(d_cub_temp_, scan_temp_bytes, d_num_tiles_touched_, d_dup_offsets_, num_splats, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_DEBUG_SYNC();
    }

    // Phase 5: emit duplicate keys/values
    {
        gs::NvtxRange nvtx_phase5("Emit_Duplicate_Keys");
        emitDuplicateKeysKernel<<<grid256, block256, 0, stream>>>(
            d_means2D_, d_radii_px_, d_depths_, num_splats, num_tiles_x, num_tiles_y,
            d_dup_offsets_, d_keys_, d_values_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_DEBUG_SYNC();
    }

    // Phase 6: radix sort (tile_id primary, depth secondary)
    {
        gs::NvtxRange nvtx_phase6("Radix_Sort_Keys");
        size_t sort_temp_bytes = 0;
        int n = static_cast<int>(duplicate_capacity);
        cub::DeviceRadixSort::SortPairs(nullptr, sort_temp_bytes, d_keys_, d_keys_sorted_, d_values_, d_values_sorted_, n, 0, 64, stream);
        if (!ensureCubTemp(sort_temp_bytes)) return false;
        cub::DeviceRadixSort::SortPairs(d_cub_temp_, sort_temp_bytes, d_keys_, d_keys_sorted_, d_values_, d_values_sorted_, n, 0, 64, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_DEBUG_SYNC();
    }

    // Phase 7: extract tile ids for RLE
    {
        gs::NvtxRange nvtx_phase7("Extract_Tile_IDs");
        int n = static_cast<int>(duplicate_capacity);
        dim3 gridDup((n + 255) / 256);
        extractTileIdsKernel<<<gridDup, block256, 0, stream>>>(d_keys_sorted_, d_tile_ids_sorted_, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_DEBUG_SYNC();
    }

    // Phase 8: run-length encode tile ids
    {
        gs::NvtxRange nvtx_phase8("RLE_Tile_IDs");
        // Initialize outputs so we can safely scan/scatter with fixed capacity.
        CUDA_CHECK(cudaMemsetAsync(d_unique_tile_ids_, 0xFF, duplicate_capacity * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMemsetAsync(d_run_lengths_, 0, duplicate_capacity * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_run_offsets_, 0, duplicate_capacity * sizeof(int), stream));

        size_t rle_temp_bytes = 0;
        int n = static_cast<int>(duplicate_capacity);
        cub::DeviceRunLengthEncode::Encode(nullptr, rle_temp_bytes,
            d_tile_ids_sorted_, d_unique_tile_ids_, d_run_lengths_, d_num_runs_device_, n, stream);
        if (!ensureCubTemp(rle_temp_bytes)) return false;
        cub::DeviceRunLengthEncode::Encode(d_cub_temp_, rle_temp_bytes,
            d_tile_ids_sorted_, d_unique_tile_ids_, d_run_lengths_, d_num_runs_device_, n, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_DEBUG_SYNC();
    }

    // Phase 9: exclusive scan of run lengths -> run offsets
    {
        gs::NvtxRange nvtx_phase9("Scan_Run_Offsets");
        size_t run_scan_temp_bytes = 0;
        int n = static_cast<int>(duplicate_capacity);
        cub::DeviceScan::ExclusiveSum(nullptr, run_scan_temp_bytes, d_run_lengths_, d_run_offsets_, n, stream);
        if (!ensureCubTemp(run_scan_temp_bytes)) return false;
        cub::DeviceScan::ExclusiveSum(d_cub_temp_, run_scan_temp_bytes, d_run_lengths_, d_run_offsets_, n, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_DEBUG_SYNC();
    }

    // Phase 9.5: compute valid duplicate count on device (tile_id != sentinel)
    {
        gs::NvtxRange nvtx_count("Count_Valid_Duplicates");
        CUDA_CHECK(cudaMemsetAsync(d_total_duplicates_device_, 0, sizeof(int), stream));
        int n = static_cast<int>(duplicate_capacity);
        int blocks = (n + 255) / 256;
        blocks = min(blocks, 1024);
        countValidDuplicatesKernel<<<blocks, 256, 0, stream>>>(d_tile_ids_sorted_, n, d_total_duplicates_device_);
        CUDA_CHECK(cudaGetLastError());
    }

    // Phase 10: scatter run offsets into tile_offsets
    {
        gs::NvtxRange nvtx_phase10("Scatter_Tile_Offsets");
        CUDA_CHECK(cudaMemsetAsync(d_tile_offsets_, 0xFF, (num_tiles + 1) * sizeof(int), stream)); // set to -1
        int run_n = static_cast<int>(duplicate_capacity);
        dim3 gridRuns((run_n + 255) / 256);
        scatterTileOffsetsKernel<<<gridRuns, block256, 0, stream>>>(
            d_unique_tile_ids_, d_run_offsets_, d_run_lengths_, run_n, num_tiles, d_tile_offsets_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_DEBUG_SYNC();

        setTileOffsetsTailFromDeviceKernel<<<1, 1, 0, stream>>>(d_tile_offsets_, num_tiles, d_total_duplicates_device_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_DEBUG_SYNC();

        fillTileOffsetGapsKernel<<<1, 1, 0, stream>>>(d_tile_offsets_, num_tiles);
        CUDA_CHECK(cudaGetLastError());
        CUDA_DEBUG_SYNC();

        // Copy sorted splat indices into tile_splat_list
        CUDA_CHECK(cudaMemcpyAsync(d_tile_splat_list_, d_values_sorted_, duplicate_capacity * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    }

    // Debug validation requires a host-side expected tail and would introduce a D2H sync.
    // Keep the tile_offsets tail correct on device; skip D2H validation in this async path.
    return true;
}

bool Rasterizer::uploadSceneData(const std::vector<gs::GaussianSplat>& gaussians) {
    gs::NvtxRange nvtx_range("Upload_Scene_Data");
    
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
    std::vector<float> h_scale(num * 3);
    std::vector<float> h_rotation(num * 4); // (w,x,y,z)
    std::vector<float> h_opacity(num);
    std::vector<float> h_dc_colors(num * 3);
    std::vector<float> h_sh_coeffs(num * 27);  // 3 channels * 9 basis functions (Degree 2)

    for (int i = 0; i < num; ++i) {
        const gs::GaussianSplat& g = gaussians[i];
        
        // Unpack position to avoid GLM padding/alignment issues
        h_pos_ws[i * 3 + 0] = g.position_ws.x;
        h_pos_ws[i * 3 + 1] = g.position_ws.y;
        h_pos_ws[i * 3 + 2] = g.position_ws.z;

        h_scale[i * 3 + 0] = g.scale.x;
        h_scale[i * 3 + 1] = g.scale.y;
        h_scale[i * 3 + 2] = g.scale.z;

        h_rotation[i * 4 + 0] = g.rotation.w;
        h_rotation[i * 4 + 1] = g.rotation.x;
        h_rotation[i * 4 + 2] = g.rotation.y;
        h_rotation[i * 4 + 3] = g.rotation.z;

        h_opacity[i] = g.opacity;
        
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
        if (d_scale_) cudaFree(d_scale_);
        if (d_rotation_) cudaFree(d_rotation_);
        if (d_opacity_scene_) cudaFree(d_opacity_scene_);
        if (d_sh_coeffs_) cudaFree(d_sh_coeffs_);
        if (d_dc_colors_) cudaFree(d_dc_colors_);

        CUDA_CHECK(cudaMalloc(&d_pos_ws_, num * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scale_, num * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rotation_, num * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_opacity_scene_, num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sh_coeffs_, num * 27 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dc_colors_, num * 3 * sizeof(float)));
        scene_splat_capacity_ = num;
    }

    // Upload to device
    {
        gs::NvtxRange nvtx_h2d("Upload_Scene_H2D");
        printf("  Uploading pos_ws: %zu bytes\n", num * 3 * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_pos_ws_, h_pos_ws.data(), num * 3 * sizeof(float), cudaMemcpyHostToDevice));
        printf("  Uploading scale: %zu bytes\n", num * 3 * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_scale_, h_scale.data(), num * 3 * sizeof(float), cudaMemcpyHostToDevice));
        printf("  Uploading rotation: %zu bytes\n", num * 4 * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_rotation_, h_rotation.data(), num * 4 * sizeof(float), cudaMemcpyHostToDevice));
        printf("  Uploading opacity: %zu bytes\n", num * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_opacity_scene_, h_opacity.data(), num * sizeof(float), cudaMemcpyHostToDevice));
        printf("  Uploading sh_coeffs: %zu bytes\n", num * 27 * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_sh_coeffs_, h_sh_coeffs.data(), num * 27 * sizeof(float), cudaMemcpyHostToDevice));
        printf("  Uploading dc_colors: %zu bytes\n", num * 3 * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_dc_colors_, h_dc_colors.data(), num * 3 * sizeof(float), cudaMemcpyHostToDevice));
    }

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
    gs::NvtxRange nvtx_render("Render_CUDA");
    
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
        gs::NvtxRange nvtx_sh("Eval_SH_Color");
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
        gs::NvtxRange nvtx_tile("GPU_Tile_Binning_Sort");
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

    {
        gs::NvtxRange nvtx_splat("Splat_Kernel");
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
            gs::NvtxRange nvtx_d2h("Download_Output_D2H");
            ScopedTimer timer("download_d2h", &t_d2h_ms);
            CUDA_CHECK(cudaMemcpy(output_image, d_output_, 
                                 width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Stop timer after copy (implies kernel completion)
        kernel_timer.stop();
    }

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
        
        // printf("  [PBO] Launching evalSHColorKernel: grid=(%u, 1, 1), block=(%u, 1, 1), num_splats=%d\n",
        //        grid.x, block.x, num_splats);
        
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
        // printf("  ✓ [PBO] evalSHColorKernel enqueued (stream-ordered execution)\n");
        // fflush(stdout);
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

bool Rasterizer::render_cuda_to_rgba8_device_v2(
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camera_pos,
    int width,
    int height,
    unsigned char* out_rgba8_device
) {
    gs::NvtxRange nvtx_frame("GPU_Preprocess_And_Render");

    if (!gaussians_ptr_) {
        fprintf(stderr, "Error: Scene data not uploaded. Call uploadSceneData() first.\n");
        return false;
    }
    if (!d_pos_ws_ || !d_scale_ || !d_rotation_ || !d_opacity_scene_) {
        fprintf(stderr, "Error: Missing scene buffers (pos/scale/rotation/opacity).\n");
        return false;
    }

    int num_splats = static_cast<int>(scene_num_splats_);
    if (num_splats <= 0) {
        fprintf(stderr, "Error: No scene splats uploaded.\n");
        return false;
    }

    if (!ensureFrameBuffers(num_splats, width, height)) return false;

    // 1) Preprocess on GPU: project + ellipse params
    {
        gs::NvtxRange nvtx_pre("GPU_Preprocess_Kernel");
        Mat4f v = toMat4f(view);
        Mat4f p = toMat4f(proj);
        dim3 block(256);
        dim3 grid((num_splats + 255) / 256);
        preprocessKernel<<<grid, block>>>(
            d_pos_ws_,
            d_scale_,
            d_rotation_,
            d_opacity_scene_,
            num_splats,
            v,
            p,
            width,
            height,
            d_means2D_,
            d_conic3D_,
            d_opacities_,
            d_radii_px_,
            d_depths_,
            d_gaussian_ids_
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA preprocess launch error: %s\n", cudaGetErrorString(err));
            return false;
        }
    }

    // 2) SH colors on GPU
    {
        gs::NvtxRange nvtx_sh("GPU_SH_Eval");
        dim3 block(256);
        dim3 grid((num_splats + 255) / 256);
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
            fprintf(stderr, "CUDA SH launch error: %s\n", cudaGetErrorString(err));
            return false;
        }
    }

    // 3) Tile binning & sorting (same as v1)
    int num_tiles_x = (width + 15) / 16;
    int num_tiles_y = (height + 15) / 16;
    int num_tiles = num_tiles_x * num_tiles_y;
    size_t total_duplicates = 0;

    {
        gs::NvtxRange nvtx_tiles("GPU_Tile_Binning");
        if (!buildTileBinning(num_splats, num_tiles_x, num_tiles_y, num_tiles, total_duplicates)) {
            fprintf(stderr, "Tile binning failed (v2 path)\n");
            return false;
        }
    }

    // 4) Render tiles + pack to RGBA8
    {
        gs::NvtxRange nvtx_render("GPU_Render_Tiles");
        dim3 block(16, 16);
        dim3 grid(num_tiles_x, num_tiles_y);
        splatKernel<<<grid, block>>>(
            d_means2D_, d_conic3D_, d_colors_, d_opacities_,
            d_tile_splat_list_, d_tile_offsets_,
            num_tiles_x,
            width, height,
            d_output_
        );
    }

    {
        gs::NvtxRange nvtx_pack("GPU_Pack_RGBA8");
        dim3 p(16, 16);
        dim3 g((width + 15) / 16, (height + 15) / 16);
        packToRGBA8<<<g, p>>>(d_output_, out_rgba8_device, width, height);
    }

    return true;
}

} // namespace CudaRasterizer
