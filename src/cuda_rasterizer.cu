#include "cuda_rasterizer.h"
#include "gs/gaussian.h"
#include "gs/screen_splat.h"
#include "gs/sh_color.h"
#include "gs/profiler.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <glm/glm.hpp>

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

namespace CudaRasterizer {

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
    const float* __restrict__ pos_ws_xyz,      // [N*3] world-space positions (x,y,z interleaved)
    const float* __restrict__ dc_colors,       // [N*3] DC colors (0-order SH)
    const float* __restrict__ sh_coeffs,       // [N*27] SH coefficients
    float cam_x, float cam_y, float cam_z,     // Camera position components
    int num_splats,
    float* __restrict__ out_colors             // [N*3] output RGB colors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_splats) return;

    // Compute view direction (extract position as float3)
    float pos_x = pos_ws_xyz[idx * 3 + 0];
    float pos_y = pos_ws_xyz[idx * 3 + 1];
    float pos_z = pos_ws_xyz[idx * 3 + 2];
    
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
    float r = dc_colors[idx * 3 + 0];
    float g = dc_colors[idx * 3 + 1];
    float b = dc_colors[idx * 3 + 2];

    // Add higher-order SH contributions (indices 1..8, skip DC at 0)
    #pragma unroll 8
    for (int basis = 1; basis < 9; ++basis) {
        float w = sh[basis];
        r += sh_coeffs[idx * 27 + 0 * 9 + basis] * w;      // R coefficients at [0..8]
        g += sh_coeffs[idx * 27 + 1 * 9 + basis] * w;      // G coefficients at [9..17]
        b += sh_coeffs[idx * 27 + 2 * 9 + basis] * w;      // B coefficients at [18..26]
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
    , d_output_(nullptr)
    , d_tile_splat_list_(nullptr)
    , d_tile_offsets_(nullptr)
    , tile_splat_list_capacity_(0)
    , tile_offsets_capacity_(0)
    , num_splats_(0)
    , width_(0)
    , height_(0)
{
}

Rasterizer::~Rasterizer() {
    free();
}

bool Rasterizer::allocateBuffers(int num_splats, int width, int height) {
    freeBuffers(); // Free old buffers if any

    num_splats_ = num_splats;
    width_ = width;
    height_ = height;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_means2D_, num_splats * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conic3D_, num_splats * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colors_, num_splats * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_opacities_, num_splats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_, width * height * 3 * sizeof(float)));

    printf("✅ Allocated CUDA buffers: %d splats, %dx%d image\n", num_splats, width, height);
    return true;
}

void Rasterizer::freeBuffers() {
    if (d_means2D_) { cudaFree(d_means2D_); d_means2D_ = nullptr; }
    if (d_conic3D_) { cudaFree(d_conic3D_); d_conic3D_ = nullptr; }
    if (d_colors_) { cudaFree(d_colors_); d_colors_ = nullptr; }
    if (d_opacities_) { cudaFree(d_opacities_); d_opacities_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    if (d_tile_splat_list_) { cudaFree(d_tile_splat_list_); d_tile_splat_list_ = nullptr; tile_splat_list_capacity_ = 0; }
    if (d_tile_offsets_) { cudaFree(d_tile_offsets_); d_tile_offsets_ = nullptr; tile_offsets_capacity_ = 0; }
}

void Rasterizer::free() {
    freeBuffers();
    // Free scene-static data
    if (d_pos_ws_) { cudaFree(d_pos_ws_); d_pos_ws_ = nullptr; }
    if (d_sh_coeffs_) { cudaFree(d_sh_coeffs_); d_sh_coeffs_ = nullptr; }
    if (d_dc_colors_) { cudaFree(d_dc_colors_); d_dc_colors_ = nullptr; }
    scene_splat_capacity_ = 0;
    scene_num_splats_ = 0;
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

bool Rasterizer::uploadSceneData(const std::vector<gs::GaussianSplat>& gaussians) {
    int num = static_cast<int>(gaussians.size());
    if (num == 0) {
        fprintf(stderr, "Error: No gaussians to upload\n");
        return false;
    }

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

    printf("🚀 CUDA V2 Tile-based Render: %d splats, %dx%d image\n", num_splats, width, height);

    double t_sh_ms = 0.0;
    double t_tile_ms = 0.0;
    double t_h2d_ms = 0.0;
    double t_d2h_ms = 0.0;
    float gpu_total_ms = 0.0f;

    // Allocate device buffers
    if (!allocateBuffers(num_splats, width, height)) {
        fprintf(stderr, "❌ Failed to allocate CUDA buffers\n");
        return false;
    }

    // Prepare host arrays for upload (convert ScreenSplat to SoA format, skip colors—GPU will compute)
    printf("🔄 Preparing %d splats for upload (colors will be computed on GPU)...\n", num_splats);
    fflush(stdout);
    
    std::vector<float> h_means2D(num_splats * 2);
    std::vector<float> h_conic3D(num_splats * 3);
    std::vector<float> h_colors(num_splats * 3);  // Will be filled by GPU kernel
    std::vector<float> h_opacities(num_splats);
    
    // Pack per-splat scene data (position, DC, SH) in screen_splats order for GPU
    std::vector<float> h_pos_ws_render(num_splats * 3);
    std::vector<float> h_dc_colors_render(num_splats * 3);
    std::vector<float> h_sh_coeffs_render(num_splats * 27);

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
            
            h_means2D[i * 2 + 0] = sp.sx;
            h_means2D[i * 2 + 1] = sp.sy;
            
            h_conic3D[i * 3 + 0] = sp.cov_inv[0][0];
            h_conic3D[i * 3 + 1] = sp.cov_inv[0][1];
            h_conic3D[i * 3 + 2] = sp.cov_inv[1][1];
            
            h_opacities[i] = sp.src->opacity;
            
            // Pack scene data in render order (following screen_splats order)
            const gs::GaussianSplat* g = sp.src;
            h_pos_ws_render[i * 3 + 0] = g->position_ws.x;
            h_pos_ws_render[i * 3 + 1] = g->position_ws.y;
            h_pos_ws_render[i * 3 + 2] = g->position_ws.z;
            
            h_dc_colors_render[i * 3 + 0] = g->dc_color.r;
            h_dc_colors_render[i * 3 + 1] = g->dc_color.g;
            h_dc_colors_render[i * 3 + 2] = g->dc_color.b;
            
            // Pack SH coefficients
            int original_n_basis = static_cast<int>(g->sh_color.coeffs.size()) / 3;
            int used_basis = std::min(original_n_basis, 9);
            for (int b = 0; b < 9; ++b) {
                if (b < used_basis) {
                    h_sh_coeffs_render[i * 27 + 0 * 9 + b] = g->sh_color.coeffs[0 * original_n_basis + b]; // R
                    h_sh_coeffs_render[i * 27 + 1 * 9 + b] = g->sh_color.coeffs[1 * original_n_basis + b]; // G
                    h_sh_coeffs_render[i * 27 + 2 * 9 + b] = g->sh_color.coeffs[2 * original_n_basis + b]; // B
                } else {
                    h_sh_coeffs_render[i * 27 + 0 * 9 + b] = 0.0f;
                    h_sh_coeffs_render[i * 27 + 1 * 9 + b] = 0.0f;
                    h_sh_coeffs_render[i * 27 + 2 * 9 + b] = 0.0f;
                }
            }
        }
    }

    printf("✅ Screen data packed. Computing SH colors on GPU...\n");
    fflush(stdout);

    // Allocate temporary device buffers for render-order scene data
    float* d_pos_ws_render = nullptr;
    float* d_dc_colors_render = nullptr;
    float* d_sh_coeffs_render = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_pos_ws_render, num_splats * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dc_colors_render, num_splats * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh_coeffs_render, num_splats * 27 * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_pos_ws_render, h_pos_ws_render.data(), 
                         num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dc_colors_render, h_dc_colors_render.data(), 
                         num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sh_coeffs_render, h_sh_coeffs_render.data(), 
                         num_splats * 27 * sizeof(float), cudaMemcpyHostToDevice));

    // Launch GPU SH evaluation kernel (evaluates colors for all splats used in this render)
    {
        ScopedTimer timer("gpu_sh_eval", &t_sh_ms);
        dim3 block(256);
        dim3 grid((num_splats + 255) / 256);
        
        printf("  Launching evalSHColorKernel: grid=(%u, 1, 1), block=(%u, 1, 1), num_splats=%d\n",
               grid.x, block.x, num_splats);
        
        evalSHColorKernel<<<grid, block>>>(
            d_pos_ws_render,
            d_dc_colors_render,
            d_sh_coeffs_render,
            camera_pos.x, camera_pos.y, camera_pos.z,
            num_splats,
            d_colors_
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Synchronize to ensure kernel completes
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel execution error: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        printf("  ✓ evalSHColorKernel completed successfully\n");
        fflush(stdout);
    }

    printf("✅ SH evaluation complete (GPU)\n");
    fflush(stdout);

    // Build tile splat lists on CPU (linear layout)
    printf("🔨 Building tile splat lists...\n");
    fflush(stdout);

    int num_tiles_x = (width + 15) / 16;
    int num_tiles_y = (height + 15) / 16;
    int num_tiles = num_tiles_x * num_tiles_y;

    std::vector<int> h_tile_splat_indices;
    std::vector<int> h_tile_offsets;
    size_t total_tile_refs = 0;

    {
        ScopedTimer timer("tile_list_build", &t_tile_ms);

        std::vector<int> tile_counts(num_tiles, 0);

        // First pass: count how many splats touch each tile
        for (int idx = 0; idx < num_splats; ++idx) {
            const gs::ScreenSplat& sp = screen_splats[idx];

            float radius = sp.radius_px;
            int tile_x_min = std::max(0, static_cast<int>((sp.sx - radius) / 16.0f));
            int tile_x_max = std::min(num_tiles_x - 1, static_cast<int>((sp.sx + radius) / 16.0f));
            int tile_y_min = std::max(0, static_cast<int>((sp.sy - radius) / 16.0f));
            int tile_y_max = std::min(num_tiles_y - 1, static_cast<int>((sp.sy + radius) / 16.0f));

            for (int ty = tile_y_min; ty <= tile_y_max; ++ty) {
                for (int tx = tile_x_min; tx <= tile_x_max; ++tx) {
                    int tile_id = ty * num_tiles_x + tx;
                    ++tile_counts[tile_id];
                }
            }
        }

        // Prefix sum: compute tile offsets (exclusive)
        h_tile_offsets.assign(num_tiles + 1, 0);
        for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
            h_tile_offsets[tile_id + 1] = h_tile_offsets[tile_id] + tile_counts[tile_id];
        }

        // Allocate flat index buffer and per-tile cursor for filling
        h_tile_splat_indices.assign(h_tile_offsets[num_tiles], 0);
        std::vector<int> tile_cursor(num_tiles, 0);

        // Second pass: fill indices in the same sorted order
        for (int idx = 0; idx < num_splats; ++idx) {
            const gs::ScreenSplat& sp = screen_splats[idx];

            float radius = sp.radius_px;
            int tile_x_min = std::max(0, static_cast<int>((sp.sx - radius) / 16.0f));
            int tile_x_max = std::min(num_tiles_x - 1, static_cast<int>((sp.sx + radius) / 16.0f));
            int tile_y_min = std::max(0, static_cast<int>((sp.sy - radius) / 16.0f));
            int tile_y_max = std::min(num_tiles_y - 1, static_cast<int>((sp.sy + radius) / 16.0f));

            for (int ty = tile_y_min; ty <= tile_y_max; ++ty) {
                for (int tx = tile_x_min; tx <= tile_x_max; ++tx) {
                    int tile_id = ty * num_tiles_x + tx;
                    int write_idx = h_tile_offsets[tile_id] + tile_cursor[tile_id]++;
                    h_tile_splat_indices[write_idx] = idx;
                }
            }
        }

        total_tile_refs = h_tile_splat_indices.size();
    }

    printf("✅ Built tile lists: %zu total splat references across %d tiles\n",
           h_tile_splat_indices.size(), num_tiles);
    fflush(stdout);

    // Allocate device memory for tile lists
    // Ensure tile buffers have capacity and upload
    if (!ensureTileBuffers(h_tile_splat_indices.size(), (size_t)(num_tiles + 1))) {
        fprintf(stderr, "Failed to ensure tile buffer capacity\n");
        return false;
    }

    // Upload data to GPU (note: d_colors_ is filled by GPU kernel, not uploaded here)
    printf("📤 Uploading data to GPU...\n");
    fflush(stdout);

    {
        ScopedTimer timer("upload_h2d", &t_h2d_ms);
        CUDA_CHECK(cudaMemcpy(d_means2D_, h_means2D.data(), 
                             num_splats * 2 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conic3D_, h_conic3D.data(), 
                             num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice));
        // d_colors_ is filled by GPU kernel, not uploaded here
        CUDA_CHECK(cudaMemcpy(d_opacities_, h_opacities.data(), 
                             num_splats * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tile_splat_list_, h_tile_splat_indices.data(),
                             h_tile_splat_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tile_offsets_, h_tile_offsets.data(),
                             (num_tiles + 1) * sizeof(int), cudaMemcpyHostToDevice));
    }

    printf("✅ Uploaded data to GPU\n");
    fflush(stdout);

    // Launch kernel
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
    
    // Free temporary render-order buffers
    cudaFree(d_pos_ws_render);
    cudaFree(d_dc_colors_render);
    cudaFree(d_sh_coeffs_render);

#if ENABLE_PROFILING
    double cpu_total_ms = t_sh_ms + t_tile_ms + t_h2d_ms + t_d2h_ms;
    std::printf("[Profile] stats: splats_total=%d, tile_refs=%zu, res=%dx%d\n",
                num_splats, total_tile_refs, width, height);
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

    // Allocate buffers and prepare SoA like in render_cuda
    if (!allocateBuffers(num_splats, width, height)) return false;

    std::vector<float> h_means2D(num_splats * 2);
    std::vector<float> h_conic3D(num_splats * 3);
    std::vector<float> h_colors(num_splats * 3);  // Will be filled by GPU kernel
    std::vector<float> h_opacities(num_splats);
    
    // Pack scene data in screen_splats order
    std::vector<float> h_pos_ws_render(num_splats * 3);
    std::vector<float> h_dc_colors_render(num_splats * 3);
    std::vector<float> h_sh_coeffs_render(num_splats * 27);

    for (int i = 0; i < num_splats; ++i) {
        const gs::ScreenSplat& sp = screen_splats[i];
        if (!sp.src) { fprintf(stderr, "Null src at %d\n", i); return false; }
        const gs::GaussianSplat* g = sp.src;
        
        h_means2D[i * 2 + 0] = sp.sx;
        h_means2D[i * 2 + 1] = sp.sy;
        h_conic3D[i * 3 + 0] = sp.cov_inv[0][0];
        h_conic3D[i * 3 + 1] = sp.cov_inv[0][1];
        h_conic3D[i * 3 + 2] = sp.cov_inv[1][1];
        h_opacities[i] = g->opacity;
        
        // Pack position
        h_pos_ws_render[i*3+0] = g->position_ws.x;
        h_pos_ws_render[i*3+1] = g->position_ws.y;
        h_pos_ws_render[i*3+2] = g->position_ws.z;
        
        // Pack DC color
        h_dc_colors_render[i*3+0] = g->dc_color.r;
        h_dc_colors_render[i*3+1] = g->dc_color.g;
        h_dc_colors_render[i*3+2] = g->dc_color.b;
        
        // Pack SH coefficients (9 basis, 3 channels)
        int original_n_basis = (int)g->sh_color.coeffs.size() / 3;
        for (int channel = 0; channel < 3; ++channel) {
            for (int b = 0; b < 9; ++b) {
                if (b < original_n_basis) {
                    h_sh_coeffs_render[i*27 + channel*9 + b] = g->sh_color.coeffs[channel * original_n_basis + b];
                } else {
                    h_sh_coeffs_render[i*27 + channel*9 + b] = 0.0f;
                }
            }
        }
    }

    // Upload render-order data to temporary device buffers
    float* d_pos_ws_render = nullptr;
    float* d_dc_colors_render = nullptr;
    float* d_sh_coeffs_render = nullptr;
    
    cudaMalloc(&d_pos_ws_render, num_splats * 3 * sizeof(float));
    cudaMalloc(&d_dc_colors_render, num_splats * 3 * sizeof(float));
    cudaMalloc(&d_sh_coeffs_render, num_splats * 27 * sizeof(float));
    
    cudaMemcpy(d_pos_ws_render, h_pos_ws_render.data(), num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dc_colors_render, h_dc_colors_render.data(), num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sh_coeffs_render, h_sh_coeffs_render.data(), num_splats * 27 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch GPU SH evaluation kernel with render-order buffers
    {
        dim3 block(256);
        dim3 grid((num_splats + 255) / 256);
        
        printf("  [PBO] Launching evalSHColorKernel: grid=(%u, 1, 1), block=(%u, 1, 1), num_splats=%d\n",
               grid.x, block.x, num_splats);
        
        evalSHColorKernel<<<grid, block>>>(
            d_pos_ws_render,
            d_dc_colors_render,
            d_sh_coeffs_render,
            camera_pos.x, camera_pos.y, camera_pos.z,
            num_splats,
            d_colors_
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel launch error (PBO): %s\n", cudaGetErrorString(err));
            cudaFree(d_pos_ws_render);
            cudaFree(d_dc_colors_render);
            cudaFree(d_sh_coeffs_render);
            return false;
        }
        
        // Synchronize to ensure kernel completes
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel execution error (PBO): %s\n", cudaGetErrorString(err));
            cudaFree(d_pos_ws_render);
            cudaFree(d_dc_colors_render);
            cudaFree(d_sh_coeffs_render);
            return false;
        }
        
        printf("  ✓ [PBO] evalSHColorKernel completed\n");
        fflush(stdout);
    }

    int num_tiles_x = (width + 15) / 16;
    int num_tiles_y = (height + 15) / 16;
    int num_tiles = num_tiles_x * num_tiles_y;
    std::vector<int> h_tile_splat_indices;
    std::vector<int> h_tile_offsets(num_tiles + 1, 0);

    // Count
    std::vector<int> tile_counts(num_tiles, 0);
    for (int idx = 0; idx < num_splats; ++idx) {
        const gs::ScreenSplat& sp = screen_splats[idx];
        float radius = sp.radius_px;
        int tx0 = std::max(0, (int)((sp.sx - radius) / 16.0f));
        int tx1 = std::min(num_tiles_x - 1, (int)((sp.sx + radius) / 16.0f));
        int ty0 = std::max(0, (int)((sp.sy - radius) / 16.0f));
        int ty1 = std::min(num_tiles_y - 1, (int)((sp.sy + radius) / 16.0f));
        for (int ty = ty0; ty <= ty1; ++ty)
            for (int tx = tx0; tx <= tx1; ++tx)
                ++tile_counts[ty * num_tiles_x + tx];
    }
    for (int t = 0; t < num_tiles; ++t) h_tile_offsets[t + 1] = h_tile_offsets[t] + tile_counts[t];
    h_tile_splat_indices.assign(h_tile_offsets[num_tiles], 0);
    std::vector<int> cursor(num_tiles, 0);
    for (int idx = 0; idx < num_splats; ++idx) {
        const gs::ScreenSplat& sp = screen_splats[idx];
        float radius = sp.radius_px;
        int tx0 = std::max(0, (int)((sp.sx - radius) / 16.0f));
        int tx1 = std::min(num_tiles_x - 1, (int)((sp.sx + radius) / 16.0f));
        int ty0 = std::max(0, (int)((sp.sy - radius) / 16.0f));
        int ty1 = std::min(num_tiles_y - 1, (int)((sp.sy + radius) / 16.0f));
        for (int ty = ty0; ty <= ty1; ++ty) {
            for (int tx = tx0; tx <= tx1; ++tx) {
                int tile_id = ty * num_tiles_x + tx;
                int w = h_tile_offsets[tile_id] + cursor[tile_id]++;
                h_tile_splat_indices[w] = idx;
            }
        }
    }

    // Upload
    // Ensure capacity and upload tile buffers
    if (!ensureTileBuffers(h_tile_splat_indices.size(), (size_t)(num_tiles + 1))) {
        fprintf(stderr, "Failed to ensure tile buffer capacity\n");
        return false;
    }
    // Note: d_colors_ is filled by GPU kernel, not uploaded here
    CUDA_CHECK(cudaMemcpy(d_means2D_, h_means2D.data(), num_splats * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conic3D_, h_conic3D.data(), num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacities_, h_opacities.data(), num_splats * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tile_splat_list_, h_tile_splat_indices.data(), h_tile_splat_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tile_offsets_, h_tile_offsets.data(), (num_tiles + 1) * sizeof(int), cudaMemcpyHostToDevice));

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

    // Clean up temporary render-order buffers
    cudaFree(d_pos_ws_render);
    cudaFree(d_dc_colors_render);
    cudaFree(d_sh_coeffs_render);

    return true;
}

} // namespace CudaRasterizer
