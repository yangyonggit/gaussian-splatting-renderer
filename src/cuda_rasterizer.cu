#include "cuda_rasterizer.h"
#include "gs/gaussian.h"
#include "gs/screen_splat.h"
#include "gs/sh_color.h"
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

// ============================================================
// Host Code: Memory Management and Rendering
// ============================================================

Rasterizer::Rasterizer()
    : d_means2D_(nullptr)
    , d_conic3D_(nullptr)
    , d_colors_(nullptr)
    , d_opacities_(nullptr)
    , d_output_(nullptr)
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
}

void Rasterizer::free() {
    freeBuffers();
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

    // Allocate device buffers
    if (!allocateBuffers(num_splats, width, height)) {
        fprintf(stderr, "❌ Failed to allocate CUDA buffers\n");
        return false;
    }

    // Prepare host arrays for upload (convert ScreenSplat to SoA format)
    printf("🔄 Preparing %d splats for upload (SH evaluation)...\n", num_splats);
    fflush(stdout);
    
    std::vector<float> h_means2D(num_splats * 2);
    std::vector<float> h_conic3D(num_splats * 3);
    std::vector<float> h_colors(num_splats * 3);
    std::vector<float> h_opacities(num_splats);

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
        
        const gs::GaussianSplat& src = *sp.src;
        glm::vec3 view_dir = glm::normalize(camera_pos - src.position_ws);
        glm::vec3 color = gs::evalSHColor(src, view_dir);
        color = glm::clamp(color, 0.0f, 1.0f);
        
        h_colors[i * 3 + 0] = color.r;
        h_colors[i * 3 + 1] = color.g;
        h_colors[i * 3 + 2] = color.b;
        
        h_opacities[i] = src.opacity;
    }

    printf("✅ SH evaluation complete\n");
    fflush(stdout);

    // Build tile splat lists on CPU
    printf("🔨 Building tile splat lists...\n");
    fflush(stdout);

    int num_tiles_x = (width + 15) / 16;
    int num_tiles_y = (height + 15) / 16;
    int num_tiles = num_tiles_x * num_tiles_y;

    // Temporary: tile splat lists (may be sparse)
    std::vector<std::vector<int>> tile_splat_lists(num_tiles);

    // For each sorted splat, determine which tiles it affects
    for (int idx = 0; idx < num_splats; ++idx) {
        const gs::ScreenSplat& sp = screen_splats[idx];

        // Compute bounding box in tile coordinates
        float radius = sp.radius_px;
        int tile_x_min = std::max(0, static_cast<int>((sp.sx - radius) / 16.0f));
        int tile_x_max = std::min(num_tiles_x - 1, static_cast<int>((sp.sx + radius) / 16.0f));
        int tile_y_min = std::max(0, static_cast<int>((sp.sy - radius) / 16.0f));
        int tile_y_max = std::min(num_tiles_y - 1, static_cast<int>((sp.sy + radius) / 16.0f));

        // Add this splat to all affected tiles
        for (int ty = tile_y_min; ty <= tile_y_max; ++ty) {
            for (int tx = tile_x_min; tx <= tile_x_max; ++tx) {
                int tile_id = ty * num_tiles_x + tx;
                tile_splat_lists[tile_id].push_back(idx);
            }
        }
    }

    // Flatten tile splat lists and create offsets
    std::vector<int> h_tile_splat_list;
    std::vector<int> h_tile_offsets(num_tiles + 1, 0);

    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        h_tile_offsets[tile_id] = h_tile_splat_list.size();
        for (int idx : tile_splat_lists[tile_id]) {
            h_tile_splat_list.push_back(idx);
        }
    }
    h_tile_offsets[num_tiles] = h_tile_splat_list.size();

    printf("✅ Built tile lists: %zu total splat references across %d tiles\n",
           h_tile_splat_list.size(), num_tiles);
    fflush(stdout);

    // Allocate device memory for tile lists
    int* d_tile_splat_list = nullptr;
    int* d_tile_offsets = nullptr;

    CUDA_CHECK(cudaMalloc(&d_tile_splat_list, h_tile_splat_list.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tile_offsets, (num_tiles + 1) * sizeof(int)));

    // Upload data to GPU
    printf("📤 Uploading data to GPU...\n");
    fflush(stdout);

    CUDA_CHECK(cudaMemcpy(d_means2D_, h_means2D.data(), 
                         num_splats * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conic3D_, h_conic3D.data(), 
                         num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colors_, h_colors.data(), 
                         num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacities_, h_opacities.data(), 
                         num_splats * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tile_splat_list, h_tile_splat_list.data(),
                         h_tile_splat_list.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tile_offsets, h_tile_offsets.data(),
                         (num_tiles + 1) * sizeof(int), cudaMemcpyHostToDevice));

    printf("✅ Uploaded data to GPU\n");
    fflush(stdout);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid(num_tiles_x, num_tiles_y);

    printf("🔧 Launching tile-based kernel: grid(%d, %d), block(%d, %d)\n",
           grid.x, grid.y, block.x, block.y);
    fflush(stdout);

    splatKernel<<<grid, block>>>(
        d_means2D_,
        d_conic3D_,
        d_colors_,
        d_opacities_,
        d_tile_splat_list,
        d_tile_offsets,
        num_tiles_x,
        width,
        height,
        d_output_
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_tile_splat_list);
        cudaFree(d_tile_offsets);
        return false;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("✅ Kernel execution completed\n");

    CUDA_CHECK(cudaMemcpy(output_image, d_output_, 
                         width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("✅ Downloaded output image from GPU\n");

    // Cleanup
    cudaFree(d_tile_splat_list);
    cudaFree(d_tile_offsets);

    return true;
}

} // namespace CudaRasterizer
