#include "cpu_rasterizer.h"
#include "gs/projection.h"
#include "gs/screen_splat.h"
#include "gs/ellipse.h"
#include "gs/sh_color.h"
#include "gs/ply_loader.h"
#include "gs/profiler.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#if defined(GS_ENABLE_IMAGE_WRITE)
#include <stb_image_write.h>
#endif

namespace CpuRasterizer {

Rasterizer::Rasterizer() {}

Rasterizer::~Rasterizer() {}

bool Rasterizer::render(
    const std::string& ply_path,
    const std::string& output_path,
    const glm::mat4& view,
    const glm::mat4& proj,
    int width,
    int height,
    const glm::vec3& camera_pos
) {
    // Enable SH color evaluation
    gs::setSHEvaluationEnabled(true);
    std::cout << "SH color evaluation: ENABLED (temporarily using DC fallback)" << std::endl;

    // Load Gaussian Splats from PLY file
    std::vector<gs::GaussianSplat> splats = gs::loadGaussianPly(ply_path);
    
    if (splats.empty()) {
        std::cerr << "Failed to load PLY file or file is empty!" << std::endl;
        return false;
    }

    std::cout << "Loaded " << splats.size() << " Gaussian splats." << std::endl;

    // Project to screen space (with ellipse parameters)
    std::vector<gs::ScreenSplat> projected;
    if (!projectSplats(splats, view, proj, width, height, projected)) {
        return false;
    }

    // Sort by depth (back-to-front)
    sortByDepth(projected);

    // Debug statistics
    float avg_radius = 0.0f;
    float max_sx = -999, min_sx = 999, max_sy = -999, min_sy = 999;
    
    for (const auto& sp : projected) {
        avg_radius += sp.radius_px;
        max_sx = std::max(max_sx, sp.sx);
        min_sx = std::min(min_sx, sp.sx);
        max_sy = std::max(max_sy, sp.sy);
        min_sy = std::min(min_sy, sp.sy);
    }
    
    if (!projected.empty()) {
        avg_radius /= projected.size();
        std::cout << "Average splat radius: " << avg_radius << " pixels\n";
        std::cout << "Screen X range: [" << min_sx << ", " << max_sx << "]\n";
        std::cout << "Screen Y range: [" << min_sy << ", " << max_sy << "]\n";
        std::cout << "Depth range: [" << projected.back().depth << ", " << projected.front().depth << "]\n";
    }

    // Rasterize
    std::vector<FloatPixel> framebuffer;
    if (!rasterize(projected, width, height, camera_pos, framebuffer)) {
        return false;
    }

    // Save output
    if (!saveFramebuffer(framebuffer, width, height, output_path)) {
        return false;
    }

    return true;
}

bool Rasterizer::projectSplats(
    const std::vector<gs::GaussianSplat>& splats,
    const glm::mat4& view,
    const glm::mat4& proj,
    int width,
    int height,
    std::vector<gs::ScreenSplat>& out_projected
) {
    out_projected.reserve(splats.size());

    for (size_t i = 0; i < splats.size(); ++i) {
        const auto& s = splats[i];
        gs::ScreenSplat sp;
        
        // Use ellipse projection instead of simple point projection
        if (!gs::projectToScreenEllipse(s, view, proj, width, height, sp))
            continue;

        sp.gaussian_id = static_cast<int>(i);

        out_projected.push_back(sp);
    }

    std::cout << "Projected " << out_projected.size() << " visible splats.\n";
    return true;
}

void Rasterizer::sortByDepth(std::vector<gs::ScreenSplat>& splats) {
    std::sort(splats.begin(), splats.end(),
        [](const gs::ScreenSplat& a, const gs::ScreenSplat& b) {
            return a.depth > b.depth;
        });
    std::cout << "Sorted " << splats.size() << " splats by depth (back-to-front).\n";
}

bool Rasterizer::rasterize(
    const std::vector<gs::ScreenSplat>& projected,
    int width,
    int height,
    const glm::vec3& camera_pos,
    std::vector<FloatPixel>& out_framebuffer
) {
    // Initialize framebuffer
    const float bgColor = 30.0f / 255.0f;
    out_framebuffer.assign(width * height, {bgColor, bgColor, bgColor, 0.0f});

    // Rasterize each splat
    for (const auto& sp : projected) {
        const gs::GaussianSplat& s = *sp.src;

        // Calculate bounding box in screen space
        int x0 = std::max(0,          static_cast<int>(std::floor(sp.sx - sp.radius_px)));
        int x1 = std::min(width - 1,  static_cast<int>(std::ceil(sp.sx + sp.radius_px)));
        int y0 = std::max(0,          static_cast<int>(std::floor(sp.sy - sp.radius_px)));
        int y1 = std::min(height - 1, static_cast<int>(std::ceil(sp.sy + sp.radius_px)));

        // Evaluate view-dependent color using SH
        glm::vec3 view_dir = glm::normalize(camera_pos - s.position_ws);
        glm::vec3 color = gs::evalSHColor(s, view_dir);

        // Clamp color to ensure valid range before premultiplication
        color = glm::clamp(color, 0.0f, 1.0f);

        // Rasterize pixels within the bounding box
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                // Compute offset from splat center
                glm::vec2 d(
                    (x + 0.5f) - sp.sx,
                    (y + 0.5f) - sp.sy
                );

                // Mahalanobis distance squared: d^T * Σ^{-1} * d
                glm::vec2 tmp = sp.cov_inv * d;
                float r2 = glm::dot(d, tmp);

                // Elliptical Gaussian weight
                float w = std::exp(-0.5f * r2);

                // Skip very small weights for performance
                if (w < 1e-4f) continue;

                // Calculate final alpha
                float alpha = s.opacity * w;
                
                // Skip almost transparent pixels
                if (alpha <= 1.0f / 255.0f) continue; 

                // Premultiplied Alpha Blending
                int idx = y * width + x;
                FloatPixel& dst = out_framebuffer[idx];

                // Pre-multiply source color
                float src_a = alpha;
                float src_r = color.r * src_a;
                float src_g = color.g * src_a;
                float src_b = color.b * src_a;

                // Calculate transmission factor
                float inv_src_a = 1.0f - src_a;

                // Accumulate
                dst.r = src_r + dst.r * inv_src_a;
                dst.g = src_g + dst.g * inv_src_a;
                dst.b = src_b + dst.b * inv_src_a;
                dst.a = src_a + dst.a * inv_src_a; 
            }
        }
    }

    std::cout << "✔ Ellipse rasterization with SH colors complete.\n";
    return true;
}

#if defined(GS_ENABLE_IMAGE_WRITE)
bool Rasterizer::saveFramebuffer(
    const std::vector<FloatPixel>& framebuffer,
    int width,
    int height,
    const std::string& output_path
) {
    std::vector<unsigned char> outBytes(framebuffer.size() * 3);
    for (size_t i = 0; i < framebuffer.size(); ++i) {
        const FloatPixel& p = framebuffer[i];
        int idx = static_cast<int>(i * 3);
        outBytes[idx + 0] = static_cast<unsigned char>(std::clamp(p.r, 0.0f, 1.0f) * 255.0f);
        outBytes[idx + 1] = static_cast<unsigned char>(std::clamp(p.g, 0.0f, 1.0f) * 255.0f);
        outBytes[idx + 2] = static_cast<unsigned char>(std::clamp(p.b, 0.0f, 1.0f) * 255.0f);
    }

    double save_png_ms = 0.0;
    int ok = 0;
    {
        ScopedTimer timer("save_png", &save_png_ms);
        ok = stbi_write_png(output_path.c_str(), width, height, 3, outBytes.data(), width * 3);
    }

    if (ok) {
        std::cout << "Saved: " << output_path << std::endl;
        return true;
    } else {
        std::cout << "Failed to save PNG!" << std::endl;
        return false;
    }
}
#else
bool Rasterizer::saveFramebuffer(
    const std::vector<FloatPixel>& framebuffer,
    int width,
    int height,
    const std::string& output_path
) {
    (void)framebuffer;
    (void)width;
    (void)height;
    (void)output_path;
    std::cout << "[Info] Image saving disabled for this build (no STB)." << std::endl;
    return true;
}
#endif

bool Rasterizer::prepareForCuda(
    const std::string& ply_path,
    const glm::mat4& view,
    const glm::mat4& proj,
    int width,
    int height,
    std::vector<gs::GaussianSplat>& out_splats,
    std::vector<gs::ScreenSplat>& out_screen_splats
) {
    std::cout << "🔄 CPU Preprocessing for CUDA V1..." << std::endl;

    double t_load_ms = 0.0;
    double t_project_ms = 0.0;
    double t_sort_ms = 0.0;

    // Load Gaussian Splats from PLY
    {
        ScopedTimer timer("load_ply", &t_load_ms);
        out_splats = gs::loadGaussianPly(ply_path);
    }
    std::vector<gs::GaussianSplat>& splats = out_splats;
    
    if (splats.empty()) {
        std::cerr << "Failed to load PLY file or file is empty!" << std::endl;
        return false;
    }

    std::cout << "Loaded " << splats.size() << " Gaussian splats." << std::endl;

    // Project to screen space
    {
        ScopedTimer timer("project_splats", &t_project_ms);
        if (!projectSplats(splats, view, proj, width, height, out_screen_splats)) {
            return false;
        }
    }

    // Sort by depth (back-to-front)
    {
        ScopedTimer timer("sort_by_depth", &t_sort_ms);
        sortByDepth(out_screen_splats);
    }

#if ENABLE_PROFILING
    double cpu_total_ms = t_load_ms + t_project_ms + t_sort_ms;
    std::printf("[Profile] total_cpu_preprocess: %.3f ms | splats: %zu -> %zu | res: %dx%d\n",
                cpu_total_ms,
                splats.size(),
                out_screen_splats.size(),
                width,
                height);
#endif

    std::cout << "✅ CPU preprocessing complete: " 
              << out_screen_splats.size() << " splats ready for CUDA" << std::endl;

    return true;
}

bool Rasterizer::reprojectSplats(
    const std::vector<gs::GaussianSplat>& splats,
    const glm::mat4& view,
    const glm::mat4& proj,
    int width,
    int height,
    std::vector<gs::ScreenSplat>& out_screen_splats
) {
    // Clear previous frame's splats before reprojecting
    out_screen_splats.clear();
    
    // Reproject with new view/proj matrices
    if (!projectSplats(splats, view, proj, width, height, out_screen_splats)) {
        return false;
    }

    // Re-sort by depth
    sortByDepth(out_screen_splats);

    return true;
}

} // namespace CpuRasterizer
