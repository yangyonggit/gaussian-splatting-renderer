// Standard Library
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// Third-party
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// Project
#include "gs/gaussian.h"
#include "gs/projection.h"
#include "gs/screen_splat.h"


const int WIDTH  = 800;
const int HEIGHT = 600;

// FloatPixel structure for alpha blending
struct FloatPixel {
    float r, g, b, a;
};

int main() {
    std::cout << "🚀 MiniGS Renderer V1-CPU Starting..." << std::endl;

    // ------------------------------------------------------------
    // 1. Camera & MVP Test
    // ------------------------------------------------------------
    glm::vec3 camPos(0.f, 0.f, 3.f);
    glm::vec3 target(0.f, 0.f, 0.f);
    glm::vec3 up(0.f, 1.f, 0.f);

    glm::mat4 view = glm::lookAt(camPos, target, up);
    glm::mat4 proj = glm::perspective(
        glm::radians(45.0f),
        float(WIDTH) / float(HEIGHT),
        0.1f, 100.0f
    );

    glm::mat4 mvp = proj * view;

    std::cout << "GLM ready. MVP[0][0] = " << mvp[0][0] << std::endl;

    // ------------------------------------------------------------
    // 2. Create Test Gaussian Splats (only positions for now)
    // ------------------------------------------------------------
    std::vector<gs::GaussianSplat> splats;
    splats.reserve(2000);

    for (int i = 0; i < 2000; i++) {
        float x = (rand() / float(RAND_MAX) - 0.5f) * 2.0f;
        float y = (rand() / float(RAND_MAX) - 0.5f) * 2.0f;
        float z = -3.0f + (rand() / float(RAND_MAX)) * 1.0f;

        splats.push_back({
            glm::vec3(x, y, z),
            glm::vec3((x+1)/2.f, (y+1)/2.f, 0.5f),
            0.05f,
            0.8f
        });
    }

    std::cout << "Created " << splats.size() << " GaussianSplats.\n";

    // ------------------------------------------------------------
    // 3. Project to ScreenSplat and collect visible splats
    // ------------------------------------------------------------
    std::vector<gs::ScreenSplat> projected;
    projected.reserve(splats.size());

    for (const auto& s : splats) {
        float sx, sy, depth;
        if (!gs::math::projectToScreen(s.position, mvp, WIDTH, HEIGHT, sx, sy, depth))
            continue;

        gs::ScreenSplat sp;
        sp.src   = &s;
        sp.sx    = sx;
        sp.sy    = sy;
        sp.depth = depth;
        projected.push_back(sp);
    }

    std::cout << "Projected " << projected.size() << " visible splats.\n";

    // ------------------------------------------------------------
    // 4. Sort by depth (Painter's Algorithm: back-to-front)
    // ------------------------------------------------------------
    std::sort(projected.begin(), projected.end(),
        [](const gs::ScreenSplat& a, const gs::ScreenSplat& b) {
            return a.depth > b.depth; // Larger depth = farther away, draw first
        });

    std::cout << "Sorted " << projected.size() << " splats by depth.\n";

    // ------------------------------------------------------------
    // 5. Initialize float framebuffer with background color
    // ------------------------------------------------------------
    std::vector<FloatPixel> framebuffer(WIDTH * HEIGHT);
    
    // Background color: dark gray (30/255 ≈ 0.1176)
    const float bgColor = 30.0f / 255.0f;
    for (auto& pixel : framebuffer) {
        pixel.r = bgColor;
        pixel.g = bgColor;
        pixel.b = bgColor;
        pixel.a = 0.0f; // Start with transparent background for blending
    }

    // ------------------------------------------------------------
    // 6. Rasterize each Gaussian splat with alpha blending
    // ------------------------------------------------------------
    const float R = 6.0f;           // Screen-space radius (pixels)
    const float sigma = R * 0.5f;
    const float inv2sigma2 = 1.0f / (2.0f * sigma * sigma);

    for (const auto& sp : projected) {
        const gs::GaussianSplat& s = *sp.src;

        // Calculate bounding box in screen space
        int x0 = std::max(0,         static_cast<int>(std::floor(sp.sx - R)));
        int x1 = std::min(WIDTH - 1, static_cast<int>(std::ceil(sp.sx + R)));
        int y0 = std::max(0,          static_cast<int>(std::floor(sp.sy - R)));
        int y1 = std::min(HEIGHT - 1, static_cast<int>(std::ceil(sp.sy + R)));

        // Rasterize pixels within the bounding box
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                // Distance from pixel center to splat center
                float dx = (x + 0.5f) - sp.sx;
                float dy = (y + 0.5f) - sp.sy;
                float dist2 = dx * dx + dy * dy;

                // Skip pixels outside the circular radius
                if (dist2 > R * R) continue;

                // Gaussian weight
                float w = std::exp(-dist2 * inv2sigma2);

                // Final alpha for this fragment
                float alpha = s.opacity * w;
                if (alpha <= 1e-4f) continue;

                // Alpha blending: src over dst
                int idx = y * WIDTH + x;
                FloatPixel& dst = framebuffer[idx];
                
                const glm::vec3& srcColor = s.color;
                float srcA = alpha;

                // Compute blended output
                float outA = srcA + dst.a * (1.0f - srcA);
                if (outA < 1e-6f) continue;

                float outR = (srcColor.r * srcA + dst.r * dst.a * (1.0f - srcA)) / outA;
                float outG = (srcColor.g * srcA + dst.g * dst.a * (1.0f - srcA)) / outA;
                float outB = (srcColor.b * srcA + dst.b * dst.a * (1.0f - srcA)) / outA;

                dst.r = outR;
                dst.g = outG;
                dst.b = outB;
                dst.a = outA;
            }
        }
    }

    std::cout << "✔ Gaussian rasterization complete.\n";

    // ------------------------------------------------------------
    // 7. Convert float framebuffer to byte array for PNG output
    // ------------------------------------------------------------
    std::vector<unsigned char> outBytes(WIDTH * HEIGHT * 3);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        const FloatPixel& p = framebuffer[i];
        int idx = i * 3;
        outBytes[idx + 0] = static_cast<unsigned char>(std::clamp(p.r, 0.0f, 1.0f) * 255.0f);
        outBytes[idx + 1] = static_cast<unsigned char>(std::clamp(p.g, 0.0f, 1.0f) * 255.0f);
        outBytes[idx + 2] = static_cast<unsigned char>(std::clamp(p.b, 0.0f, 1.0f) * 255.0f);
    }

    // ------------------------------------------------------------
    // 8. Save PNG
    // ------------------------------------------------------------
    const char* filename = "output_gaussian.png";
    int ok = stbi_write_png(filename, WIDTH, HEIGHT, 3, outBytes.data(), WIDTH * 3);

    if (ok)
        std::cout << "Saved: " << filename << std::endl;
    else
        std::cout << "Failed to save PNG!" << std::endl;

    return 0;
}