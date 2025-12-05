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
#include "gs/ply_loader.h"


const int WIDTH  = 800;
const int HEIGHT = 600;

// FloatPixel structure for alpha blending
struct FloatPixel {
    float r, g, b, a;
};

// ------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------

/**
 * Parse command line arguments
 */
struct CommandLineArgs {
    std::string inputFile;
    std::string outputFile;
    bool showHelp;
};

CommandLineArgs parseCommandLine(int argc, char* argv[]) {
    CommandLineArgs args;
    args.inputFile = "scene.ply";
    args.outputFile = "output_real_scene.png";
    args.showHelp = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--scene" && i + 1 < argc) {
            args.inputFile = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            args.outputFile = argv[++i];
        }
        else if (arg == "--help" || arg == "-h") {
            args.showHelp = true;
        }
    }

    return args;
}

/**
 * Print usage information
 */
void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --scene <file>   Input PLY file (default: scene.ply)\n";
    std::cout << "  --output <file>  Output PNG file (default: output_real_scene.png)\n";
    std::cout << "  --help, -h       Show this help message\n";
}

/**
 * Build Model-View-Projection matrix from camera parameters
 */
glm::mat4 buildMVP(const glm::vec3& camPos, const glm::vec3& target, const glm::vec3& up) {
    glm::mat4 view = glm::lookAt(camPos, target, up);
    glm::mat4 proj = glm::perspective(
        glm::radians(45.0f),
        float(WIDTH) / float(HEIGHT),
        0.1f, 100.0f
    );
    return proj * view;
}

/**
 * Project 3D Gaussian splats to screen space
 */
std::vector<gs::ScreenSplat> projectSplats(
    const std::vector<gs::GaussianSplat>& splats,
    const glm::mat4& mvp,
    int width,
    int height
) {
    std::vector<gs::ScreenSplat> projected;
    projected.reserve(splats.size());

    for (const auto& s : splats) {
        float sx, sy, depth;
        if (!gs::math::projectToScreen(s.position, mvp, width, height, sx, sy, depth))
            continue;

        gs::ScreenSplat sp;
        sp.src   = &s;
        sp.sx    = sx;
        sp.sy    = sy;
        sp.depth = depth;
        projected.push_back(sp);
    }

    return projected;
}

/**
 * Sort splats by depth (Painter's Algorithm: back-to-front)
 */
void sortSplatsByDepth(std::vector<gs::ScreenSplat>& projected) {
    std::sort(projected.begin(), projected.end(),
        [](const gs::ScreenSplat& a, const gs::ScreenSplat& b) {
            return a.depth > b.depth;
        });
}

/**
 * Initialize framebuffer with background color
 */
std::vector<FloatPixel> initializeFramebuffer(int width, int height, float bgColor) {
    std::vector<FloatPixel> framebuffer(width * height);
    for (auto& pixel : framebuffer) {
        pixel.r = bgColor;
        pixel.g = bgColor;
        pixel.b = bgColor;
        pixel.a = 0.0f;
    }
    return framebuffer;
}

/**
 * Rasterize Gaussian splats with alpha blending
 */
void rasterizeGaussians(
    std::vector<FloatPixel>& framebuffer,
    const std::vector<gs::ScreenSplat>& projected,
    int width,
    int height,
    float radius
) {
    const float sigma = radius * 0.5f;
    const float inv2sigma2 = 1.0f / (2.0f * sigma * sigma);

    for (const auto& sp : projected) {
        const gs::GaussianSplat& s = *sp.src;

        // Calculate bounding box in screen space
        int x0 = std::max(0,           static_cast<int>(std::floor(sp.sx - radius)));
        int x1 = std::min(width - 1,   static_cast<int>(std::ceil(sp.sx + radius)));
        int y0 = std::max(0,           static_cast<int>(std::floor(sp.sy - radius)));
        int y1 = std::min(height - 1,  static_cast<int>(std::ceil(sp.sy + radius)));

        // Rasterize pixels within the bounding box
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                // Distance from pixel center to splat center
                float dx = (x + 0.5f) - sp.sx;
                float dy = (y + 0.5f) - sp.sy;
                float dist2 = dx * dx + dy * dy;

                // Skip pixels outside the circular radius
                if (dist2 > radius * radius) continue;

                // Gaussian weight
                float w = std::exp(-dist2 * inv2sigma2);

                // Final alpha for this fragment
                float alpha = s.opacity * w;
                if (alpha <= 1e-4f) continue;

                // Alpha blending: src over dst
                int idx = y * width + x;
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
}

/**
 * Convert float framebuffer to byte array for PNG output
 */
std::vector<unsigned char> convertToBytes(const std::vector<FloatPixel>& framebuffer) {
    std::vector<unsigned char> outBytes(framebuffer.size() * 3);
    for (size_t i = 0; i < framebuffer.size(); ++i) {
        const FloatPixel& p = framebuffer[i];
        int idx = i * 3;
        outBytes[idx + 0] = static_cast<unsigned char>(std::clamp(p.r, 0.0f, 1.0f) * 255.0f);
        outBytes[idx + 1] = static_cast<unsigned char>(std::clamp(p.g, 0.0f, 1.0f) * 255.0f);
        outBytes[idx + 2] = static_cast<unsigned char>(std::clamp(p.b, 0.0f, 1.0f) * 255.0f);
    }
    return outBytes;
}

// ------------------------------------------------------------
// Main Function
// ------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::cout << "🚀 MiniGS Renderer V1-CPU Starting..." << std::endl;

    // ------------------------------------------------------------
    // 1. Parse command line arguments
    // ------------------------------------------------------------
    CommandLineArgs args = parseCommandLine(argc, argv);

    if (args.showHelp) {
        printUsage(argv[0]);
        return 0;
    }

    std::cout << "Input:  " << args.inputFile << std::endl;
    std::cout << "Output: " << args.outputFile << std::endl;

    // ------------------------------------------------------------
    // 2. Setup Camera & MVP
    // ------------------------------------------------------------
    glm::vec3 camPos(0.f, 0.f, 3.f);
    glm::vec3 target(0.f, 0.f, 0.f);
    glm::vec3 up(0.f, 1.f, 0.f);

    glm::mat4 mvp = buildMVP(camPos, target, up);
    std::cout << "GLM ready. MVP[0][0] = " << mvp[0][0] << std::endl;

    // ------------------------------------------------------------
    // 3. Load Gaussian Splats from PLY file
    // ------------------------------------------------------------
    std::vector<gs::GaussianSplat> splats = gs::loadGaussianPly(args.inputFile);
    
    if (splats.empty()) {
        std::cerr << "Failed to load PLY file or file is empty!" << std::endl;
        return 1;
    }

    // ------------------------------------------------------------
    // 3. Project to screen space & Sort by depth
    // ------------------------------------------------------------
    std::vector<gs::ScreenSplat> projected = projectSplats(splats, mvp, WIDTH, HEIGHT);
    std::cout << "Projected " << projected.size() << " visible splats.\n";

    sortSplatsByDepth(projected);
    std::cout << "Sorted " << projected.size() << " splats by depth.\n";

    // ------------------------------------------------------------
    // 4. Rasterize Gaussians
    // ------------------------------------------------------------
    const float bgColor = 30.0f / 255.0f;
    std::vector<FloatPixel> framebuffer = initializeFramebuffer(WIDTH, HEIGHT, bgColor);

    const float radius = 6.0f;
    rasterizeGaussians(framebuffer, projected, WIDTH, HEIGHT, radius);
    std::cout << "✔ Gaussian rasterization complete.\n";

    // ------------------------------------------------------------
    // 5. Save output PNG
    // ------------------------------------------------------------
    std::vector<unsigned char> outBytes = convertToBytes(framebuffer);
    
    int ok = stbi_write_png(args.outputFile.c_str(), WIDTH, HEIGHT, 3, outBytes.data(), WIDTH * 3);

    if (ok)
        std::cout << "Saved: " << args.outputFile << std::endl;
    else
        std::cout << "Failed to save PNG!" << std::endl;

    return 0;
}