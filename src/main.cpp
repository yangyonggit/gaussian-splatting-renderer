// Standard Library
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>

#include <nlohmann/json.hpp>

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
#include "gs/ellipse.h"
#include "gs/sh_color.h"


const int DEFAULT_WIDTH  = 1959;
const int DEFAULT_HEIGHT = 1090;

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
    std::string cameraConfigFile;
    int camId;
};

CommandLineArgs parseCommandLine(int argc, char* argv[]) {
    CommandLineArgs args;
    args.inputFile = "scene.ply";
    args.outputFile = "output_real_scene.png";
    args.showHelp = false;
    args.camId = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--scene" && i + 1 < argc) {
            args.inputFile = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            args.outputFile = argv[++i];
        }
        else if (arg == "--camera_config" && i + 1 < argc) {
            args.cameraConfigFile = argv[++i];
        }
        else if (arg == "--cam_id" && i + 1 < argc) {
            args.camId = std::stoi(argv[++i]);
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
    std::cout << "  --camera_config <file> JSON camera list (array of cameras)\n";
    std::cout << "  --cam_id <id>    Camera id to pick from the JSON\n";
    std::cout << "  --help, -h       Show this help message\n";
}

struct CameraConfig {
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    float fx = 1159.5880733038064f;
    float fy = 1164.6601287484507f;
    glm::vec3 position{ 2.2517861865205795f, 0.3663530997455256f, 3.8251153433115346f };
    glm::mat3 rotation_c2w{ 1.0f }; // camera-to-world rotation (columns: right, up, forward)
};

struct RenderCamera {
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    glm::vec3 position{0.0f};
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
};

bool loadCameraConfig(const std::string& path, int camId, CameraConfig& out)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Failed to open camera config: " << path << "\n";
        return false;
    }

    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse camera config JSON: " << e.what() << "\n";
        return false;
    }

    if (!j.is_array()) {
        std::cerr << "Camera config must be a JSON array of camera objects.\n";
        return false;
    }

    bool found = false;
    for (const auto& cam : j) {
        if (!cam.contains("id")) continue;
        int id = cam["id"].get<int>();
        if (id != camId) continue;

        if (!cam.contains("position") || !cam.contains("rotation") ||
            !cam.contains("fx") || !cam.contains("fy") ||
            !cam.contains("width") || !cam.contains("height")) {
            std::cerr << "Camera entry missing required fields (id=" << camId << ").\n";
            return false;
        }

        auto pos = cam["position"];
        auto rot = cam["rotation"];

        if (!pos.is_array() || pos.size() != 3 || !rot.is_array() || rot.size() != 3 ||
            !rot[0].is_array() || !rot[1].is_array() || !rot[2].is_array()) {
            std::cerr << "Camera entry has invalid position or rotation format.\n";
            return false;
        }

        out.position = glm::vec3(
            static_cast<float>(pos[0].get<double>()),
            static_cast<float>(pos[1].get<double>()),
            static_cast<float>(pos[2].get<double>())
        );

        // rotation provided as 3x3 row-major matrix (camera-to-world).
        // Columns: right, up, forward.
        out.rotation_c2w = glm::mat3(1.0f);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                out.rotation_c2w[c][r] = static_cast<float>(rot[r][c].get<double>());
            }
        }

        out.fx = static_cast<float>(cam["fx"].get<double>());
        out.fy = static_cast<float>(cam["fy"].get<double>());
        out.width = cam["width"].get<int>();
        out.height = cam["height"].get<int>();

        found = true;
        break;
    }

    if (!found) {
        std::cerr << "Camera id " << camId << " not found in config." << std::endl;
        return false;
    }

    return true;
}

glm::mat4 buildViewFromCamera(const CameraConfig& cam)
{
    glm::vec3 right   = glm::vec3(cam.rotation_c2w[0][0], cam.rotation_c2w[0][1], cam.rotation_c2w[0][2]);
    glm::vec3 up      = glm::vec3(cam.rotation_c2w[1][0], cam.rotation_c2w[1][1], cam.rotation_c2w[1][2]);
    glm::vec3 forward = glm::vec3(cam.rotation_c2w[2][0], cam.rotation_c2w[2][1], cam.rotation_c2w[2][2]);

    glm::vec3 cam_target = cam.position + forward;
    return glm::lookAt(cam.position, cam_target, up);
}

glm::mat4 buildProjFromCamera(const CameraConfig& cam)
{
    float aspect = static_cast<float>(cam.width) / static_cast<float>(cam.height);
    float fov_y = 2.0f * std::atan(static_cast<float>(cam.height) / (2.0f * cam.fy));
    return glm::perspective(fov_y, aspect, 0.01f, 100.0f);
}

bool setupRenderCamera(const CommandLineArgs& args, RenderCamera& outCam)
{
    if (!args.cameraConfigFile.empty()) {
        if (args.camId < 0) {
            std::cerr << "--cam_id must be provided when using --camera_config" << std::endl;
            return false;
        }

        CameraConfig cam;
        if (!loadCameraConfig(args.cameraConfigFile, args.camId, cam)) {
            return false;
        }

        outCam.width = cam.width;
        outCam.height = cam.height;
        outCam.position = cam.position;
        outCam.view = buildViewFromCamera(cam);
        outCam.proj = buildProjFromCamera(cam);

        std::cout << "Loaded camera id=" << args.camId << " (" << cam.width << "x" << cam.height
                  << ", fx=" << cam.fx << ", fy=" << cam.fy << ")" << std::endl;
        std::cout << "Camera position: (" << outCam.position.x << ", " << outCam.position.y << ", " << outCam.position.z << ")" << std::endl;
        return true;
    }

    // Default camera fallback
    outCam.width = DEFAULT_WIDTH;
    outCam.height = DEFAULT_HEIGHT;
    outCam.position = glm::vec3(
        2.2517861865205795f,
        0.3663530997455256f,
        3.8251153433115346f
    );

    glm::vec3 forward(
        -0.32975178298961f,
        0.04662275181511159f,
        -0.942915733577693f
    );

    glm::vec3 up(
        0.08904122545822853f,
        0.9958634384862263f,
        0.01810171415328269f
    );

    glm::vec3 cam_target = outCam.position + forward;

    outCam.view = glm::lookAt(outCam.position, cam_target, up);

    float fov_y_deg = 50.3f;

    outCam.proj = glm::perspective(
        glm::radians(fov_y_deg),
        static_cast<float>(outCam.width) / static_cast<float>(outCam.height),
        0.01f,
        100.0f
    );

    std::cout << "Using default camera. Position: (" << outCam.position.x << ", " << outCam.position.y << ", " << outCam.position.z << ")" << std::endl;
    return true;
}

/**
 * Build Model-View-Projection matrix from camera parameters
 */
glm::mat4 buildMVP(const glm::vec3& camPos, const glm::vec3& target, const glm::vec3& up) {
    glm::mat4 view = glm::lookAt(camPos, target, up);
    glm::mat4 proj = glm::perspective(
        glm::radians(50.3f),
        float(DEFAULT_WIDTH) / float(DEFAULT_HEIGHT),
        0.01f, 100.0f
    );
    return proj * view;
}

/**
 * Project 3D Gaussian splats to screen space with ellipse support
 */
std::vector<gs::ScreenSplat> projectSplats(
    const std::vector<gs::GaussianSplat>& splats,
    const glm::mat4& view,
    const glm::mat4& proj,
    int width,
    int height
) {
    std::vector<gs::ScreenSplat> projected;
    projected.reserve(splats.size());

    for (const auto& s : splats) {
        gs::ScreenSplat sp;
        
        // Use ellipse projection instead of simple point projection
        if (!gs::projectToScreenEllipse(s, view, proj, width, height, sp))
            continue;

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
 * Rasterize Gaussian splats with ellipse-based (Mahalanobis distance) blending
 */
void rasterizeGaussians(
    std::vector<FloatPixel>& framebuffer,
    const std::vector<gs::ScreenSplat>& projected,
    int width,
    int height,
    const glm::vec3& camera_pos
) {
    for (const auto& sp : projected) {
        const gs::GaussianSplat& s = *sp.src;

        // Calculate bounding box in screen space
        int x0 = std::max(0,           static_cast<int>(std::floor(sp.sx - sp.radius_px)));
        int x1 = std::min(width - 1,   static_cast<int>(std::ceil(sp.sx + sp.radius_px)));
        int y0 = std::max(0,           static_cast<int>(std::floor(sp.sy - sp.radius_px)));
        int y1 = std::min(height - 1,  static_cast<int>(std::ceil(sp.sy + sp.radius_px)));

        // Evaluate view-dependent color using SH
        glm::vec3 view_dir = glm::normalize(camera_pos - s.position_ws);
        glm::vec3 color = gs::evalSHColor(s, view_dir);

        // Rasterize pixels within the bounding box
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                // Compute offset from splat center
                glm::vec2 d(
                    (x + 0.5f) - sp.sx,
                    (y + 0.5f) - sp.sy
                );

                // Mahalanobis distance squared: d^T Σ^{-1} d
                glm::vec2 tmp = sp.cov_inv * d;
                float r2 = glm::dot(d, tmp);

                // Elliptical Gaussian weight
                float w = std::exp(-0.5f * r2);

                // Skip very small weights
                if (w < 1e-4f) continue;

                // Final alpha combining opacity and Gaussian weight
                float alpha = s.opacity * w;
                if (alpha <= 1e-6f) continue;

                // Alpha blending: src over dst
                int idx = y * width + x;
                FloatPixel& dst = framebuffer[idx];
                
                float srcA = alpha;

                // Compute blended output
                float outA = srcA + dst.a * (1.0f - srcA);
                if (outA < 1e-6f) continue;

                float outR = (color.r * srcA + dst.r * dst.a * (1.0f - srcA)) / outA;
                float outG = (color.g * srcA + dst.g * dst.a * (1.0f - srcA)) / outA;
                float outB = (color.b * srcA + dst.b * dst.a * (1.0f - srcA)) / outA;

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
    std::cout << "🚀 MiniGS Renderer V3-CPU (Ellipse + SH) Starting..." << std::endl;

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
    if (!args.cameraConfigFile.empty()) {
        std::cout << "Camera config: " << args.cameraConfigFile << " (id=" << args.camId << ")" << std::endl;
    }

    // Enable SH color evaluation
    gs::setSHEvaluationEnabled(true);
    std::cout << "SH color evaluation: ENABLED (temporarily using DC fallback)" << std::endl;

    // ------------------------------------------------------------
    // 2. Setup Camera & View/Projection Matrices
    // ------------------------------------------------------------
    RenderCamera cam;
    if (!setupRenderCamera(args, cam)) {
        return 1;
    }

    // ------------------------------------------------------------
    // 3. Load Gaussian Splats from PLY file
    // ------------------------------------------------------------
    std::vector<gs::GaussianSplat> splats = gs::loadGaussianPly(args.inputFile);
    
    if (splats.empty()) {
        std::cerr << "Failed to load PLY file or file is empty!" << std::endl;
        return 1;
    }

    // ------------------------------------------------------------
    // 4. Project to screen space (with ellipse parameters) & Sort by depth
    // ------------------------------------------------------------
    std::vector<gs::ScreenSplat> projected = projectSplats(splats, cam.view, cam.proj, cam.width, cam.height);
    std::cout << "Projected " << projected.size() << " visible splats.\n";

    sortSplatsByDepth(projected);
    std::cout << "Sorted " << projected.size() << " splats by depth.\n";
    
    if (!projected.empty()) {
        std::cout << "Depth range: [" << projected.back().depth << ", " << projected.front().depth << "]\n";
    }

    // Debug statistics
    float avg_radius = 0.0f;
    int degenerate_count = 0;
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
    }

    // ------------------------------------------------------------
    // 5. Rasterize Gaussians (ellipse + SH color)
    // ------------------------------------------------------------
    const float bgColor = 30.0f / 255.0f;
    std::vector<FloatPixel> framebuffer = initializeFramebuffer(cam.width, cam.height, bgColor);

    rasterizeGaussians(framebuffer, projected, cam.width, cam.height, cam.position);
    std::cout << "✔ Ellipse rasterization with SH colors complete.\n";

    // ------------------------------------------------------------
    // 6. Save output PNG
    // ------------------------------------------------------------
    std::vector<unsigned char> outBytes = convertToBytes(framebuffer);
    
    int ok = stbi_write_png(args.outputFile.c_str(), cam.width, cam.height, 3, outBytes.data(), cam.width * 3);

    if (ok)
        std::cout << "Saved: " << args.outputFile << std::endl;
    else
        std::cout << "Failed to save PNG!" << std::endl;

    return 0;
}