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
#include "gs/camera.h"
#include "gs/profiler.h"
#include "cpu_rasterizer.h"
#include "cuda_rasterizer.h"


const int DEFAULT_WIDTH  = 1959;
const int DEFAULT_HEIGHT = 1090;

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
    bool useCuda;
};

CommandLineArgs parseCommandLine(int argc, char* argv[]) {
    CommandLineArgs args;
    args.inputFile = "scene.ply";
    args.outputFile = "output_real_scene.png";
    args.showHelp = false;
    args.camId = -1;
    args.useCuda = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--scene" && i + 1 < argc) {
            args.inputFile = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            args.outputFile = argv[++i];
        }
        else if (arg == "--cuda") {
            args.useCuda = true;
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
    std::cout << "  --cuda           Use CUDA rasterizer (default: CPU)\n";
    std::cout << "  --help, -h       Show this help message\n";
}

// ============================================================
// Helper Functions for Rendering
// ============================================================

// ------------------------------------------------------------
// Main Function
// ------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::cout << "🚀 MiniGS Renderer V3 (CPU + CUDA) Starting..." << std::endl;

    // Parse command line arguments
    CommandLineArgs args = parseCommandLine(argc, argv);

    if (args.showHelp) {
        printUsage(argv[0]);
        return 0;
    }

    std::cout << "Input:  " << args.inputFile << std::endl;
    std::cout << "Output: " << args.outputFile << std::endl;
    std::cout << "Mode:   " << (args.useCuda ? "CUDA" : "CPU") << std::endl;
    if (!args.cameraConfigFile.empty()) {
        std::cout << "Camera config: " << args.cameraConfigFile << " (id=" << args.camId << ")" << std::endl;
    }

    // Setup Camera & View/Projection Matrices
    gs::RenderCamera cam;
    if (!gs::Camera::setupRenderCamera(args.cameraConfigFile, args.camId, cam)) {
        return 1;
    }

    // Choose rendering path
    if (args.useCuda) {
        // CUDA V1 Rendering Path
        std::cout << "\n🎮 CUDA V1 Rendering Path" << std::endl;
        std::cout << "========================================" << std::endl;

        // CPU Preprocessing: project and sort
        CpuRasterizer::Rasterizer cpu_prep;
        std::vector<gs::GaussianSplat> splats;  // Keep splats alive!
        std::vector<gs::ScreenSplat> screen_splats;

        if (!cpu_prep.prepareForCuda(
            args.inputFile,
            cam.view,
            cam.proj,
            cam.width,
            cam.height,
            splats,
            screen_splats
        )) {
            std::cerr << "❌ CPU preprocessing failed" << std::endl;
            return 1;
        }

        // CUDA Rendering
        CudaRasterizer::Rasterizer cuda_rasterizer;
        std::vector<float> output_image(cam.width * cam.height * 3);

        // Upload scene-static data (SH coefficients, positions) to GPU
        std::cout << "📤 Uploading scene data to GPU..." << std::endl;
        if (!cuda_rasterizer.uploadSceneData(splats)) {
            std::cerr << "❌ Failed to upload scene data to GPU" << std::endl;
            return 1;
        }

        if (!cuda_rasterizer.render_cuda(
            screen_splats,
            cam.position,
            cam.width,
            cam.height,
            output_image.data()
        )) {
            std::cerr << "❌ CUDA rendering failed" << std::endl;
            return 1;
        }

        // Save output
        std::vector<unsigned char> output_bytes(cam.width * cam.height * 3);
        for (size_t i = 0; i < output_image.size(); ++i) {
            output_bytes[i] = static_cast<unsigned char>(
                std::clamp(output_image[i], 0.0f, 1.0f) * 255.0f
            );
        }

        double png_ms = 0.0;
        int ok = 0;
        {
            ScopedTimer timer("write_png", &png_ms);
            ok = stbi_write_png(args.outputFile.c_str(), cam.width, cam.height, 3,
                               output_bytes.data(), cam.width * 3);
        }

        if (!ok) {
            std::cerr << "❌ Failed to save PNG" << std::endl;
            return 1;
        }

        std::cout << "✅ Saved: " << args.outputFile << std::endl;

    } else {
        // CPU Rendering Path
        std::cout << "\n💻 CPU Rendering Path" << std::endl;
        std::cout << "========================================" << std::endl;

        CpuRasterizer::Rasterizer cpu_rasterizer;
        if (!cpu_rasterizer.render(
            args.inputFile,
            args.outputFile,
            cam.view,
            cam.proj,
            cam.width,
            cam.height,
            cam.position
        )) {
            std::cerr << "❌ CPU rendering failed" << std::endl;
            return 1;
        }
    }

    std::cout << "\n🎉 Rendering complete!" << std::endl;

    return 0;
}