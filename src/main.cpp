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
    if (!args.cameraConfigFile.empty()) {
        std::cout << "Camera config: " << args.cameraConfigFile << " (id=" << args.camId << ")" << std::endl;
    }

    // Setup Camera & View/Projection Matrices
    gs::RenderCamera cam;
    if (!gs::Camera::setupRenderCamera(args.cameraConfigFile, args.camId, cam)) {
        return 1;
    }

    // Load Gaussian splats from PLY file
    std::cout << "\n📂 Loading Gaussian splats from " << args.inputFile << "..." << std::endl;
    std::vector<gs::GaussianSplat> splats = gs::loadGaussianPly(args.inputFile);
    if (splats.empty()) {
        std::cerr << "❌ Failed to load PLY file or file is empty" << std::endl;
        return 1;
    }
    std::cout << "✅ Loaded " << splats.size() << " Gaussian splats" << std::endl;

    // Initialize CUDA rasterizer and upload data
    std::cout << "\n🎮 Initializing CUDA rasterizer..." << std::endl;
    CudaRasterizer::Rasterizer cuda_rasterizer;
    cuda_rasterizer.loadFromSplats(splats);

    // Test render call (placeholder)
    cuda_rasterizer.render();

    // Clean up GPU memory
    cuda_rasterizer.free();

    std::cout << "\n✅ CUDA initialization complete!" << std::endl;

    return 0;
}