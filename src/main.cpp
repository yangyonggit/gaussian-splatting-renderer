#include <iostream>
#include <vector>

// Include GLM (OpenGL Mathematics)
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Include STB Image Write
// This macro must be defined in exactly one source file to implement the library
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define canvas dimensions
const int WIDTH = 800;
const int HEIGHT = 600;

int main() {
    std::cout << "🚀 MiniGS Renderer V1-CPU Starting..." << std::endl;

    // ------------------------------------------------------------------------
    // 1. Verify GLM Math Library
    // ------------------------------------------------------------------------
    glm::vec3 cameraPos(0.0f, 0.0f, 3.0f);
    glm::vec3 target(0.0f, 0.0f, 0.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    
    // Construct a View Matrix to test GLM linking
    glm::mat4 view = glm::lookAt(cameraPos, target, up);
    std::cout << "✅ GLM math library loaded. View Matrix[0][0]: " << view[0][0] << std::endl;

    // ------------------------------------------------------------------------
    // 2. Initialize Framebuffer
    // ------------------------------------------------------------------------
    // Buffer size: Width * Height * 3 channels (RGB)
    // Initializing with 0 (Black)
    std::vector<unsigned char> framebuffer(WIDTH * HEIGHT * 3, 0);

    // ------------------------------------------------------------------------
    // 3. Draw Dummy Content (Background + Circle)
    // ------------------------------------------------------------------------
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            int index = (y * WIDTH + x) * 3;
            
            // Set Background Color (Dark Grey)
            framebuffer[index + 0] = 30; // R
            framebuffer[index + 1] = 30; // G
            framebuffer[index + 2] = 30; // B

            // Draw a simple Red Circle in the center to verify logic
            float dx = x - WIDTH / 2.0f;
            float dy = y - HEIGHT / 2.0f;
            float radius = 50.0f;

            if (dx * dx + dy * dy < radius * radius) {
                framebuffer[index + 0] = 0; // R
                framebuffer[index + 1] = 128;   // G
                framebuffer[index + 2] = 0;   // B
            }
        }
    }

    // ------------------------------------------------------------------------
    // 4. Output Image to Disk
    // ------------------------------------------------------------------------
    const char* filename = "output_test.png";
    // stbi_write_png arguments: filename, width, height, channels, data, stride_in_bytes
    int result = stbi_write_png(filename, WIDTH, HEIGHT, 3, framebuffer.data(), WIDTH * 3);

    if (result) {
        std::cout << "✅ Image saved successfully to " << filename << std::endl;
    } else {
        std::cerr << "❌ Failed to save image!" << std::endl;
        return 1;
    }

    return 0;
}