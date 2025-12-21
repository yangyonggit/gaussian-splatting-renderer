/**
 * Stage A: Interactive Gaussian Splatting Viewer
 * 
 * Features:
 * - GLFW window with OpenGL rendering
 * - FPS camera with mouse/keyboard controls
 * - Real-time CUDA rasterization (slow path, no interop)
 * - Full-screen quad display of rendered image
 * 
 * Build: cmake --build . && viewer_main [options]
 * 
 * Controls:
 * - WASD: move forward/back/strafe
 * - Q/E: move up/down
 * - Shift+WASD: faster movement
 * - Mouse: look around
 * - ESC: exit
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>

// CRITICAL: GLAD must be included before GLFW or any OpenGL headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "gs/fps_camera.h"
#include "gs/gl_utils.h"
#include "gs/profiler.h"
#include "gs/sh_color.h"
#include "cpu_rasterizer.h"
#include "cuda_rasterizer.h"

// ============================================================
// Configuration
// ============================================================

static const int WINDOW_WIDTH = 1959;   // provided config width
static const int WINDOW_HEIGHT = 1090;  // provided config height
static const char* CAMERA_CONFIG_PATH = "cameras.json";

// Provided camera parameters (intrinsics + pose)
static const glm::vec3 CAM_INIT_POSITION(
    -3.0089893469241797f,
    -0.11086489695181866f,
    -3.7527640949141428f
);

// Rotation matrix rows (provided as camera-to-world, row-major)
static const float CAM_R_ROW0[3] = { 0.876134201218856f,  0.06925962026449776f,  0.47706599800804744f };
static const float CAM_R_ROW1[3] = {-0.04747421839895102f, 0.9972110940209488f, -0.057586739349882114f};
static const float CAM_R_ROW2[3] = {-0.4797239414934443f, 0.027805376500959853f, 0.8769787916452908f};

// Camera-to-world basis (columns: right, up, forward) from the provided row-major matrix
static const glm::vec3 CAM_RIGHT   (CAM_R_ROW0[0], CAM_R_ROW1[0], CAM_R_ROW2[0]);
static const glm::vec3 CAM_UP      (CAM_R_ROW0[1], CAM_R_ROW1[1], CAM_R_ROW2[1]);
static const glm::vec3 CAM_FORWARD (CAM_R_ROW0[2], CAM_R_ROW1[2], CAM_R_ROW2[2]);

static float computeFovYDegFromFy(float fy, float height_px) {
    return glm::degrees(2.0f * std::atan(0.5f * height_px / fy));
}

static void computeYawPitchFromForward(const glm::vec3& forward, float& out_yaw_deg, float& out_pitch_deg) {
    glm::vec3 f = glm::normalize(forward);
    out_pitch_deg = glm::degrees(std::asin(glm::clamp(f.y, -1.0f, 1.0f)));
    out_yaw_deg = glm::degrees(std::atan2(f.z, f.x));
}

static glm::mat4 buildProvidedViewMatrix() {
    // Match CLI: view = lookAt(position, position + forward, up)
    return glm::lookAt(CAM_INIT_POSITION, CAM_INIT_POSITION + CAM_FORWARD, CAM_UP);
}

static glm::mat4 buildProvidedProjMatrix() {
    float fov_y = 2.0f * std::atan(static_cast<float>(WINDOW_HEIGHT) / (2.0f * 1164.6601287484507f));
    float aspect = WINDOW_WIDTH / static_cast<float>(WINDOW_HEIGHT);
    return glm::perspective(fov_y, aspect, 0.01f, 100.0f);
}

// ============================================================
// Global State (for GLFW callbacks)
// ============================================================

static gs::FpsCamera* g_camera = nullptr;
static double g_last_mouse_x = 0.0;
static double g_last_mouse_y = 0.0;
static bool g_mouse_first_move = true;
static bool g_mouse_left_pressed = false;  // Track left mouse button state

// Keyboard state
static bool g_key_w = false;
static bool g_key_a = false;
static bool g_key_s = false;
static bool g_key_d = false;
static bool g_key_q = false;
static bool g_key_e = false;
static bool g_key_shift = false;

// ============================================================
// GLFW Callbacks
// ============================================================

static void glfwErrorCallback(int error, const char* description) {
    std::fprintf(stderr, "GLFW Error (%d): %s\n", error, description);
}

static void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }

    bool pressed = (action == GLFW_PRESS || action == GLFW_REPEAT);

    switch (key) {
        case GLFW_KEY_W: g_key_w = pressed; break;
        case GLFW_KEY_A: g_key_a = pressed; break;
        case GLFW_KEY_S: g_key_s = pressed; break;
        case GLFW_KEY_D: g_key_d = pressed; break;
        case GLFW_KEY_Q: g_key_q = pressed; break;
        case GLFW_KEY_E: g_key_e = pressed; break;
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:
            g_key_shift = pressed;
            break;
    }
}

static void glfwMouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (g_mouse_first_move) {
        g_last_mouse_x = xpos;
        g_last_mouse_y = ypos;
        g_mouse_first_move = false;
        return;
    }

    double dx = xpos - g_last_mouse_x;
    double dy = ypos - g_last_mouse_y;
    g_last_mouse_x = xpos;
    g_last_mouse_y = ypos;

    // Only rotate camera when left mouse button is pressed
    if (g_camera && g_mouse_left_pressed) {
        g_camera->processMouse(static_cast<float>(dx), static_cast<float>(dy));
    }
}

static void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_mouse_left_pressed = (action == GLFW_PRESS);
    }
}

// ============================================================
// Shader Sources
// ============================================================

static const char* VERTEX_SHADER_SOURCE = R"(
#version 330 core
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
)";

static const char* FRAGMENT_SHADER_SOURCE = R"(
#version 330 core
in vec2 vTexCoord;

uniform sampler2D tex;

out vec4 fragColor;

void main() {
    // Sample texture directly; upload already matches GL's origin
    fragColor = texture(tex, vTexCoord);
}
)";

// ============================================================
// Forward Declarations
// ============================================================

struct GLResources {
    GLuint program = 0;
    GLuint pbo = 0;
    cudaGraphicsResource* cuda_pbo_resource = nullptr;
    gs::gl::FullscreenQuad quad;
    gs::gl::Texture render_target;
};

static bool initGLFW();
static GLFWwindow* createWindow();
static bool initOpenGL();
static void setupCallbacks(GLFWwindow* window);
static gs::FpsCamera createCamera();
static bool createGLResources(GLResources& resources);
static bool loadAndPrepareScene(
    const char* ply_path,
    CpuRasterizer::Rasterizer& cpu_prep,
    CudaRasterizer::Rasterizer& cuda_rasterizer,
    std::vector<gs::GaussianSplat>& splats,
    std::vector<gs::ScreenSplat>& screen_splats
);
static void mainLoop(
    GLFWwindow* window,
    gs::FpsCamera& camera,
    GLResources& resources,
    CpuRasterizer::Rasterizer& cpu_prep,
    CudaRasterizer::Rasterizer& cuda_rasterizer,
    std::vector<gs::GaussianSplat>& splats,
    std::vector<gs::ScreenSplat>& screen_splats
);
static void cleanup(GLFWwindow* window, GLResources& resources, CudaRasterizer::Rasterizer& cuda_rasterizer);

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    const char* ply_path = (argc > 1) ? argv[1] : "train.ply";
    
    std::cout << "🎮 Stage A: Interactive Gaussian Splatting Viewer" << std::endl;
    std::cout << "   WASD: move | Q/E: up/down | Mouse: look | ESC: quit" << std::endl;
    std::cout << "   Loading: " << ply_path << std::endl;

    if (!initGLFW()) return 1;

    GLFWwindow* window = createWindow();
    if (!window) {
        glfwTerminate();
        return 1;
    }

    if (!initOpenGL()) {
        glfwTerminate();
        return 1;
    }

    setupCallbacks(window);
    gs::FpsCamera camera = createCamera();
    g_camera = &camera;

    GLResources resources;
    if (!createGLResources(resources)) {
        glfwTerminate();
        return 1;
    }

    CpuRasterizer::Rasterizer cpu_prep;
    CudaRasterizer::Rasterizer cuda_rasterizer;
    std::vector<gs::GaussianSplat> splats;
    std::vector<gs::ScreenSplat> screen_splats;

    if (!loadAndPrepareScene(ply_path, cpu_prep, cuda_rasterizer, splats, screen_splats)) {
        cleanup(window, resources, cuda_rasterizer);
        glfwTerminate();
        return 1;
    }

    mainLoop(window, camera, resources, cpu_prep, cuda_rasterizer, splats, screen_splats);

    cleanup(window, resources, cuda_rasterizer);
    glfwTerminate();

    std::cout << "✅ Goodbye!" << std::endl;
    return 0;
}

// ============================================================
// Implementation
// ============================================================

bool initGLFW() {
    std::cout << "\n📺 Initializing GLFW..." << std::endl;
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    return true;
}

GLFWwindow* createWindow() {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
                                          "MiniGS Viewer - Stage A", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Vsync
    return window;
}

bool initOpenGL() {
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to load OpenGL extensions" << std::endl;
        return false;
    }

    std::cout << "✅ OpenGL " << glGetString(GL_VERSION) << std::endl;
    return true;
}

void setupCallbacks(GLFWwindow* window) {
    glfwSetKeyCallback(window, glfwKeyCallback);
    glfwSetCursorPosCallback(window, glfwMouseCallback);
    glfwSetMouseButtonCallback(window, glfwMouseButtonCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

gs::FpsCamera createCamera() {
    std::cout << "\n📷 Initializing camera..." << std::endl;
    float init_yaw_deg = -90.0f;
    float init_pitch_deg = 0.0f;
    computeYawPitchFromForward(CAM_FORWARD, init_yaw_deg, init_pitch_deg);

    const float init_fov_y_deg = computeFovYDegFromFy(1164.6601287484507f, static_cast<float>(WINDOW_HEIGHT));

    return gs::FpsCamera(
        CAM_INIT_POSITION,
        init_yaw_deg,
        init_pitch_deg,
        init_fov_y_deg,
        10.0f,
        0.1f
    );
}

bool createGLResources(GLResources& resources) {
    std::cout << "\n🎨 Creating OpenGL resources..." << std::endl;

    GLuint vs = gs::gl::compileShader(GL_VERTEX_SHADER, VERTEX_SHADER_SOURCE);
    GLuint fs = gs::gl::compileShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);
    if (!vs || !fs) {
        std::cerr << "Shader compilation failed" << std::endl;
        return false;
    }

    resources.program = gs::gl::linkProgram(vs, fs);
    if (!resources.program) {
        std::cerr << "Program linking failed" << std::endl;
        return false;
    }

    if (!resources.quad.init(resources.program)) {
        std::cerr << "Failed to initialize quad" << std::endl;
        return false;
    }

    if (!resources.render_target.init(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA8)) {
        std::cerr << "Failed to initialize render target texture" << std::endl;
        return false;
    }

    glGenBuffers(1, &resources.pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, resources.pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WINDOW_WIDTH * WINDOW_HEIGHT * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (cudaGraphicsGLRegisterBuffer(&resources.cuda_pbo_resource, resources.pbo, 
                                     cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
        std::cerr << "Failed to register PBO with CUDA" << std::endl;
        return false;
    }

    std::cout << "✅ GL resources created" << std::endl;
    return true;
}

bool loadAndPrepareScene(
    const char* ply_path,
    CpuRasterizer::Rasterizer& cpu_prep,
    CudaRasterizer::Rasterizer& cuda_rasterizer,
    std::vector<gs::GaussianSplat>& splats,
    std::vector<gs::ScreenSplat>& screen_splats
) {
    std::cout << "\n🚀 Preparing CUDA renderer..." << std::endl;

    glm::mat4 view = buildProvidedViewMatrix();
    glm::mat4 proj = buildProvidedProjMatrix();

    if (!cpu_prep.prepareForCuda(ply_path, view, proj,
                                  WINDOW_WIDTH, WINDOW_HEIGHT,
                                  splats, screen_splats)) {
        std::cerr << "Failed to preprocess scene" << std::endl;
        return false;
    }

    std::cout << "✅ Scene loaded: " << splats.size() << " splats" << std::endl;

    std::cout << "📤 Uploading scene-static SH data to GPU..." << std::endl;
    if (!cuda_rasterizer.uploadSceneData(splats)) {
        std::cerr << "Failed to upload scene data to GPU" << std::endl;
        return false;
    }

    return true;
}

void mainLoop(
    GLFWwindow* window,
    gs::FpsCamera& camera,
    GLResources& resources,
    CpuRasterizer::Rasterizer& cpu_prep,
    CudaRasterizer::Rasterizer& cuda_rasterizer,
    std::vector<gs::GaussianSplat>& splats,
    std::vector<gs::ScreenSplat>& screen_splats
) {
    std::cout << "\n▶️  Entering main loop..." << std::endl;
    double last_time = glfwGetTime();
    int frame_count = 0;

    while (!glfwWindowShouldClose(window)) {
        double current_time = glfwGetTime();
        float delta_time = static_cast<float>(current_time - last_time);
        last_time = current_time;

        camera.processKeyboard(
            g_key_w ? 1.0f : 0.0f,
            g_key_s ? 1.0f : 0.0f,
            g_key_a ? 1.0f : 0.0f,
            g_key_d ? 1.0f : 0.0f,
            g_key_e ? 1.0f : 0.0f,
            g_key_q ? 1.0f : 0.0f,
            delta_time,
            g_key_shift
        );

        {
            glm::mat4 view = camera.getViewMatrix();
            float aspect = static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT);
            glm::mat4 proj = camera.getProjMatrix(aspect);

            if (!cpu_prep.reprojectSplats(splats, view, proj, WINDOW_WIDTH, WINDOW_HEIGHT, screen_splats)) {
                std::cerr << "Failed to reproject splats" << std::endl;
                break;
            }
        }

        {
            ScopedTimer timer("cuda_render_frame");
            cudaGraphicsMapResources(1, &resources.cuda_pbo_resource);
            void* d_ptr = nullptr;
            size_t mapped_size = 0;
            cudaGraphicsResourceGetMappedPointer(&d_ptr, &mapped_size, resources.cuda_pbo_resource);

            if (!cuda_rasterizer.render_cuda_to_rgba8_device(
                    screen_splats, camera.getPosition(),
                    WINDOW_WIDTH, WINDOW_HEIGHT,
                    reinterpret_cast<unsigned char*>(d_ptr))) {
                std::cerr << "CUDA rendering (PBO) failed" << std::endl;
                cudaGraphicsUnmapResources(1, &resources.cuda_pbo_resource);
                break;
            }

            cudaGraphicsUnmapResources(1, &resources.cuda_pbo_resource);

            glBindTexture(GL_TEXTURE_2D, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, resources.pbo);
            resources.render_target.bind(0);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                            GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        resources.render_target.bind(0);
        GLint tex_loc = glGetUniformLocation(resources.program, "tex");
        glUniform1i(tex_loc, 0);

        resources.quad.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();

        frame_count++;
        if (frame_count % 60 == 0) {
            std::printf("Frame %d | pos: (%.1f, %.1f, %.1f) | yaw: %.1f | pitch: %.1f | dt: %.3f ms\n",
                       frame_count,
                       camera.getPosition().x, camera.getPosition().y, camera.getPosition().z,
                       camera.getYaw(), camera.getPitch(),
                       delta_time * 1000.0f);
        }
    }
}

void cleanup(GLFWwindow* window, GLResources& resources, CudaRasterizer::Rasterizer& cuda_rasterizer) {
    std::cout << "\n🧹 Cleaning up..." << std::endl;
    resources.quad.cleanup();
    resources.render_target.cleanup();
    if (resources.cuda_pbo_resource) cudaGraphicsUnregisterResource(resources.cuda_pbo_resource);
    if (resources.pbo) glDeleteBuffers(1, &resources.pbo);
    if (resources.program) glDeleteProgram(resources.program);
    cuda_rasterizer.free();
    glfwDestroyWindow(window);
}
