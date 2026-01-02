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
#include <fstream>

// CRITICAL: GLAD must be included before GLFW or any OpenGL headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <nlohmann/json.hpp>

#include "gs/fps_camera.h"
#include "gs/gl_utils.h"
#include "gs/profiler.h"
#include "gs/sh_color.h"
#include "gs/nvtx_helper.h"
#include "gs/ply_loader.h"
#include "cpu_rasterizer.h"
#include "cuda_rasterizer.h"

using json = nlohmann::json;

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
// Camera Config Loading
// ============================================================

struct CameraConfig {
    glm::vec3 position;
    float fx, fy, cx, cy;
    glm::mat4 view;
    glm::mat4 proj;
    bool valid = false;
};

static CameraConfig loadCameraFromJson(const std::string& json_path, int camera_id = 0) {
    CameraConfig config;
    
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cout << "⚠️  Camera config not found at: " << json_path << std::endl;
        return config;
    }

    try {
        json j;
        file >> j;
        // The file is a top-level array of camera entries
        if (j.is_array() && !j.empty()) {
            // Prefer selecting by matching `id`; fall back to clamped index
            int chosen_index = -1;
            for (size_t i = 0; i < j.size(); ++i) {
                const auto& node = j[i];
                if (node.contains("id") && node["id"].is_number_integer() && node["id"].get<int>() == camera_id) {
                    chosen_index = static_cast<int>(i);
                    break;
                }
            }
            if (chosen_index == -1) {
                chosen_index = std::min<int>(camera_id, static_cast<int>(j.size() - 1));
                if (chosen_index < 0) chosen_index = 0;
            }

            const auto& cam = j[chosen_index];

            // Position
            if (cam.contains("position") && cam["position"].is_array() && cam["position"].size() == 3) {
                config.position = glm::vec3(
                    cam["position"][0].get<float>(),
                    cam["position"][1].get<float>(),
                    cam["position"][2].get<float>()
                );
            }

            // Rotation rows: 3x3 matrix (camera-to-world, row-major)
            bool have_rotation = (
                cam.contains("rotation") && cam["rotation"].is_array() && cam["rotation"].size() == 3 &&
                cam["rotation"][0].is_array() && cam["rotation"][1].is_array() && cam["rotation"][2].is_array() &&
                cam["rotation"][0].size() == 3 && cam["rotation"][1].size() == 3 && cam["rotation"][2].size() == 3
            );

            if (have_rotation) {
                float r00 = cam["rotation"][0][0].get<float>();
                float r01 = cam["rotation"][0][1].get<float>();
                float r02 = cam["rotation"][0][2].get<float>();
                float r10 = cam["rotation"][1][0].get<float>();
                float r11 = cam["rotation"][1][1].get<float>();
                float r12 = cam["rotation"][1][2].get<float>();
                float r20 = cam["rotation"][2][0].get<float>();
                float r21 = cam["rotation"][2][1].get<float>();
                float r22 = cam["rotation"][2][2].get<float>();

                // Convert row-major rotation to column vectors (GLM is column-major)
                glm::vec3 right(r00, r10, r20);
                glm::vec3 up(r01, r11, r21);
                glm::vec3 forward(r02, r12, r22);

                glm::mat4 c2w(1.0f);
                c2w[0] = glm::vec4(right, 0.0f);
                c2w[1] = glm::vec4(up, 0.0f);
                c2w[2] = glm::vec4(forward, 0.0f);
                c2w[3] = glm::vec4(config.position, 1.0f);

                // world-to-camera = inverse(camera-to-world)
                config.view = glm::inverse(c2w);
            } else {
                // Fallback if rotation missing
                config.view = buildProvidedViewMatrix();
            }

            // Intrinsics
            if (cam.contains("fx")) config.fx = cam["fx"].get<float>();
            if (cam.contains("fy")) config.fy = cam["fy"].get<float>();

            int w = WINDOW_WIDTH;
            int h = WINDOW_HEIGHT;
            if (cam.contains("width")) w = cam["width"].get<int>();
            if (cam.contains("height")) h = cam["height"].get<int>();

            float fov_y = 2.0f * std::atan(static_cast<float>(h) / (2.0f * config.fy));
            float aspect = static_cast<float>(w) / static_cast<float>(h);
            config.proj = glm::perspective(fov_y, aspect, 0.01f, 100.0f);

            config.valid = true;
            std::cout << "✅ Loaded camera id=" << camera_id << " (index=" << chosen_index << ") from: " << json_path << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "⚠️  Failed to parse JSON: " << e.what() << std::endl;
    }

    file.close();
    return config;
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
    gs::NvtxRange nvtx_main("Viewer_Main");
    
    // Parse command line arguments
    const char* ply_path = nullptr;
    std::string camera_config_path = "cameras.json";  // default
    int camera_id = 0;  // default

    // Simple argument parser: viewer_main <ply> [camera_config] [camera_id]
    if (argc > 1) {
        ply_path = argv[1];
    } else {
        std::cerr << "Usage: viewer_main <ply_file> [camera_config.json] [camera_id]" << std::endl;
        return 1;
    }

    if (argc > 2) {
        camera_config_path = argv[2];
    }

    if (argc > 3) {
        camera_id = std::atoi(argv[3]);
    }
    
    std::cout << "🎮 Stage A: Interactive Gaussian Splatting Viewer" << std::endl;
    std::cout << "   WASD: move | Q/E: up/down | Mouse: look | ESC: quit" << std::endl;
    std::cout << "   PLY: " << ply_path << std::endl;
    std::cout << "   Camera config: " << camera_config_path << " (id=" << camera_id << ")" << std::endl;

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

    // Try to load camera config from JSON; if not available, use hardcoded defaults
    CameraConfig cam_config = loadCameraFromJson(camera_config_path, camera_id);
    
    glm::mat4 view_matrix, proj_matrix;
    if (cam_config.valid) {
        view_matrix = cam_config.view;
        proj_matrix = cam_config.proj;
        std::cout << "📷 Using camera config from JSON" << std::endl;

        // Sync interactive camera with loaded config
        camera.setPosition(cam_config.position);
        // Derive yaw/pitch from forward vector in c2w
        glm::mat4 c2w = glm::inverse(view_matrix);
        glm::vec3 forward(c2w[2].x, c2w[2].y, c2w[2].z);
        float cfg_yaw = -90.0f, cfg_pitch = 0.0f;
        computeYawPitchFromForward(forward, cfg_yaw, cfg_pitch);
        camera.setYaw(cfg_yaw);
        camera.setPitch(cfg_pitch);

        // Set camera FOV from fy (fallback to provided value if missing)
        float fy_for_fov = (cam_config.fy > 0.0f) ? cam_config.fy : 1164.6601287484507f;
        camera.setFovY(computeFovYDegFromFy(fy_for_fov, static_cast<float>(WINDOW_HEIGHT)));
    } else {
        view_matrix = buildProvidedViewMatrix();
        proj_matrix = buildProvidedProjMatrix();
        std::cout << "📷 Using hardcoded default camera" << std::endl;
    }

    // Reproject with the selected camera
    if (!cpu_prep.reprojectSplats(splats, view_matrix, proj_matrix, WINDOW_WIDTH, WINDOW_HEIGHT, screen_splats)) {
        std::cerr << "Failed to reproject splats" << std::endl;
        cleanup(window, resources, cuda_rasterizer);
        glfwTerminate();
        return 1;
    }
    std::cout << "✅ Projected " << screen_splats.size() << " visible splats" << std::endl;

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
    gs::NvtxRange nvtx_scene("Load_And_Prepare_Scene");
    
    std::cout << "\n🚀 Preparing CUDA renderer..." << std::endl;

    (void)cpu_prep;
    screen_splats.clear();

    splats = gs::loadGaussianPly(ply_path);
    if (splats.empty()) {
        std::cerr << "Failed to load PLY file or file is empty!" << std::endl;
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

        (void)cpu_prep;
        (void)splats;
        (void)screen_splats;

        glm::mat4 view = camera.getViewMatrix();
        float aspect = static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT);
        glm::mat4 proj = camera.getProjMatrix(aspect);

        {
            gs::NvtxRange nvtx_cuda_frame("CUDA_Render_Frame_V2");
            ScopedTimer timer("cuda_render_frame");
            cudaGraphicsMapResources(1, &resources.cuda_pbo_resource);
            void* d_ptr = nullptr;
            size_t mapped_size = 0;
            cudaGraphicsResourceGetMappedPointer(&d_ptr, &mapped_size, resources.cuda_pbo_resource);

            {
                gs::NvtxRange nvtx_render_call("Render_CUDA_To_PBO_V2");
                if (!cuda_rasterizer.render_cuda_to_rgba8_device_v2(
                        view, proj, camera.getPosition(),
                        WINDOW_WIDTH, WINDOW_HEIGHT,
                        reinterpret_cast<unsigned char*>(d_ptr))) {
                    std::cerr << "CUDA rendering (PBO) failed" << std::endl;
                    cudaGraphicsUnmapResources(1, &resources.cuda_pbo_resource);
                    break;
                }
            }

            cudaGraphicsUnmapResources(1, &resources.cuda_pbo_resource);

            {
                gs::NvtxRange nvtx_texupload("Update_Texture_From_PBO");
                glBindTexture(GL_TEXTURE_2D, 0);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, resources.pbo);
                resources.render_target.bind(0);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                                GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            }
        }

        {
            gs::NvtxRange nvtx_gl_display("OpenGL_Display_Frame");
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            resources.render_target.bind(0);
            GLint tex_loc = glGetUniformLocation(resources.program, "tex");
            glUniform1i(tex_loc, 0);

            resources.quad.draw();

            glfwSwapBuffers(window);
        }
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
