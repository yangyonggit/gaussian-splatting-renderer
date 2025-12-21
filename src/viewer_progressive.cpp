/**
 * Progressive Viewer: renders in small splat chunks to reveal the image gradually.
 * Based on viewer_main but submits ~1% of splats per draw starting from screen center.
 * Buffer is only cleared when the camera changes.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// GL
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

static const int WINDOW_WIDTH = 1959;
static const int WINDOW_HEIGHT = 1090;
static const char* PLY_PATH = "train.ply";
static const float BG_COLOR = 30.0f / 255.0f;

// Provided camera parameters (same as viewer_main)
static const glm::vec3 CAM_INIT_POSITION(
    -3.0089893469241797f,
    -0.11086489695181866f,
    -3.7527640949141428f
);

static const float CAM_R_ROW0[3] = { 0.876134201218856f,  0.06925962026449776f,  0.47706599800804744f };
static const float CAM_R_ROW1[3] = {-0.04747421839895102f, 0.9972110940209488f, -0.057586739349882114f};
static const float CAM_R_ROW2[3] = {-0.4797239414934443f, 0.027805376500959853f, 0.8769787916452908f};

static const glm::vec3 CAM_RIGHT   (CAM_R_ROW0[0], CAM_R_ROW1[0], CAM_R_ROW2[0]);
static const glm::vec3 CAM_UP      (CAM_R_ROW0[1], CAM_R_ROW1[1], CAM_R_ROW2[1]);
static const glm::vec3 CAM_FORWARD (CAM_R_ROW0[2], CAM_R_ROW1[2], CAM_R_ROW2[2]);

// Global state for callbacks
static gs::FpsCamera* g_camera = nullptr;
static double g_last_mouse_x = 0.0;
static double g_last_mouse_y = 0.0;
static bool g_mouse_first_move = true;
static bool g_mouse_left_pressed = false;

static bool g_key_w = false;
static bool g_key_a = false;
static bool g_key_s = false;
static bool g_key_d = false;
static bool g_key_q = false;
static bool g_key_e = false;
static bool g_key_shift = false;

static float computeFovYDegFromFy(float fy, float height_px) {
    return glm::degrees(2.0f * std::atan(0.5f * height_px / fy));
}

static void computeYawPitchFromForward(const glm::vec3& forward, float& out_yaw_deg, float& out_pitch_deg) {
    glm::vec3 f = glm::normalize(forward);
    out_pitch_deg = glm::degrees(std::asin(glm::clamp(f.y, -1.0f, 1.0f)));
    out_yaw_deg = glm::degrees(std::atan2(f.z, f.x));
}

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

    if (g_camera && g_mouse_left_pressed) {
        g_camera->processMouse(static_cast<float>(dx), static_cast<float>(dy));
    }
}

static void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_mouse_left_pressed = (action == GLFW_PRESS);
    }
}

static const char* VERT_SRC = R"(
#version 330 core
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main(){ gl_Position = vec4(aPosition,0.0,1.0); vTexCoord=aTexCoord; }
)";

static const char* FRAG_SRC = R"(
#version 330 core
in vec2 vTexCoord; uniform sampler2D tex; out vec4 fragColor;
void main(){ fragColor = texture(tex,vTexCoord); }
)";

struct CameraState {
    glm::vec3 pos{};
    float yaw = 0.0f;
    float pitch = 0.0f;
};

static bool cameraChanged(const gs::FpsCamera& cam, const CameraState& last) {
    if (glm::length(cam.getPosition() - last.pos) > 1e-4f) return true;
    if (std::abs(cam.getYaw() - last.yaw) > 1e-4f) return true;
    if (std::abs(cam.getPitch() - last.pitch) > 1e-4f) return true;
    return false;
}

static void updateCameraState(const gs::FpsCamera& cam, CameraState& out) {
    out.pos = cam.getPosition();
    out.yaw = cam.getYaw();
    out.pitch = cam.getPitch();
}

int main() {
    std::cout << "🎮 Progressive Gaussian Splatting Viewer" << std::endl;

    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) { std::cerr << "Failed to init GLFW\n"; return 1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "MiniGS Progressive Viewer", nullptr, nullptr);
    if (!window) { std::cerr << "Failed to create window\n"; glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, glfwKeyCallback);
    glfwSetCursorPosCallback(window, glfwMouseCallback);
    glfwSetMouseButtonCallback(window, glfwMouseButtonCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to load GL\n"; return 1;
    }

    GLuint vs = gs::gl::compileShader(GL_VERTEX_SHADER, VERT_SRC);
    GLuint fs = gs::gl::compileShader(GL_FRAGMENT_SHADER, FRAG_SRC);
    GLuint prog = gs::gl::linkProgram(vs, fs);
    if (!vs || !fs || !prog) { std::cerr << "Shader error\n"; return 1; }

    gs::gl::FullscreenQuad quad;
    if (!quad.init(prog)) { std::cerr << "Quad init failed\n"; return 1; }

    gs::gl::Texture render_tex;
    if (!render_tex.init(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA8)) {
        std::cerr << "Texture init failed\n"; return 1;
    }

    GLuint pbo = 0; glGenBuffers(1,&pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WINDOW_WIDTH*WINDOW_HEIGHT*4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsResource* cuda_pbo = nullptr;
    if (cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
        std::cerr << "CUDA register PBO failed\n"; return 1;
    }

    CpuRasterizer::Rasterizer cpu;
    CudaRasterizer::Rasterizer cuda;
    std::vector<gs::GaussianSplat> splats;
    std::vector<gs::ScreenSplat> screen_splats;

    // Initial camera (match viewer_main)
    float init_yaw_deg = -90.0f;
    float init_pitch_deg = 0.0f;
    computeYawPitchFromForward(CAM_FORWARD, init_yaw_deg, init_pitch_deg);

    gs::FpsCamera cam(CAM_INIT_POSITION, init_yaw_deg, init_pitch_deg,
                      60.0f, 10.0f, 0.1f);
    g_camera = &cam;
    CameraState last_cam{};
    updateCameraState(cam, last_cam);

    // Initial projection using provided fy from viewer_main
    float init_fov_y_deg = computeFovYDegFromFy(1164.6601287484507f, static_cast<float>(WINDOW_HEIGHT));
    cam.setFovY(init_fov_y_deg);

    // Load scene once
    {
        glm::mat4 view = cam.getViewMatrix();
        glm::mat4 proj = cam.getProjMatrix(static_cast<float>(WINDOW_WIDTH)/WINDOW_HEIGHT);
        if (!cpu.prepareForCuda(PLY_PATH, view, proj, WINDOW_WIDTH, WINDOW_HEIGHT, splats, screen_splats)) {
            std::cerr << "Scene prep failed\n"; return 1;
        }
    }
    if (!cuda.uploadSceneData(splats)) { std::cerr << "Scene upload failed\n"; return 1; }

    // Progressive state
    std::vector<gs::ScreenSplat> sorted_splats;
    std::vector<gs::ScreenSplat> chunk_splats;
    size_t rendered_count = 0;
    size_t chunk_size = 1;

    auto resort_and_reset = [&](const glm::mat4& view, const glm::mat4& proj){
        if (!cpu.reprojectSplats(splats, view, proj, WINDOW_WIDTH, WINDOW_HEIGHT, screen_splats)) {
            std::cerr << "Reproject failed\n"; return false;
        }
        sorted_splats = screen_splats;
        const float cx = 0.5f * WINDOW_WIDTH;
        const float cy = 0.5f * WINDOW_HEIGHT;
        std::stable_sort(sorted_splats.begin(), sorted_splats.end(), [cx, cy](const gs::ScreenSplat& a, const gs::ScreenSplat& b){
            float da = (a.sx - cx)*(a.sx - cx) + (a.sy - cy)*(a.sy - cy);
            float db = (b.sx - cx)*(b.sx - cx) + (b.sy - cy)*(b.sy - cy);
            return da < db;
        });
        rendered_count = 0;
        // Reveal ~1/10000 of splats per chunk (at least 1)
        chunk_size = std::max<size_t>(sorted_splats.size() / 100, 1);
        glClearColor(BG_COLOR, BG_COLOR, BG_COLOR, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        return true;
    };

    // initial sort
    resort_and_reset(cam.getViewMatrix(), cam.getProjMatrix(static_cast<float>(WINDOW_WIDTH)/WINDOW_HEIGHT));

    double last_time = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double current_time = glfwGetTime();
        float delta_time = static_cast<float>(current_time - last_time);
        last_time = current_time;

        // Update camera from keyboard input
        cam.processKeyboard(
            g_key_w ? 1.0f : 0.0f,
            g_key_s ? 1.0f : 0.0f,
            g_key_a ? 1.0f : 0.0f,
            g_key_d ? 1.0f : 0.0f,
            g_key_e ? 1.0f : 0.0f,
            g_key_q ? 1.0f : 0.0f,
            delta_time,
            g_key_shift
        );

        bool cam_dirty = cameraChanged(cam, last_cam);
        if (cam_dirty) {
            updateCameraState(cam, last_cam);
            if (!resort_and_reset(cam.getViewMatrix(), cam.getProjMatrix(static_cast<float>(WINDOW_WIDTH)/WINDOW_HEIGHT))) break;
        }

        bool did_render = false;
        if (rendered_count < sorted_splats.size()) {
            size_t next = std::min(rendered_count + chunk_size, sorted_splats.size());
            chunk_splats.assign(sorted_splats.begin(), sorted_splats.begin() + next);

            // Maintain correct alpha compositing: draw back-to-front by depth
            std::sort(chunk_splats.begin(), chunk_splats.end(), [](const gs::ScreenSplat& a, const gs::ScreenSplat& b){
                return a.depth > b.depth;
            });

            cudaGraphicsMapResources(1, &cuda_pbo);
            void* d_ptr = nullptr; size_t sz = 0;
            cudaGraphicsResourceGetMappedPointer(&d_ptr, &sz, cuda_pbo);

            if (!cuda.render_cuda_to_rgba8_device(chunk_splats, cam.getPosition(), WINDOW_WIDTH, WINDOW_HEIGHT, reinterpret_cast<unsigned char*>(d_ptr))) {
                std::cerr << "CUDA render failed\n"; cudaGraphicsUnmapResources(1, &cuda_pbo); break;
            }
            cudaGraphicsUnmapResources(1, &cuda_pbo);

            glBindTexture(GL_TEXTURE_2D, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            render_tex.bind(0);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            rendered_count = next;
            did_render = true;
        }

        // Present
        render_tex.bind(0);
        glUseProgram(prog);
        glUniform1i(glGetUniformLocation(prog, "tex"), 0);
        quad.draw();
        glfwSwapBuffers(window);

        // When finished all splats and no new camera movement, we simply keep showing the last texture.
        if (!did_render) {
            // Slight sleep could be added to reduce CPU, omitted for simplicity.
        }
    }

    // Cleanup
    quad.cleanup();
    render_tex.cleanup();
    if (cuda_pbo) cudaGraphicsUnregisterResource(cuda_pbo);
    if (pbo) glDeleteBuffers(1,&pbo);
    glDeleteProgram(prog);
    cuda.free();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
