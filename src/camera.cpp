#include "gs/camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cmath>

namespace gs {

bool Camera::loadConfigFromFile(const std::string& path, int camId, CameraConfig& out) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Failed to open camera config: " << path << "\n";
        return false;
    }

    nlohmann::json j;
    try {
        in >> j;
    }
    catch (const std::exception& e) {
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

glm::mat4 Camera::buildViewMatrix(const CameraConfig& cam) {
    glm::vec3 right   = glm::vec3(cam.rotation_c2w[0][0], cam.rotation_c2w[0][1], cam.rotation_c2w[0][2]);
    glm::vec3 up      = glm::vec3(cam.rotation_c2w[1][0], cam.rotation_c2w[1][1], cam.rotation_c2w[1][2]);
    glm::vec3 forward = glm::vec3(cam.rotation_c2w[2][0], cam.rotation_c2w[2][1], cam.rotation_c2w[2][2]);

    glm::vec3 cam_target = cam.position + forward;
    return glm::lookAt(cam.position, cam_target, up);
}

glm::mat4 Camera::buildProjectionMatrix(const CameraConfig& cam) {
    float aspect = static_cast<float>(cam.width) / static_cast<float>(cam.height);
    float fov_y = 2.0f * std::atan(static_cast<float>(cam.height) / (2.0f * cam.fy));
    return glm::perspective(fov_y, aspect, 0.01f, 100.0f);
}

RenderCamera Camera::getDefaultCamera() {
    RenderCamera outCam;
    outCam.width = 1959;
    outCam.height = 1090;
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

    return outCam;
}

bool Camera::setupRenderCamera(const std::string& configFile, int camId, RenderCamera& outCam) {
    if (!configFile.empty()) {
        if (camId < 0) {
            std::cerr << "--cam_id must be provided when using --camera_config" << std::endl;
            return false;
        }

        CameraConfig cam;
        if (!loadConfigFromFile(configFile, camId, cam)) {
            return false;
        }

        outCam.width = cam.width;
        outCam.height = cam.height;
        outCam.position = cam.position;
        outCam.view = buildViewMatrix(cam);
        outCam.proj = buildProjectionMatrix(cam);

        std::cout << "Loaded camera id=" << camId << " (" << cam.width << "x" << cam.height
                  << ", fx=" << cam.fx << ", fy=" << cam.fy << ")" << std::endl;
        std::cout << "Camera position: (" << outCam.position.x << ", " << outCam.position.y << ", " << outCam.position.z << ")" << std::endl;
        return true;
    }

    // Default camera fallback
    outCam = getDefaultCamera();
    std::cout << "Using default camera. Position: (" << outCam.position.x << ", " << outCam.position.y << ", " << outCam.position.z << ")" << std::endl;
    return true;
}

} // namespace gs
