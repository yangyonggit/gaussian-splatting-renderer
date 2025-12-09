#pragma once

#include <glm/glm.hpp>
#include <string>

namespace gs {

/**
 * Camera configuration loaded from file or defaults
 */
struct CameraConfig {
    int width = 1959;
    int height = 1090;
    float fx = 1159.5880733038064f;
    float fy = 1164.6601287484507f;
    glm::vec3 position{ 2.2517861865205795f, 0.3663530997455256f, 3.8251153433115346f };
    glm::mat3 rotation_c2w{ 1.0f }; // camera-to-world rotation (columns: right, up, forward)
};

/**
 * Render camera with view and projection matrices
 */
struct RenderCamera {
    int width = 1959;
    int height = 1090;
    glm::vec3 position{ 0.0f };
    glm::mat4 view{ 1.0f };
    glm::mat4 proj{ 1.0f };
};

/**
 * Camera utilities: loading configs, building matrices, setup
 */
class Camera {
public:
    /**
     * Load camera configuration from JSON file
     * @param path Path to JSON camera list file
     * @param camId Camera ID to select from the list
     * @param out Output camera config
     * @return true if successful, false otherwise
     */
    static bool loadConfigFromFile(const std::string& path, int camId, CameraConfig& out);

    /**
     * Build view matrix from camera configuration
     * @param cam Camera configuration with position and rotation
     * @return View matrix
     */
    static glm::mat4 buildViewMatrix(const CameraConfig& cam);

    /**
     * Build projection matrix from camera configuration
     * @param cam Camera configuration with focal lengths and dimensions
     * @return Projection matrix
     */
    static glm::mat4 buildProjectionMatrix(const CameraConfig& cam);

    /**
     * Setup render camera from command line arguments or defaults
     * @param configFile Path to camera config file (optional)
     * @param camId Camera ID to select (required if configFile is provided)
     * @param outCam Output render camera
     * @return true if successful, false otherwise
     */
    static bool setupRenderCamera(const std::string& configFile, int camId, RenderCamera& outCam);

    /**
     * Get default camera (fallback when no config is provided)
     * @return Default render camera
     */
    static RenderCamera getDefaultCamera();
};

} // namespace gs
