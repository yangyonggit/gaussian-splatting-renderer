#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

namespace gs {

/**
 * Interactive FPS-style camera for real-time viewing.
 * 
 * Controls:
 * - Mouse: look around (yaw/pitch)
 * - WASD: forward/back/strafe left/right
 * - Q/E: move down/up
 * - Shift: faster movement (optional, user responsibility to detect)
 */
class FpsCamera {
public:
    FpsCamera(
        const glm::vec3& position = glm::vec3(0.0f, 0.0f, 5.0f),
        float yaw_deg = -90.0f,
        float pitch_deg = 0.0f,
        float fov_y_deg = 45.0f,
        float move_speed = 5.0f,
        float mouse_sensitivity = 0.1f
    )
        : position_(position)
        , yaw_deg_(yaw_deg)
        , pitch_deg_(pitch_deg)
        , fov_y_deg_(fov_y_deg)
        , move_speed_(move_speed)
        , mouse_sensitivity_(mouse_sensitivity)
    {
        updateVectors();
    }

    /**
     * Process mouse movement (delta in pixels)
     */
    void processMouse(float dx, float dy) {
        yaw_deg_ -= dx * mouse_sensitivity_;
        pitch_deg_ += dy * mouse_sensitivity_;

        // Clamp pitch to avoid gimbal lock
        if (pitch_deg_ > 89.0f) pitch_deg_ = 89.0f;
        if (pitch_deg_ < -89.0f) pitch_deg_ = -89.0f;

        updateVectors();
    }

    /**
     * Process keyboard input (WASD, QE for up/down)
     * Keys should be a function or array indicating which keys are pressed.
     * For simplicity, use: bool keys[256] indexed by GLFW key constants.
     * Or pass individual bools: forward, back, left, right, up, down, fast.
     * 
     * delta_time in seconds.
     */
    void processKeyboard(float forward, float back, float left, float right, 
                        float up, float down, float delta_time, bool fast = false) {
        float speed = move_speed_ * (fast ? 0.5f : 0.05f) * delta_time;

        glm::vec3 move(0.0f);
        if (forward > 0.5f) move += front_;
        if (back > 0.5f) move -= front_;
        if (left > 0.5f) move += right_;
        if (right > 0.5f) move -= right_;
        if (up > 0.5f) move += glm::vec3(0.0f, 1.0f, 0.0f);
        if (down > 0.5f) move -= glm::vec3(0.0f, 1.0f, 0.0f);

        position_ += move * speed;
    }

    /**
     * Get view matrix (world-to-camera)
     */
    glm::mat4 getViewMatrix() const {
        return glm::lookAt(position_, position_ + front_, up_);
    }

    /**
     * Get projection matrix (camera-to-clip, perspective)
     * aspect_ratio = width / height
     */
    glm::mat4 getProjMatrix(float aspect_ratio, float near_plane = 0.1f, float far_plane = 1000.0f) const {
        return glm::perspective(glm::radians(fov_y_deg_), aspect_ratio, near_plane, far_plane);
    }

    // Getters
    const glm::vec3& getPosition() const { return position_; }
    float getYaw() const { return yaw_deg_; }
    float getPitch() const { return pitch_deg_; }
    float getFovY() const { return fov_y_deg_; }
    const glm::vec3& getFront() const { return front_; }
    const glm::vec3& getRight() const { return right_; }
    const glm::vec3& getUp() const { return up_; }

    // Setters
    void setPosition(const glm::vec3& pos) { position_ = pos; }
    void setYaw(float yaw_deg) { yaw_deg_ = yaw_deg; updateVectors(); }
    void setPitch(float pitch_deg) { pitch_deg_ = pitch_deg; updateVectors(); }
    void setFovY(float fov_y_deg) { fov_y_deg_ = fov_y_deg; }

private:
    glm::vec3 position_;
    float yaw_deg_;
    float pitch_deg_;
    float fov_y_deg_;
    float move_speed_;
    float mouse_sensitivity_;

    glm::vec3 front_;
    glm::vec3 right_;
    glm::vec3 up_;

    void updateVectors() {
        // Convert degrees to radians
        float yaw_rad = glm::radians(yaw_deg_);
        float pitch_rad = glm::radians(pitch_deg_);

        // Compute front vector (looking direction)
        front_.x = std::cos(yaw_rad) * std::cos(pitch_rad);
        front_.y = std::sin(pitch_rad);
        front_.z = std::sin(yaw_rad) * std::cos(pitch_rad);
        front_ = glm::normalize(front_);

        // Compute right vector (perpendicular to front, in the xz plane)
        right_ = glm::normalize(glm::cross(front_, glm::vec3(0.0f, 1.0f, 0.0f)));

        // Recompute up (perpendicular to front and right)
        up_ = glm::normalize(glm::cross(right_, front_));
    }
};

} // namespace gs
