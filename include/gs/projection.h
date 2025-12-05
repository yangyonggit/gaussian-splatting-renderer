#pragma once
#include <glm/glm.hpp>

namespace gs {
namespace math {

/**
 * Project a 3D world space point into 2D screen space.
 *
 * @param p_world  world space coordinate
 * @param mvp      model-view-projection matrix
 * @param width    framebuffer width
 * @param height   framebuffer height
 * @param out_x    output: screen x
 * @param out_y    output: screen y
 * @param out_depth output: depth 0~1
 * @return true if point is inside the screen, false if clipped
 */
inline bool projectToScreen(
    const glm::vec3& p_world,
    const glm::mat4& mvp,
    int width, int height,
    float& out_x,
    float& out_y,
    float& out_depth
) {
    glm::vec4 clip = mvp * glm::vec4(p_world, 1.0f);

    if (clip.w == 0.0f) return false;

    glm::vec3 ndc = glm::vec3(clip) / clip.w;  // perspective divide

    // Reject points outside the clip space
    if (ndc.x < -1.f || ndc.x > 1.f ||
        ndc.y < -1.f || ndc.y > 1.f ||
        ndc.z < -1.f || ndc.z > 1.f) 
        return false;

    // Convert NDC → screen
    out_x = (ndc.x * 0.5f + 0.5f) * width;
    out_y = (1.0f - (ndc.y * 0.5f + 0.5f)) * height;
    out_depth = ndc.z * 0.5f + 0.5f;

    return true;
}

} // namespace math
} // namespace gs
