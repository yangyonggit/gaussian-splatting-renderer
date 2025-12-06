#pragma once
#include <glm/glm.hpp>
#include "gaussian.h"

namespace gs {

// Forward declaration
struct ScreenSplat;

/**
 * Project a 3D Gaussian (with scale and rotation) to screen space
 * and compute the 2D covariance matrix for ellipse-based rasterization.
 *
 * Uses a geometric approximation:
 * 1. Transform 3D ellipsoid to view space
 * 2. Project principal axes to screen
 * 3. Construct 2×2 covariance from projected axes
 *
 * @param g Input Gaussian splat with position, scale, and rotation
 * @param view View matrix (world to view space)
 * @param proj Projection matrix (view to clip space)
 * @param width Screen width in pixels
 * @param height Screen height in pixels
 * @param out Output ScreenSplat with 2D ellipse parameters
 * @return true if projection successful and splat is visible, false otherwise
 */
bool projectToScreenEllipse(
    const GaussianSplat& g,
    const glm::mat4& view,
    const glm::mat4& proj,
    int width,
    int height,
    ScreenSplat& out
);

} // namespace gs
