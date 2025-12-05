#pragma once
#include <glm/glm.hpp>

namespace gs {

/**
 * Minimal Gaussian Splat structure for V1-CPU pipeline.
 * 
 * This version is intentionally simple:
 * - Uses world-space position
 * - Uses RGB color (0~1)
 * - Uses a single scalar radius (instead of full covariance)
 * - Uses opacity (0~1)
 *
 * Later in V2, this will be extended with:
 * - anisotropic covariance (2D ellipse)
 * - SH lighting coefficients
 * - scale & rotation from real 3DGS .ply files
 */
struct GaussianSplat {
    glm::vec3 position;   // world space position
    glm::vec3 color;      // RGB (0~1)
    float radius;         // simplified radius for V1
    float opacity;        // alpha blending factor (0~1)
};

} // namespace gs
