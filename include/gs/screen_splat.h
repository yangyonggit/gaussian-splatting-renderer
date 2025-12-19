#pragma once
#include <glm/glm.hpp>
#include "gaussian.h"

namespace gs {

struct ScreenSplat {
    const GaussianSplat* src;   // Pointer to original splat data
    int gaussian_id;            // Stable index of the source Gaussian
    
    float sx;                   // Screen x coordinate in pixels (float)
    float sy;                   // Screen y coordinate in pixels (float)
    float depth;                // Depth value [0,1], smaller = closer (from NDC)
    
    // 2D ellipse parameters for anisotropic Gaussian rasterization
    glm::mat2 cov;              // 2x2 covariance matrix Σ in screen space
    glm::mat2 cov_inv;          // Inverse of covariance Σ^{-1}
    float det_cov;              // Determinant of Σ (for normalization and debug)
    float radius_px;            // Conservative radius in pixels for bounding box (~3σ coverage)
};

} // namespace gs
