#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

namespace gs {

/**
 * Spherical Harmonics color representation
 * Stores coefficients for R, G, B channels separately
 * Layout: [R_c0, R_c1, ..., R_c8, G_c0, ..., G_c8, B_c0, ..., B_c8] for degree=2
 */
struct SHColor {
    std::vector<float> coeffs;  // SH coefficients in RGB interleaved or channel layout
    int degree;                 // SH degree: 0, 1, 2, or 3
};

/**
 * V2 Gaussian Splat with full 3D parameters and SH colors
 * 
 * Contains complete information from 3DGS training:
 * - 3D center position in world space
 * - 3D scale factors (stored as log in .ply)
 * - 3D rotation as quaternion
 * - Spherical Harmonics coefficients for view-dependent color
 * - Opacity (stored as logit in .ply)
 *
 * This enables:
 * - Anisotropic 2D Gaussian rendering (ellipse)
 * - View-dependent color evaluation
 */
struct GaussianSplat {
    // 3D world-space parameters
    glm::vec3 position_ws;      // center position (from x, y, z in PLY)
    glm::vec3 scale;            // scale factors (from scale_0, scale_1, scale_2; stored as log)
    glm::quat rotation;         // rotation quaternion (from rot_0..3 in PLY)
    
    // Opacity
    float opacity;              // converted from logit in PLY (0~1)
    
    // Color representation
    glm::vec3 dc_color;         // DC (0-order SH) color, from f_dc_0..2, mapped to [0,1]
    SHColor sh_color;           // Higher-order SH coefficients from f_rest_*
};

} // namespace gs

