#include "gs/sh_color.h"
#include "gs/gaussian.h"

#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>

namespace gs {

// Global flag to enable/disable SH evaluation
static bool g_sh_enabled = true;

void setSHEvaluationEnabled(bool enabled) {
    g_sh_enabled = enabled;
}

bool isSHEvaluationEnabled() {
    return g_sh_enabled;
}

// Evaluate 9 real SH basis functions (up to l = 2)
// Ordering matches the common layout used in NeRF / 3DGS:
//   0:  0.282095
//   1: -0.488603 * y
//   2:  0.488603 * z
//   3: -0.488603 * x
//   4:  1.092548 * x * y
//   5: -1.092548 * y * z
//   6:  0.315392 * (3 z^2 - 1)
//   7: -1.092548 * x * z
//   8:  0.546274 * (x^2 - y^2)
static inline void evalSH9(const glm::vec3& dir, float* sh)
{
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    sh[0] = 0.28209479177387814f;
    sh[1] = -0.4886025119029199f * y;
    sh[2] =  0.4886025119029199f * z;
    sh[3] = -0.4886025119029199f * x;
    sh[4] =  1.0925484305920792f * x * y;
    sh[5] = -1.0925484305920792f * y * z;
    sh[6] =  0.31539156525252005f * (3.0f * z * z - 1.0f);
    sh[7] = -1.0925484305920792f * x * z;
    sh[8] =  0.5462742152960396f * (x * x - y * y);
}

glm::vec3 evalSHColor(const GaussianSplat& g, const glm::vec3& view_dir)
{
    // Global switch or missing data: fall back to DC-only color
    if (!g_sh_enabled || g.sh_color.coeffs.empty()) {
        return g.dc_color;
    }

    int n_basis = static_cast<int>(g.sh_color.coeffs.size() / 3);
    if (n_basis <= 1) {
        // Only DC available
        return g.dc_color;
    }

    // We currently only support up to 9 basis (l <= 2)
    constexpr int kMaxSupportedBasis = 9;
    int used_basis = std::min(n_basis, kMaxSupportedBasis);

    glm::vec3 dir = glm::normalize(view_dir);
    float sh[kMaxSupportedBasis];
    evalSH9(dir, sh);

    int baseR = 0;
    int baseG = n_basis;
    int baseB = n_basis * 2;

    // Start from DC color that we precomputed in loader
    glm::vec3 color = g.dc_color;

    // Add higher-order SH contributions (skip basis 0, which is DC)
    for (int b = 1; b < used_basis; ++b) {
        float w = sh[b];
        color.r += g.sh_color.coeffs[baseR + b] * w;
        color.g += g.sh_color.coeffs[baseG + b] * w;
        color.b += g.sh_color.coeffs[baseB + b] * w;
    }

    // Clamp to valid range [0, 1]
    color = glm::clamp(color, 0.0f, 1.0f);
    return color;
}

} // namespace gs
