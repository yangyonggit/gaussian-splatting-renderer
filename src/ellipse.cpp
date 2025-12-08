#include "gs/ellipse.h"
#include "gs/screen_splat.h"
#include "gs/gaussian.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <cmath>

namespace gs {

// Convert NDC (-1..1) to pixel coordinates (origin at top-left)
static inline glm::vec2 ndcToPixel(const glm::vec3& ndc, int width, int height)
{
    float sx = (1.0f - (ndc.x * 0.5f + 0.5f)) * static_cast<float>(width);
    float sy = (ndc.y * 0.5f + 0.5f) * static_cast<float>(height);
    return glm::vec2(sx, sy);
}

bool projectToScreenEllipse(
    const GaussianSplat& g,
    const glm::mat4&      view,
    const glm::mat4&      proj,
    int                   width,
    int                   height,
    ScreenSplat&          out)
{
    // ---------------------------------------------------------------------
    // 1. Project center: world -> clip -> ndc -> pixel
    // ---------------------------------------------------------------------
    glm::vec4 center_ws(g.position_ws, 1.0f);
    glm::vec4 center_cs = proj * view * center_ws;

    if (center_cs.w <= 0.0f) {
        // Behind the camera
        return false;
    }

    glm::vec3 center_ndc = glm::vec3(center_cs) / center_cs.w;
    glm::vec2 center_px  = ndcToPixel(center_ndc, width, height);

    // Early reject: center is far outside the view frustum in X/Y
    // This prevents splats with huge NDC from generating enormous footprints.
    // const float ndc_pad = 2.0f; // allow a bit outside [-1,1]
    // if (std::abs(center_ndc.x) > ndc_pad || std::abs(center_ndc.y) > ndc_pad) {
    //     return false;
    // }

    // Optional early reject on Z as well (you已经有 z 检查可以保留 / 合并)
    if (center_ndc.z < -1.5f || center_ndc.z > 1.5f) {
        return false;
    }

    // ---------------------------------------------------------------------
    // 2. Build three 3D principal axes in world space
    //    g.scale already stores exp(scale_log), i.e. standard deviation.
    // ---------------------------------------------------------------------
    glm::mat3 R_ws = glm::mat3_cast(g.rotation);

    glm::vec3 axes_ws[3];
    axes_ws[0] = R_ws[0] * g.scale.x; // principal axis X
    axes_ws[1] = R_ws[1] * g.scale.y; // principal axis Y
    axes_ws[2] = R_ws[2] * g.scale.z; // principal axis Z

    // ---------------------------------------------------------------------
    // 3. Project each axis endpoint and compute 2D screen-space axes
    // ---------------------------------------------------------------------
    glm::vec2 axes_2d[3];
    float     max_len2 = 0.0f;

    glm::mat4 vp = proj * view;

    for (int i = 0; i < 3; ++i) {
        glm::vec4 end_ws(g.position_ws + axes_ws[i], 1.0f);
        glm::vec4 end_cs = vp * end_ws;

        if (end_cs.w <= 0.0f) {
            // Push slightly in front of the camera to avoid division by zero
            end_cs.w = 1e-3f;
        }

        glm::vec3 end_ndc = glm::vec3(end_cs) / end_cs.w;
        glm::vec2 end_px  = ndcToPixel(end_ndc, width, height);

        glm::vec2 axis_2d = end_px - center_px;
        axes_2d[i]        = axis_2d;

        float len2 = glm::dot(axis_2d, axis_2d);
        if (len2 > max_len2) {
            max_len2 = len2;
        }
    }

    if (max_len2 < 1e-8f || !std::isfinite(max_len2)) {
        // Degenerate footprint
        return false;
    }

    // ---------------------------------------------------------------------
    // 4. Build 2D covariance matrix in screen space
    //
    //    We approximate Σ_2D as sum of outer products of projected axes:
    //      Σ = Σ_i (a_i a_i^T)
    //    where a_i are the 2D axis vectors in pixels.
    // ---------------------------------------------------------------------
    glm::mat2 cov(0.0f);

    for (int i = 0; i < 3; ++i) {
        const glm::vec2& a = axes_2d[i];
        cov[0][0] += a.x * a.x;
        cov[0][1] += a.x * a.y;
        cov[1][0] += a.y * a.x;
        cov[1][1] += a.y * a.y;
    }

    // Small regularization for numerical stability
    float trace = cov[0][0] + cov[1][1];
    float eps   = 1e-3f * trace + 1e-6f;
    cov[0][0]  += eps;
    cov[1][1]  += eps;

    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    if (!(det > 0.0f) || !std::isfinite(det)) {
        return false;
    }

    glm::mat2 cov_inv = glm::inverse(cov);

    // ---------------------------------------------------------------------
    // 5. Estimate a conservative pixel radius for bounding box
    //    Use 3 * max axis length (≈ 3σ) and clamp to a reasonable range.
    // ---------------------------------------------------------------------
    float max_len   = std::sqrt(max_len2);
    float radius_px = max_len * 3.0f; // ~3 sigma

    // Clamp radius to a reasonable range
    const float min_radius = 1.0f;
    const float max_radius = 1024.0f;   // tune this if needed
    radius_px = std::clamp(radius_px, min_radius, max_radius);

    // ---------------------------------------------------------------------
    // 6. Fill ScreenSplat
    // ---------------------------------------------------------------------
    out.src      = &g;
    out.sx       = center_px.x;
    out.sy       = center_px.y;
    // Map NDC z (-1..1) to [0,1] for sorting
    out.depth    = center_ndc.z * 0.5f + 0.5f;
    out.cov      = cov;
    out.cov_inv  = cov_inv;
    out.det_cov  = det;
    out.radius_px = radius_px;

    return true;
}

} // namespace gs
