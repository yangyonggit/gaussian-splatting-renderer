#pragma once
#include <glm/glm.hpp>
#include "gaussian.h"

namespace gs {

/**
 * Evaluate Spherical Harmonics color for a given view direction
 * 
 * Uses real SH basis functions up to degree 2 (9 basis functions).
 * Each RGB channel has independent SH coefficients.
 *
 * @param g Gaussian splat with SH color coefficients
 * @param view_dir Normalized view direction in world space
 *                 (from splat position toward camera)
 * @return RGB color evaluated at the given view direction (0~1)
 */
glm::vec3 evalSHColor(
    const GaussianSplat& g,
    const glm::vec3& view_dir
);

/**
 * Enable/disable SH color evaluation
 * When disabled, returns only DC component
 */
void setSHEvaluationEnabled(bool enabled);
bool isSHEvaluationEnabled();

} // namespace gs
