#pragma once
#include <glm/glm.hpp>
#include "gaussian.h"

namespace gs {

struct ScreenSplat {
    const GaussianSplat* src; // Pointer to original splat data
    float sx;                 // Screen x coordinate (float)
    float sy;                 // Screen y coordinate (float)
    float depth;              // Depth value [0,1], smaller = closer (from NDC)
};

} // namespace gs
