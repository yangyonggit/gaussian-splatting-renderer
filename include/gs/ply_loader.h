#pragma once
#include <string>
#include <vector>
#include "gaussian.h"

namespace gs {

/**
 * Load Gaussian Splatting data from a PLY file.
 * Supports binary_little_endian format exported by 3DGS training.
 * 
 * @param filename Path to the .ply file
 * @return Vector of GaussianSplat structures
 */
std::vector<GaussianSplat> loadGaussianPly(const std::string& filename);

} // namespace gs
