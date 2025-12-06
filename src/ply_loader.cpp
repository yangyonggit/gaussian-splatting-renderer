#include "gs/ply_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <glm/glm.hpp>
#include <cmath>

namespace gs {

std::vector<GaussianSplat> loadGaussianPly(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open PLY file: " << filename << std::endl;
        return {};
    }

    // ============================================================
    // 1. Parse ASCII header
    // ============================================================
    std::string line;
    std::unordered_map<std::string, int> propIndex;
    int vertexCount = 0;
    int propertyCount = 0;
    bool isBinaryLittleEndian = false;

    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        if (line.empty()) continue;
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "format") {
            std::string format;
            iss >> format;
            if (format == "binary_little_endian") {
                isBinaryLittleEndian = true;
            } else {
                std::cerr << "Unsupported PLY format: " << format << std::endl;
                return {};
            }
        }
        else if (keyword == "element") {
            std::string elementType;
            int count;
            iss >> elementType >> count;
            if (elementType == "vertex") {
                vertexCount = count;
            }
        }
        else if (keyword == "property") {
            std::string dataType, propName;
            iss >> dataType >> propName;

            // Store property index (assume all properties are float)
            propIndex[propName] = propertyCount;
            propertyCount++;
        }
        else if (keyword == "end_header") {
            break;
        }
    }

    if (!isBinaryLittleEndian) {
        std::cerr << "Only binary_little_endian format is supported." << std::endl;
        return {};
    }

    if (vertexCount == 0 || propertyCount == 0) {
        std::cerr << "Invalid PLY header: no vertices or properties." << std::endl;
        return {};
    }

    std::cout << "PLY Header parsed: " << vertexCount << " vertices, "
              << propertyCount << " properties per vertex." << std::endl;

    // ============================================================
    // 2. Read binary vertex data
    // ============================================================
    std::vector<GaussianSplat> splats;
    splats.reserve(vertexCount);

    std::vector<float> row(propertyCount);

    for (int i = 0; i < vertexCount; ++i) {
        // Read one vertex (all properties as floats)
        file.read(reinterpret_cast<char*>(row.data()), propertyCount * sizeof(float));

        if (!file) {
            std::cerr << "Failed to read vertex " << i << std::endl;
            break;
        }

        // ============================================================
        // 3. Map PLY fields to GaussianSplat (full parameters)
        // ============================================================
        GaussianSplat splat;

        // Position (required)
        if (propIndex.count("x") && propIndex.count("y") && propIndex.count("z")) {
            splat.position_ws = glm::vec3(
                row[propIndex["x"]],
                row[propIndex["y"]],
                row[propIndex["z"]]
            );
        } else {
            std::cerr << "Missing position properties (x, y, z)" << std::endl;
            continue;
        }

        // Scale (stored as log in PLY, need to exponentiate)
        if (propIndex.count("scale_0") && propIndex.count("scale_1") && propIndex.count("scale_2")) {
            splat.scale = glm::vec3(
                std::exp(row[propIndex["scale_0"]]),
                std::exp(row[propIndex["scale_1"]]),
                std::exp(row[propIndex["scale_2"]])
            );
        } else {
            splat.scale = glm::vec3(1.0f);
        }

        // Rotation (quaternion, stored as rot_0..3 in PLY)
        // 3DGS uses [x, y, z, w] order in the file.
        if (propIndex.count("rot_0") && propIndex.count("rot_1") &&
            propIndex.count("rot_2") && propIndex.count("rot_3")) {

            float qx = row[propIndex["rot_0"]];
            float qy = row[propIndex["rot_1"]];
            float qz = row[propIndex["rot_2"]];
            float qw = row[propIndex["rot_3"]];

            // glm::quat(w, x, y, z)
            splat.rotation = glm::normalize(glm::quat(qw, qx, qy, qz));
        } else {
            splat.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }

        // Opacity (stored as logit in PLY, convert to [0, 1])
        if (propIndex.count("opacity")) {
            float logit = row[propIndex["opacity"]];
            splat.opacity = 1.0f / (1.0f + std::exp(-logit));
        } else {
            splat.opacity = 0.8f;
        }

        // DC Color from SH DC (f_dc_0..2)
        // RGB = 0.5 + SH_C0 * f_dc
        const float SH_C0 = 0.28209479177387814f;
        float fdc0 = 0.0f, fdc1 = 0.0f, fdc2 = 0.0f;
        if (propIndex.count("f_dc_0") && propIndex.count("f_dc_1") && propIndex.count("f_dc_2")) {
            fdc0 = row[propIndex["f_dc_0"]];
            fdc1 = row[propIndex["f_dc_1"]];
            fdc2 = row[propIndex["f_dc_2"]];

            glm::vec3 color(fdc0, fdc1, fdc2);
            color = glm::vec3(0.5f) + SH_C0 * color;
            splat.dc_color = glm::clamp(color, 0.0f, 1.0f);
        } else {
            splat.dc_color = glm::vec3(0.5f);
        }

        // ------------------------------------------------------------
        // Spherical Harmonics coefficients
        //
        // f_rest_* are stored interleaved as:
        //   [c1_R, c1_G, c1_B, c2_R, c2_G, c2_B, ...]
        //
        // We want to store them as:
        //   coeffs = [R0..R(N-1), G0..G(N-1), B0..B(N-1)]
        // where 0 is DC, and N = number of SH bases.
        // ------------------------------------------------------------

        // Collect all f_rest coefficients for this vertex
        std::vector<float> f_rest_coeffs;
        f_rest_coeffs.reserve(48); // typical value, will grow if needed
        for (int k = 0;; ++k) {
            std::string name = "f_rest_" + std::to_string(k);
            auto it = propIndex.find(name);
            if (it == propIndex.end()) break;
            f_rest_coeffs.push_back(row[it->second]);
        }

        int n_rest = static_cast<int>(f_rest_coeffs.size());
        if (n_rest == 0) {
            // No higher-order SH, degree 0 (DC only)
            splat.sh_color.degree = 0;
            splat.sh_color.coeffs.assign(3, 0.0f); // R0, G0, B0

            splat.sh_color.coeffs[0] = fdc0;
            splat.sh_color.coeffs[1] = fdc1;
            splat.sh_color.coeffs[2] = fdc2;
        } else {
            if (n_rest % 3 != 0) {
                std::cerr << "Warning: f_rest count (" << n_rest
                          << ") is not divisible by 3. Data layout may be unexpected.\n";
            }

            // Number of SH bases per channel: 1 (DC) + rest/3
            int n_basis = 1 + n_rest / 3;

            // Infer degree from n_basis ≈ (l_max + 1)^2
            int l_max = static_cast<int>(std::sqrt(static_cast<float>(n_basis)) + 0.5f) - 1;
            if (l_max < 0) l_max = 0;
            splat.sh_color.degree = l_max;

            int coeffs_per_channel = n_basis;
            int total_coeffs = coeffs_per_channel * 3;

            splat.sh_color.coeffs.assign(total_coeffs, 0.0f);

            int baseR = 0;
            int baseG = coeffs_per_channel;
            int baseB = coeffs_per_channel * 2;

            // DC term (basis 0)
            splat.sh_color.coeffs[baseR + 0] = fdc0;
            splat.sh_color.coeffs[baseG + 0] = fdc1;
            splat.sh_color.coeffs[baseB + 0] = fdc2;

            // Higher-order terms (basis 1..n_basis-1)
            int available_basis = n_rest / 3;
            int used_basis = std::min(available_basis, n_basis - 1);

            for (int b = 0; b < used_basis; ++b) {
                int coeff_index = b * 3;
                float cr = f_rest_coeffs[coeff_index + 0];
                float cg = f_rest_coeffs[coeff_index + 1];
                float cb = f_rest_coeffs[coeff_index + 2];

                int basis_idx = b + 1; // basis 0 is DC
                splat.sh_color.coeffs[baseR + basis_idx] = cr;
                splat.sh_color.coeffs[baseG + basis_idx] = cg;
                splat.sh_color.coeffs[baseB + basis_idx] = cb;
            }
        }

        splats.push_back(splat);
    }

    file.close();

    std::cout << "Loaded " << splats.size() << " gaussians from PLY." << std::endl;

    // Debug: print position range
    if (!splats.empty()) {
        glm::vec3 minPos = splats[0].position_ws;
        glm::vec3 maxPos = splats[0].position_ws;
        for (const auto& s : splats) {
            minPos = glm::min(minPos, s.position_ws);
            maxPos = glm::max(maxPos, s.position_ws);
        }
        std::cout << "Position range: [" << minPos.x << ", " << maxPos.x << "] x ";
        std::cout << "[" << minPos.y << ", " << maxPos.y << "] x ";
        std::cout << "[" << minPos.z << ", " << maxPos.z << "]\n";
    }

    return splats;
}

} // namespace gs
