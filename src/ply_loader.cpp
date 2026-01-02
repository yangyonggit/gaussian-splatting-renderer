#include "gs/ply_loader.h"
#include "gs/timer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <glm/glm.hpp>
#include <cmath>

namespace gs {

std::vector<GaussianSplat> loadGaussianPly(const std::string& filename) {
    Timer totalTimer("PLY load total");
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open PLY file: " << filename << std::endl;
        return {};
    }

    // 1. Parse Header
    std::string line;
    std::unordered_map<std::string, int> propIndex;
    int vertexCount = 0;
    int propertyCount = 0;
    bool isBinaryLittleEndian = false;

    {
        Timer headerTimer("PLY header parse");
        while (std::getline(file, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            line.erase(0, first);
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;

            if (keyword == "format") {
                std::string format;
                iss >> format;
                if (format == "binary_little_endian") isBinaryLittleEndian = true;
            } else if (keyword == "element") {
                std::string type;
                int count;
                iss >> type >> count;
                if (type == "vertex") vertexCount = count;
            } else if (keyword == "property") {
                std::string type, name;
                iss >> type >> name;
                propIndex[name] = propertyCount++;
            } else if (keyword == "end_header") {
                break;
            }
        }
    }

    if (!isBinaryLittleEndian) {
        std::cerr << "Error: Only binary_little_endian supported.\n";
        return {};
    }

    // ============================================================
    // PRE-CACHE INDICES (Performance Optimization)
    // Avoid doing string map lookups inside the loop!
    // ============================================================
    auto getIdx = [&](const std::string& name) -> int {
        return propIndex.count(name) ? propIndex[name] : -1;
    };

    int idx_x = getIdx("x");
    int idx_y = getIdx("y");
    int idx_z = getIdx("z");
    
    int idx_s0 = getIdx("scale_0");
    int idx_s1 = getIdx("scale_1");
    int idx_s2 = getIdx("scale_2");

    int idx_r0 = getIdx("rot_0");
    int idx_r1 = getIdx("rot_1");
    int idx_r2 = getIdx("rot_2");
    int idx_r3 = getIdx("rot_3");

    int idx_op = getIdx("opacity");

    int idx_dc0 = getIdx("f_dc_0");
    int idx_dc1 = getIdx("f_dc_1");
    int idx_dc2 = getIdx("f_dc_2");

    // Collect SH Rest indices
    std::vector<int> idx_rest;
    for (int k = 0; ; ++k) {
        int idx = getIdx("f_rest_" + std::to_string(k));
        if (idx == -1) break;
        idx_rest.push_back(idx);
    }

    std::cout << "Header parsed. Vertices: " << vertexCount << ". SH Coeffs (Rest): " << idx_rest.size() << std::endl;

    // ============================================================
    // 2. Read Data
    // ============================================================
    std::vector<GaussianSplat> splats;
    splats.reserve(vertexCount);

    const size_t floatsPerVertex = static_cast<size_t>(propertyCount);
    const size_t totalFloats = static_cast<size_t>(vertexCount) * floatsPerVertex;
    std::vector<float> allRows(totalFloats);
    
    // Constant for SH conversion
    const float SH_C0 = 0.28209479177387814f;

    {
        Timer readTimer("PLY vertex data read");
        file.read(reinterpret_cast<char*>(allRows.data()), static_cast<std::streamsize>(totalFloats * sizeof(float)));
    }

    if (!file) {
        std::cerr << "Failed while reading PLY vertex data: " << filename << std::endl;
        return {};
    }

    {
        Timer parseTimer("PLY vertex data parse");
        for (int i = 0; i < vertexCount; ++i) {
            const float* row = allRows.data() + static_cast<size_t>(i) * floatsPerVertex;

        GaussianSplat splat;

        // 1. Position
        if (idx_x >= 0) {
            splat.position_ws = glm::vec3(row[idx_x], row[idx_y], row[idx_z]);
        }

        // 2. Scale (Exp)
        if (idx_s0 >= 0) {
            splat.scale = glm::vec3(std::exp(row[idx_s0]), std::exp(row[idx_s1]), std::exp(row[idx_s2]));
        } else {
            splat.scale = glm::vec3(0.01f);
        }

        // 3. Rotation (FIXED ORDER: W, X, Y, Z)
        if (idx_r0 >= 0) {
            // Standard 3DGS PLY: rot_0 is Real(W), rot_1..3 is Imaginary(X,Y,Z)
            float qw = row[idx_r0];
            float qx = row[idx_r1];
            float qy = row[idx_r2];
            float qz = row[idx_r3];
            splat.rotation = glm::normalize(glm::quat(qw, qx, qy, qz));
        } else {
            splat.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }

        // 4. Opacity (Sigmoid)
        if (idx_op >= 0) {
            splat.opacity = 1.0f / (1.0f + std::exp(-row[idx_op]));
        }

        // 5. Color (DC)
        if (idx_dc0 >= 0) {
            glm::vec3 color(row[idx_dc0], row[idx_dc1], row[idx_dc2]);
            // Convert SH_0 to RGB
            splat.dc_color = glm::vec3(0.5f) + SH_C0 * color;
            // Optional: Clamp mainly for visual safety, though SH can go out of bounds
            // splat.dc_color = glm::clamp(splat.dc_color, 0.0f, 1.0f); 
        }

        // 6. SH (Rest)
        int n_rest = (int)idx_rest.size();
        if (n_rest > 0) {
            int n_basis = 1 + n_rest / 3; // +1 for DC
            splat.sh_color.degree = 3; // Assume max degree
            // Resize buffer (DC + Rest) -> Total 48 floats for Degree 3
            splat.sh_color.coeffs.resize(48, 0.0f);
            
            // Copy DC first
            splat.sh_color.coeffs[0] = row[idx_dc0]; // R_dc
            splat.sh_color.coeffs[16] = row[idx_dc1]; // G_dc (Planar offset 16)
            splat.sh_color.coeffs[32] = row[idx_dc2]; // B_dc (Planar offset 32)

            // Copy Rest (Interleaved in PLY -> Planar in Struct)
            // PLY:  R1 G1 B1, R2 G2 B2...
            // Dest: R1 R2... | G1 G2... | B1 B2...
            for (int b = 0; b < n_rest / 3; ++b) {
                float r = row[idx_rest[b * 3 + 0]];
                float g = row[idx_rest[b * 3 + 1]];
                float b_val = row[idx_rest[b * 3 + 2]];

                // Target index in planar array (skip index 0 which is DC)
                splat.sh_color.coeffs[0 + b + 1]  = r;
                splat.sh_color.coeffs[16 + b + 1] = g;
                splat.sh_color.coeffs[32 + b + 1] = b_val;
            }
        }
        
            splats.push_back(splat);
        }
    }

    return splats;
}

} // namespace gs