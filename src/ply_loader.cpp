#include "gs/ply_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <glm/glm.hpp>

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
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty()) continue;

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
        // 3. Map PLY fields to GaussianSplat
        // ============================================================
        GaussianSplat splat;

        // Position (required)
        if (propIndex.count("x") && propIndex.count("y") && propIndex.count("z")) {
            splat.position = glm::vec3(
                row[propIndex["x"]],
                row[propIndex["y"]],
                row[propIndex["z"]]
            );
        } else {
            std::cerr << "Missing position properties (x, y, z)" << std::endl;
            continue;
        }

        // Color from SH DC components (f_dc_0, f_dc_1, f_dc_2)
        if (propIndex.count("f_dc_0") && propIndex.count("f_dc_1") && propIndex.count("f_dc_2")) {
            // SH DC coefficient is the 0th band, represents ambient color
            // The DC component needs to be converted from SH space
            const float SH_C0 = 0.28209479177387814f; // 0.5 * sqrt(1/pi)
            
            glm::vec3 color(
                row[propIndex["f_dc_0"]],
                row[propIndex["f_dc_1"]],
                row[propIndex["f_dc_2"]]
            );
            
            // Convert from SH to RGB: RGB = 0.5 + SH_C0 * dc
            color = glm::vec3(0.5f) + SH_C0 * color;
            splat.color = glm::clamp(color, 0.0f, 1.0f);
        } else {
            // Fallback: use white color
            splat.color = glm::vec3(0.5f);
        }

        // Radius (approximate from scale)
        if (propIndex.count("scale_0") && propIndex.count("scale_1") && propIndex.count("scale_2")) {
            float scaleX = std::exp(row[propIndex["scale_0"]]);  // Scales are stored as log
            float scaleY = std::exp(row[propIndex["scale_1"]]);
            float scaleZ = std::exp(row[propIndex["scale_2"]]);
            
            // Average scale as a simple approximation
            float avgScale = (scaleX + scaleY + scaleZ) / 3.0f;
            
            // Map to screen-space pixels (empirical scaling factor)
            splat.radius = avgScale * 50.0f;
        } else {
            // Fallback: default radius
            splat.radius = 0.05f;
        }

        // Opacity
        if (propIndex.count("opacity")) {
            // Opacity is stored as logit, convert to [0,1]
            float logit = row[propIndex["opacity"]];
            splat.opacity = 1.0f / (1.0f + std::exp(-logit));
        } else {
            // Fallback: default opacity
            splat.opacity = 0.8f;
        }

        splats.push_back(splat);
    }

    file.close();

    std::cout << "Loaded " << splats.size() << " gaussians from PLY." << std::endl;

    return splats;
}

} // namespace gs
