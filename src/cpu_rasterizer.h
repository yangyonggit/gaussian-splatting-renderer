#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "gs/gaussian.h"
#include "gs/screen_splat.h"

namespace CpuRasterizer {

/**
 * CPU-based Gaussian Splatting Rasterizer.
 * Handles projection, rasterization, and output of Gaussian splats on CPU.
 */
class Rasterizer {
public:
    Rasterizer();
    ~Rasterizer();

    /**
     * Render Gaussian splats from a PLY file.
     * @param ply_path Path to the input PLY file containing Gaussian splats.
     * @param output_path Path to save the output PNG image.
     * @param view View matrix for the camera.
     * @param proj Projection matrix for the camera.
     * @param width Output framebuffer width.
     * @param height Output framebuffer height.
     * @param camera_pos Camera position in world space.
     * @return true if rendering succeeded, false otherwise.
     */
    bool render(
        const std::string& ply_path,
        const std::string& output_path,
        const glm::mat4& view,
        const glm::mat4& proj,
        int width,
        int height,
        const glm::vec3& camera_pos
    );

    /**
     * Prepare screen-space splats for CUDA rendering.
     * Projects gaussians, computes 2D conics, and sorts by depth.
     * This is the CPU preprocessing step for CUDA V1 rendering.
     * 
     * @param ply_path Path to input PLY file
     * @param view View matrix
     * @param proj Projection matrix
     * @param width Image width
     * @param height Image height
     * @param out_screen_splats Output: projected screen-space splats
     * @param out_sorted_indices Output: back-to-front sorted indices
     * @return true on success
     */
    bool prepareForCuda(
        const std::string& ply_path,
        const glm::mat4& view,
        const glm::mat4& proj,
        int width,
        int height,
        std::vector<gs::GaussianSplat>& out_splats,
        std::vector<gs::ScreenSplat>& out_screen_splats,
        std::vector<int>& out_sorted_indices
    );

private:
    // Internal helper structures
    struct FloatPixel {
        float r, g, b, a;
    };

    // Internal rendering pipeline methods
    bool projectSplats(
        const std::vector<gs::GaussianSplat>& splats,
        const glm::mat4& view,
        const glm::mat4& proj,
        int width,
        int height,
        std::vector<gs::ScreenSplat>& out_projected
    );

    void sortByDepth(std::vector<gs::ScreenSplat>& splats);

    bool rasterize(
        const std::vector<gs::ScreenSplat>& projected,
        int width,
        int height,
        const glm::vec3& camera_pos,
        std::vector<FloatPixel>& out_framebuffer
    );

    bool saveFramebuffer(
        const std::vector<FloatPixel>& framebuffer,
        int width,
        int height,
        const std::string& output_path
    );
};

} // namespace CpuRasterizer
