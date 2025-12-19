#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <cstdio>

namespace gs {
namespace gl {

/**
 * Compile a shader from source.
 * Returns the compiled shader handle or 0 on failure.
 */
inline GLuint compileShader(GLenum shader_type, const char* source) {
    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Check for compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::fprintf(stderr, "Shader compilation failed:\n%s\n", infoLog);
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

/**
 * Link a program from vertex and fragment shaders.
 * Returns the program handle or 0 on failure.
 */
inline GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    // Check for link errors
    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::fprintf(stderr, "Program linking failed:\n%s\n", infoLog);
        glDeleteProgram(program);
        return 0;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

/**
 * Full-screen quad renderer.
 * Handles VAO/VBO setup and drawing.
 */
class FullscreenQuad {
public:
    FullscreenQuad() : vao_(0), vbo_(0), program_(0) {}

    ~FullscreenQuad() {
        cleanup();
    }

    /**
     * Initialize the quad with a shader program.
     * Shader program should have a uniform sampler2D 'tex' and no other inputs.
     */
    bool init(GLuint program) {
        program_ = program;

        // Vertex positions for a full-screen quad (NDC: -1 to 1)
        float vertices[] = {
            // Position     TexCoord
            -1.0f,  1.0f,   0.0f, 0.0f,  // Top-left
             1.0f,  1.0f,   1.0f, 0.0f,  // Top-right
            -1.0f, -1.0f,   0.0f, 1.0f,  // Bottom-left
             1.0f, -1.0f,   1.0f, 1.0f,  // Bottom-right
        };

        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        // Position attribute (location 0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        // TexCoord attribute (location 1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        return true;
    }

    /**
     * Draw the full-screen quad.
     * Assumes the texture is already bound to the active texture unit.
     */
    void draw() {
        glUseProgram(program_);
        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }

    void cleanup() {
        if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
        if (vbo_) { glDeleteBuffers(1, &vbo_); vbo_ = 0; }
        // Don't delete program_ here; caller may reuse it
    }

private:
    GLuint vao_;
    GLuint vbo_;
    GLuint program_;
};

/**
 * Simple RGBA8 texture wrapper for display.
 */
class Texture {
public:
    Texture() : handle_(0), width_(0), height_(0) {}

    ~Texture() {
        cleanup();
    }

    /**
     * Create or resize an RGBA8 texture.
     */
    bool init(int width, int height, GLenum format = GL_RGBA8) {
        width_ = width;
        height_ = height;

        if (!handle_) {
            glGenTextures(1, &handle_);
        }

        glBindTexture(GL_TEXTURE_2D, handle_);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, 
                     (format == GL_RGBA8 ? GL_RGBA : GL_RGB), GL_UNSIGNED_BYTE, nullptr);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glBindTexture(GL_TEXTURE_2D, 0);
        return true;
    }

    /**
     * Update texture data from CPU memory.
     * This is the "slow path" for Stage A (CPU -> GPU copy via glTexSubImage2D).
     * 
     * @param data Pixel data (RGBA8 or RGB8 depending on texture format)
     */
    void update(const void* data, GLenum format = GL_RGBA) {
        glBindTexture(GL_TEXTURE_2D, handle_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, 
                        format, GL_UNSIGNED_BYTE, data);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    /**
     * Bind texture to a slot for rendering.
     */
    void bind(int slot = 0) {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_2D, handle_);
    }

    GLuint getHandle() const { return handle_; }
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }

    void cleanup() {
        if (handle_) { glDeleteTextures(1, &handle_); handle_ = 0; }
    }

private:
    GLuint handle_;
    int width_;
    int height_;
};

} // namespace gl
} // namespace gs
