cmake_minimum_required(VERSION 3.14)

# Prevent CMake from using vcpkg toolchain if set
set(CMAKE_TOOLCHAIN_FILE "" CACHE FILEPATH "CMake toolchain file" FORCE)

# Define the project name and version
project(MiniGSRenderer VERSION 1.0 LANGUAGES CXX CUDA)

# ----------------------------------------------------------
# Build options (speed up iterative builds)
# ----------------------------------------------------------
# User request: disable the apps backed by src/main.cpp (MiniGSRenderer)
# and src/viewer_progressive.cpp by default to avoid wasting build time.
option(BUILD_MINIGSRENDERER "Build MiniGSRenderer (src/main.cpp)" OFF)
option(BUILD_VIEWER_MAIN "Build viewer_main (interactive viewer)" ON)
option(BUILD_VIEWER_PROGRESSIVE "Build viewer_progressive" OFF)

# Set C++ Standard to C++17 (Required for modern features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Set CUDA architectures (RTX 20/30/40 series)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Enable standard compiler warnings (Good practice)
if(MSVC)
    # Apply C++ warnings only to C++ files
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W3>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

# ----------------------------------------------------------
# 1. Dependency Management (FetchContent)
# ----------------------------------------------------------
include(FetchContent)

message(STATUS "📦 Configuring Dependencies...")

# --- Dependency: GLM (Math Library) ---
message(STATUS "⬇️  Fetching GLM...")
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0.9.9.8  # Pinning to a stable release version
)
# This downloads GLM and adds it to the project
FetchContent_MakeAvailable(glm)

# --- Dependency: STB (Image I/O) ---
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)

FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()

# --- Dependency: nlohmann_json (header-only JSON library) ---
message(STATUS "⬇️  Fetching nlohmann_json...")
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

FetchContent_MakeAvailable(nlohmann_json)

# --- Dependency: GLFW (Window & Input) ---
message(STATUS "⬇️  Fetching GLFW...")
# Disable GLFW extras to avoid building examples/tests/docs
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build GLFW examples" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "Build GLFW tests"    FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "Build GLFW docs"     FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "Install GLFW"        FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
FetchContent_MakeAvailable(glfw)

# --- Vendored GLAD (OpenGL Loader) ---
# GLAD files are pre-generated and stored in external/glad/
# No network access required - fully offline build
message(STATUS "🎨 Configuring vendored GLAD...")

add_library(glad STATIC
    ${CMAKE_SOURCE_DIR}/external/glad/src/glad.c
)

target_include_directories(glad PUBLIC
    ${CMAKE_SOURCE_DIR}/external/glad/include
)

# Link OpenGL on Windows
if(WIN32)
    target_link_libraries(glad PUBLIC opengl32)
endif()

message(STATUS "✅ GLAD configured (vendored, offline)")

# ----------------------------------------------------------
# 2. Source Code Configuration
# ----------------------------------------------------------
# Recursively find all .cpp and .cu files in the src/ directory
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

# Exclude viewer_main.cpp from MiniGSRenderer target
list(REMOVE_ITEM CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/viewer_main.cpp")
# Exclude progressive viewer TU from CLI target to avoid duplicate symbols
list(REMOVE_ITEM CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/viewer_progressive.cpp")
# Exclude legacy STB impl TU to avoid duplicate symbol definitions
list(REMOVE_ITEM CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/stb_image_write_impl.cpp")

# Create the executable target (optional)
if(BUILD_MINIGSRENDERER)
    add_executable(MiniGSRenderer ${CPP_SOURCES} ${CUDA_SOURCES})
endif()

# ----------------------------------------------------------
# 3. Linking & Includes
# ----------------------------------------------------------

# Find CUDA Toolkit (required for cudaMemcpy, cudaMalloc, etc.)
find_package(CUDAToolkit REQUIRED)

if(TARGET MiniGSRenderer)
    # Link GLM
    # GLM provides a modern CMake target "glm::glm" that handles include paths automatically.
    target_link_libraries(MiniGSRenderer PRIVATE glm::glm)

    # Link nlohmann_json (header-only target)
    target_link_libraries(MiniGSRenderer PRIVATE nlohmann_json::nlohmann_json)

    # Link CUDA runtime
    target_link_libraries(MiniGSRenderer PRIVATE CUDA::cudart)

    # Include STB directory
    # STB is a header-only library without a CMake target, so we manually add the include path.
    # ${stb_SOURCE_DIR} is automatically defined by FetchContent.
    target_include_directories(MiniGSRenderer PRIVATE ${stb_SOURCE_DIR})

    target_include_directories(MiniGSRenderer
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )
endif()

# ----------------------------------------------------------
# 4. Viewer Target (Stage A - Interactive Viewer)
# ----------------------------------------------------------
message(STATUS "🎮 Building viewer target...")

# Viewer executable with vendored GLAD (no network required)
if(BUILD_VIEWER_MAIN)
    add_executable(viewer_main 
        src/viewer_main.cpp 
        src/cuda_rasterizer.cu 
        src/cpu_rasterizer.cpp 
        src/ellipse.cpp 
        src/ply_loader.cpp 
        src/sh_color.cpp
    )

    target_link_libraries(viewer_main PRIVATE
        glm::glm
        glfw
        glad              # Vendored GLAD static library
        CUDA::cudart
        nlohmann_json::nlohmann_json
    )

    target_include_directories(viewer_main 
        PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
    )
endif()

# ----------------------------------------------------------
# 4b. Viewer Target (Progressive reveal viewer)
# ----------------------------------------------------------
message(STATUS "🎮 Building progressive viewer target...")

if(BUILD_VIEWER_PROGRESSIVE)
    add_executable(viewer_progressive
        src/viewer_progressive.cpp
        src/cuda_rasterizer.cu
        src/cpu_rasterizer.cpp
        src/ellipse.cpp
        src/ply_loader.cpp
        src/sh_color.cpp
    )

    target_link_libraries(viewer_progressive PRIVATE
        glm::glm
        glfw
        glad
        CUDA::cudart
        nlohmann_json::nlohmann_json
    )

    target_include_directories(viewer_progressive
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
endif()

set(_runtime_targets "")
if(TARGET viewer_main)
    list(APPEND _runtime_targets viewer_main)
endif()
if(TARGET viewer_progressive)
    list(APPEND _runtime_targets viewer_progressive)
endif()
if(TARGET MiniGSRenderer)
    list(APPEND _runtime_targets MiniGSRenderer)
endif()
if(_runtime_targets)
    # Put enabled executables in the same directory
    set_target_properties(${_runtime_targets} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    )
endif()

# Enable image writing only for the CLI target
if(TARGET MiniGSRenderer)
    target_compile_definitions(MiniGSRenderer PRIVATE GS_ENABLE_IMAGE_WRITE=1)
endif()

message(STATUS "✅ Viewer target configured")

# Use UTF-8 source encoding on MSVC to silence C4819 warnings
# Apply correctly per language to avoid passing `/utf-8` raw to NVCC
if(MSVC)
    if(TARGET MiniGSRenderer)
        target_compile_options(MiniGSRenderer PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/utf-8">
        )
    endif()
    if(TARGET viewer_main)
        target_compile_options(viewer_main PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/utf-8">
        )
    endif()
    if(TARGET viewer_progressive)
        target_compile_options(viewer_progressive PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/utf-8">
        )
    endif()
endif()

message(STATUS "✅ Setup Complete! Ready to build.")
