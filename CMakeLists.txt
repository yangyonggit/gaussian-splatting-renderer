cmake_minimum_required(VERSION 3.14)

# Prevent CMake from using vcpkg toolchain if set
set(CMAKE_TOOLCHAIN_FILE "" CACHE FILEPATH "CMake toolchain file" FORCE)

# Define the project name and version
project(MiniGSRenderer VERSION 1.0 LANGUAGES CXX CUDA)

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


# ----------------------------------------------------------
# 2. Source Code Configuration
# ----------------------------------------------------------
# Recursively find all .cpp and .cu files in the src/ directory
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

# Create the executable target
add_executable(MiniGSRenderer ${CPP_SOURCES} ${CUDA_SOURCES})

# ----------------------------------------------------------
# 3. Linking & Includes
# ----------------------------------------------------------

# Find CUDA Toolkit (required for cudaMemcpy, cudaMalloc, etc.)
find_package(CUDAToolkit REQUIRED)

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

message(STATUS "✅ Setup Complete! Ready to build.")