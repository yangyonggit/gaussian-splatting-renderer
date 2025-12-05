cmake_minimum_required(VERSION 3.14)

# Define the project name and version
project(MiniGSRenderer VERSION 1.0)

# Set C++ Standard to C++17 (Required for modern features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable standard compiler warnings (Good practice)
if(MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-Wall -Wextra)
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


# ----------------------------------------------------------
# 2. Source Code Configuration
# ----------------------------------------------------------
# Recursively find all .cpp files in the src/ directory
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create the executable target
add_executable(MiniGSRenderer ${SOURCES})

# ----------------------------------------------------------
# 3. Linking & Includes
# ----------------------------------------------------------

# Link GLM
# GLM provides a modern CMake target "glm::glm" that handles include paths automatically.
target_link_libraries(MiniGSRenderer PRIVATE glm::glm)

# Include STB directory
# STB is a header-only library without a CMake target, so we manually add the include path.
# ${stb_SOURCE_DIR} is automatically defined by FetchContent.
target_include_directories(MiniGSRenderer PRIVATE ${stb_SOURCE_DIR})

message(STATUS "✅ Setup Complete! Ready to build.")