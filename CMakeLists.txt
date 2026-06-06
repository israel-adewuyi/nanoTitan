cmake_minimum_required(VERSION 3.18)

project(nanoTitan
    VERSION 0.1.0
    LANGUAGES CXX
)

# -----------------------------
# C++ settings
# -----------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------
# PyTorch / CUDA / NCCL
# -----------------------------

# Prefer passing CMAKE_PREFIX_PATH from the command line, but keep this fallback.
list(APPEND CMAKE_PREFIX_PATH
    "/home/israel/nanoTitan/.venv/lib/python3.13/site-packages/torch"
)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find NCCL from the PyTorch wheel first, then system locations.
find_library(NCCL_LIBRARY
    NAMES nccl
    PATHS
        "/home/israel/nanoTitan/.venv/lib/python3.13/site-packages/nvidia/nccl/lib"
        "/usr/local/cuda-12.2/lib64"
        "/usr/local/cuda/lib64"
        "/usr/lib/x86_64-linux-gnu"
        "/usr/lib"
)

if (NOT NCCL_LIBRARY)
    message(FATAL_ERROR "Could not find NCCL library")
endif()

message(STATUS "Found NCCL: ${NCCL_LIBRARY}")

# Helpful compile warnings
if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------
# Core native library
# -----------------------------

add_library(nanotitan_core
    csrc/random.cpp
    csrc/runtime/cuda_stream.cpp
)

target_include_directories(nanotitan_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/runtime
        ${TORCH_INCLUDE_DIRS}
)

target_compile_options(nanotitan_core
    PRIVATE
        ${TORCH_CXX_FLAGS}
)

target_link_libraries(nanotitan_core
    PUBLIC
        ${TORCH_LIBRARIES}
        CUDA::cudart
        ${NCCL_LIBRARY}
)

# -----------------------------
# Benchmark executable
# -----------------------------

add_executable(nanotitan_bench
    benchmarks/bench.cpp
)

target_link_libraries(nanotitan_bench
    PRIVATE
        nanotitan_core
)

target_include_directories(nanotitan_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/runtime
)

# -----------------------------
# Tests
# -----------------------------

option(NANOTITAN_BUILD_TESTS "Build C++ tests" ON)

if (NANOTITAN_BUILD_TESTS)
    include(CTest)
    enable_testing()

    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

    file(GLOB_RECURSE NANOTITAN_CPP_TESTS
        CONFIGURE_DEPENDS
        tests/cpp/*.cpp
    )

    if (NANOTITAN_CPP_TESTS)
        add_executable(nanotitan_cpp_tests
            ${NANOTITAN_CPP_TESTS}
        )

        target_link_libraries(nanotitan_cpp_tests
            PRIVATE
                nanotitan_core
                GTest::gtest_main
        )

        target_include_directories(nanotitan_cpp_tests
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/csrc
                ${CMAKE_CURRENT_SOURCE_DIR}/csrc/runtime
        )

        include(GoogleTest)
        gtest_discover_tests(nanotitan_cpp_tests)
    else()
        message(STATUS "No C++ test files found in tests/cpp/")
    endif()
endif()