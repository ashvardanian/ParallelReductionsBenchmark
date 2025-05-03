# Let's use a recent CMake version:
#
# * 3.16+ for native sanitizers support
# * 3.17+ for `FindCUDAToolkit`
# * 3.18 for BLAS::BLAS target
# * 3.25.2 for CUDA20 support
#
# The good news is that Ubuntu 24.04 comes with 3.28!
cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)

# ------------------------------------------------------------------------------
# Project Setup
# ------------------------------------------------------------------------------
project(
    ParallelReductionsBenchmark
    VERSION 0.4.0
    LANGUAGES CXX
    DESCRIPTION "Parallel Reductions Benchmark for CPUs & GPUs"
    HOMEPAGE_URL "https://github.com/ashvardanian/ParallelReductionsBenchmark"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

# Some extra logging for the user:
message(STATUS "----------------------------------------")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "----------------------------------------")

# Default to Release if no build type is set:
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# ------------------------------------------------------------------------------
# Detect CUDA Support
# ------------------------------------------------------------------------------
set(ENABLE_CUDA OFF)
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(ENABLE_CUDA ON)
    message(STATUS "CUDA detected! Using compiler: ${CMAKE_CUDA_COMPILER}")
else ()
    message(STATUS "CUDA not detected. Skipping CUDA-specific builds.")
endif ()

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(USE_INTEL_TBB "Use Intel TBB for parallel STL algorithms" ON)
option(USE_NVIDIA_CCCL "Use Nvidia CCCL for CUDA acceleration" ON)

# Enable or disable options based on system and CUDA support
if (ENABLE_CUDA)
    set(USE_NVIDIA_CCCL ON)
    set(USE_INTEL_TBB OFF) # Prioritize CUDA acceleration
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(USE_NVIDIA_CCCL OFF) # Can't compile CCCL w/o CUDA
    set(USE_INTEL_TBB ON) # Default to TBB on Linux without CUDA
endif ()

message(STATUS "USE_INTEL_TBB: ${USE_INTEL_TBB}")
message(STATUS "USE_NVIDIA_CCCL: ${USE_NVIDIA_CCCL}")

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenMP QUIET)
find_package(OpenCL QUIET)

set(FETCHCONTENT_QUIET OFF)
include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.1.2
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
    fork_union
    GIT_REPOSITORY https://github.com/ashvardanian/fork_union.git
    GIT_TAG main
)
FetchContent_MakeAvailable(fork_union)

# Fetch GBenchmark and suppress internal tests.
# https://github.com/google/benchmark/blob/main/docs/user_guide.md#using-register-benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
set(BENCHMARK_ENABLE_DOXYGEN OFF)
set(BENCHMARK_INSTALL_DOCS OFF)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_USE_BUNDLED_GTEST ON)
FetchContent_MakeAvailable(benchmark)

# Intel TBB for "Parallel STL" algorithms: https://github.com/oneapi-src/oneTBB/tree/onetbb_2021
if (USE_INTEL_TBB)
    FetchContent_Declare(
        IntelTBB
        GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
        GIT_TAG master
    )

    # Suppress TBB's own tests:
    set(TBB_TEST
        OFF
        CACHE BOOL "Do not build TBB tests" FORCE
    )
    FetchContent_MakeAvailable(IntelTBB)

    # ------------------------------------------------------------------------------
    # TBB fix for -Wstringop-overflow warnings treated as errors
    # ------------------------------------------------------------------------------
    # The TBB library target is typically called "tbb". We can explicitly disable the `stringop-overflow` warning for
    # TBB only:
    if (TARGET tbb)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(tbb PRIVATE -Wno-stringop-overflow)
        endif ()
    endif ()
endif ()

# Nvidia's CUDA Core Compute Libraries for GPU acceleration
if (USE_NVIDIA_CCCL)
    # CUB, Thrust, and other libraries of interest are now included into the CUDA Toolkit, so we don't need this
    # anymore:
    #
    # FetchContent_Declare(NvidiaCCCL GIT_REPOSITORY https://github.com/nvidia/cccl.git)
    # FetchContent_MakeAvailable(NvidiaCCCL)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit Include Path: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA Toolkit Libraries Path: ${CUDAToolkit_LIBRARY_DIR}")
endif ()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(CMAKE_GCC_FLAGS "${CMAKE_GCC_FLAGS} -march=native -fopenmp")

add_executable(reduce_bench reduce_bench.cpp)
target_link_libraries(reduce_bench PRIVATE benchmark::benchmark fmt::fmt fork_union Threads::Threads BLAS::BLAS)

if (USE_INTEL_TBB)
    target_link_libraries(reduce_bench PRIVATE TBB::tbb)
endif ()

if (USE_NVIDIA_CCCL)
    target_link_libraries(reduce_bench PRIVATE CUDA::cudart CUDA::cublas)
endif ()

if (OpenMP_FOUND)
    target_link_libraries(reduce_bench PRIVATE OpenMP::OpenMP_CXX)
endif ()

if (OpenCL_FOUND)
    target_link_libraries(reduce_bench PRIVATE OpenCL::OpenCL)
endif ()

# On Linux we use libnuma for NUMA support.
if (UNIX AND NOT APPLE)
    find_path(
        NUMA_INCLUDE_DIRS
        NAMES numa.h
        HINTS $ENV{HOME}/local/include /opt/local/include /usr/local/include /usr/include
    )

    find_library(
        NUMA_LIBRARIES
        NAMES numa
        HINTS $ENV{HOME}/local/lib64
              $ENV{HOME}/local/lib
              /usr/local/lib64
              /usr/local/lib
              /opt/local/lib64
              /opt/local/lib
              /usr/lib64
              /usr/lib
              /usr/lib/x86_64-linux-gnu/
              /usr/lib/aarch64-linux-gnu/
    )

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_LIBRARIES NUMA_INCLUDE_DIRS)

    if (NUMA_FOUND)
        message(STATUS "Found NUMA: includes=${NUMA_INCLUDE_DIRS}, library=${NUMA_LIBRARIES}")
        target_include_directories(reduce_bench PRIVATE ${NUMA_INCLUDE_DIRS})
        target_link_libraries(reduce_bench PRIVATE ${NUMA_LIBRARIES})
    else ()
        message(WARNING "NUMA library not found.")
        message(WARNING "Please install it using your package manager:")
        message(WARNING "    Debian/Ubuntu: sudo apt-get install libnuma1 libnuma-dev")
        message(WARNING "    RHEL/CentOS: sudo yum install numactl numactl-devel")
    endif ()
endif ()

# List of all possible compiler IDs: https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR CMAKE_CUDA_COMPILER_ID STREQUAL "NVHPC")
    enable_language(CUDA)
    message("-- Detected Nvidia Compiler")
    set_property(SOURCE reduce_bench.cpp PROPERTY LANGUAGE CUDA)
    set_target_properties(reduce_bench PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(reduce_bench PROPERTIES CUDA_ARCHITECTURES "86")

    set(CMAKE_CXX_FLAGS "${CMAKE_GCC_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_GCC_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND 0)
    set(METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/reduce_metal.msl)
    set(METALLIB_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/reduce_metal.metallib)

    add_custom_command(
        OUTPUT ${METALLIB_OUTPUT}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SRC} -o reduce_metal.air
        COMMAND xcrun -sdk macosx metallib reduce_metal.air -o ${METALLIB_OUTPUT}
        DEPENDS ${METAL_SRC}
        COMMENT "Compiling Metal shader: ${METAL_SRC} -> ${METALLIB_OUTPUT}"
        VERBATIM
    )

    # Create a pseudo target to build the metallib
    add_custom_target(MetalLibBuild ALL DEPENDS ${METALLIB_OUTPUT})

    enable_language(OBJCXX)
    set_property(SOURCE reduce_metal.hpp PROPERTY LANGUAGE OBJCXX)
    set_property(SOURCE reduce_bench.cpp PROPERTY LANGUAGE OBJCXX)
    target_link_libraries(reduce_bench PRIVATE "-framework Metal" "-framework Foundation")
    set_source_files_properties(reduce_bench.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++ -fobjc-arc")

    # Make sure reduce_bench depends on MetalLibBuild so the .metallib is built first
    add_dependencies(reduce_bench MetalLibBuild)

    # Copy the `metallib` to the same folder as reduce_bench
    add_custom_command(
        TARGET reduce_bench
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${METALLIB_OUTPUT} $<TARGET_FILE_DIR:reduce_bench>
    )

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message("-- Detected Clang Compiler")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message("-- Detected GCC Compiler")
    set(CMAKE_CXX_FLAGS "${CMAKE_GCC_FLAGS}")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message("-- Detected Intel Compiler")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1")
endif ()
