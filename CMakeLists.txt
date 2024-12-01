cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Define the CUDA architectures.
# Empty value is forbidden.
project(ParallelReductionsBenchmark LANGUAGES CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Make Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Get external content: Google Benchmark, cccl
include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fmt)

# Fetch GBenchmark and surpress internal tests.
# https://github.com/google/benchmark/blob/main/docs/user_guide.md#using-register-benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
set(BENCHMARK_ENABLE_DOXYGEN OFF)
set(BENCHMARK_INSTALL_DOCS OFF)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_USE_BUNDLED_GTEST ON)
FetchContent_MakeAvailable(benchmark)

# Nvidia CCCL can be configured with Intel TBB.
# https://github.com/rapidsai/cuml/issues/3540
FetchContent_Declare(
    cccl
    GIT_REPOSITORY https://github.com/nvidia/cccl.git
    GIT_TAG v2.2.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cccl)

# We need TBB for Parallel CPU Algorithms in GCC.
# https://github.com/oneapi-src/oneTBB/blob/onetbb_2021/cmake/README.md
FetchContent_Declare(
    TBB
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG v2021.11.0
    GIT_SHALLOW TRUE
)
set(TBB_TEST OFF CACHE BOOL "Do not build TBB tests" FORCE)
FetchContent_MakeAvailable(TBB)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(CMAKE_GCC_FLAGS "${CMAKE_GCC_FLAGS} -march=native -fopenmp")

# Add CUDA and cccl dependencies
find_package(Threads REQUIRED)

add_executable(reduce_bench reduce_bench.cpp)
target_link_libraries(reduce_bench benchmark::benchmark fmt::fmt Threads::Threads TBB::tbb)

find_package(OpenMP)

if(OpenMP_FOUND)
    target_link_libraries(reduce_bench OpenMP::OpenMP_CXX)
endif()

find_package(OpenCL)

if(OpenCL_FOUND)
    target_link_libraries(reduce_bench OpenCL::OpenCL)
endif()

# List of all possibkle compiler IDs:
# https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR CMAKE_CUDA_COMPILER_ID STREQUAL "NVHPC")
    enable_language(CUDA)
    message("-- Detected Nvidia Compiler")
    set_property(SOURCE reduce_bench.cpp PROPERTY LANGUAGE CUDA)
    set_target_properties(reduce_bench PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(reduce_bench PROPERTIES CUDA_ARCHITECTURES "86")

    set(CMAKE_CXX_FLAGS "${CMAKE_GCC_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_GCC_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message("-- Detected Clang Compiler")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message("-- Detected GCC Compiler")
    set(CMAKE_CXX_FLAGS "${CMAKE_GCC_FLAGS}")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message("-- Detected Intel Compiler")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1")
endif()