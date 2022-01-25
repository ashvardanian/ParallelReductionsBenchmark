cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Get external content: Google Benchmark, Thrust
include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(fmt)

# Fetch GTest, as its needed for GBenchmark
# https://stackoverflow.com/a/60417587
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.11.0
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

# Fetch GBenchmark and surpress internal tests.
# https://github.com/google/benchmark/blob/main/docs/user_guide.md#using-register-benchmark
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.6.1  
)
set(BENCHMARK_ENABLE_TESTING off)
FetchContent_GetProperties(googlebenchmark)
if(NOT googlebenchmark_POPULATED)
    FetchContent_Populate(googlebenchmark)
    add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

# Thrust can be configured with Intel TBB.
# https://github.com/rapidsai/cuml/issues/3540
# https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md
FetchContent_Declare(
    thrust
    GIT_REPOSITORY https://github.com/thrust/thrust.git
    GIT_TAG        1.15.0
)
FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
    FetchContent_Populate(thrust)
endif()
include_directories(BEFORE ${thrust_SOURCE_DIR})

# Define the CUDA architectures.
# Empty value is forbidden.
set(CMAKE_CUDA_ARCHITECTURES OFF)
string(APPEND CMAKE_CUDA_FLAGS " -arch=sm_75")
project(gpgpu_frameworks LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Make Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61 --expt-relaxed-constexpr --extended-lambda -march=native")

# Add CUDA and Thrust dependencies
find_package(CUDA REQUIRED)
find_package(OpenMP)
find_package(OpenCL)

# https://developer.nvidia.com/blog/building-cuda-applications-cmake/
add_executable(reduce_bench reduce_bench.cu)
set_target_properties(reduce_bench PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(reduce_bench benchmark::benchmark fmt::fmt Threads::Threads OpenCL::OpenCL)