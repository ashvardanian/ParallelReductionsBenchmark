cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Get external content: Google Benchmark, Thrust
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
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
set(BENCHMARK_ENABLE_DOXYGEN OFF)
set(BENCHMARK_INSTALL_DOCS OFF)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_USE_BUNDLED_GTEST ON)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.7.0
)
FetchContent_MakeAvailable(benchmark)

# Thrust can be configured with Intel TBB.
# https://github.com/rapidsai/cuml/issues/3540
# https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md
FetchContent_Declare(
    thrust
    GIT_REPOSITORY https://github.com/thrust/thrust.git
    GIT_TAG 1.15.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(thrust)

include_directories(BEFORE ${thrust_SOURCE_DIR})

# We need TBB for Parallel CPU Algorithms in GCC.
# https://github.com/oneapi-src/oneTBB/blob/onetbb_2021/cmake/README.md
FetchContent_Declare(
    TBB
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG v2021.5.0
    GIT_SHALLOW TRUE
)
FetchContent_Populate(TBB)

if(NOT TBB_POPULATED)
    include_directories(BEFORE ${TBB_SOURCE_DIR})
endif()

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

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(CMAKE_GCC_FLAGS "${CMAKE_GCC_FLAGS} -march=native -fopenmp")

# List of all possibkle compiler IDs:
# https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_GCC_FLAGS}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVIDIA" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set(CMAKE_CXX_FLAGS "${CMAKE_GCC_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_GCC_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1")
endif()

# Add CUDA and Thrust dependencies
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)
find_package(CUDA)

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

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_GCC_FLAGS}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVIDIA" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set_property(SOURCE reduce_bench.cpp PROPERTY LANGUAGE CUDA)
    set_target_properties(reduce_bench PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(reduce_bench PROPERTIES CUDA_ARCHITECTURES "86")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
endif()