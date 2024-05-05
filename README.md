# Parallel Reductions Benchmark for CPUs & GPUs

![Parallel Reductions Benchmark](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/ParallelReductionsBenchmark.jpg?raw=true)

One of the canonical examples when designing parallel algorithms is implementing parallel reductions or its special case of accumulating a bunch of numbers located in a continuous block of memory.
In modern C++, most developers would call `std::accumulate(array.begin(), array.end(), 0)`, and in Python, it's just a `sum(array)`.
Implementing those operations with high utilization in many-core systems is surprisingly non-trivial and depends heavily on the hardware architecture.
This repository contains several educational examples showcasing the performance differences between different solutions:

- AVX2 single-threaded, but SIMD-parallel code.
- OpenMP `reduction` clause.
- Thrust with its `thrust::reduce`.
- CUDA kernels with warp-reductions.
- OpenCL kernels, eight of them.
- Parallel STL `<algorithm>'s in GCC with Intel oneTBB.

Previously, it also compared ArrayFire, Halide, and Vulkan queues for SPIR-V kernels and SyCL.
Examples were collected from early 2010s until 2019 and later updated in 2022.

- [Lecture Slides](blob/master/Presentation.pdf) from 2019.
- [CppRussia Talk](https://youtu.be/AA4RI6o0h1U) in Russia in 2019.
- [JetBrains Talk](https://youtu.be/BUtHOftDm_Y) in Germany & Russia in 2019.

## Build & Run in 1 Line

The following script will, by default, generate a 1GB array of numbers and reduce them using every available backend.
All the classical Google Benchmark arguments are supported, including `--benchmark_filter=opencl`.
All the library dependencies will be automatically fetched: GTest, GBench, Intel oneTBB, FMT, and Thrust with CUB.
You are expected to build this on an x86 machine with CUDA drivers installed.

```sh
cmake -B ./build_release
cmake --build ./build_release --config Release
./build_release/reduce_bench # To run all available benchmarks on default array size
./build_release/reduce_bench --benchmark_filter="" # Control Google Benchmark params
PARALLEL_REDUCTIONS_LENGTH=1000 ./build_release/reduce_bench # Try different array size
```

Need a more fine-grained control to run only CUDA-based backends?

```sh
cmake -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -B ./build_release
cmake --build ./build_release --config Release
./build_release/reduce_bench --benchmark_filter=cuda
```

To debug or introspect, the procedure is similar:

```sh
cmake -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug -B ./build_debug
cmake --build ./build_debug --config Debug
```

And then run your favorite debugger.

Optional backends:

- To enable [Intel OpenCL](https://github.com/intel/compute-runtime/blob/master/README.md) on CPUs: `apt-get install intel-opencl-icd`.
- To run on integrated Intel GPU, follow [this guide](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/prerequisites.html).
