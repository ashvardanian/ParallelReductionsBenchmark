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

## Build & Run

The following script will, by default, generate a 1GB array of numbers and reduce them using every available backend.
All the classical Google Benchmark arguments are supported, including `--benchmark_filter=opencl`.
All the library dependencies will be automatically fetched: GTest, GBench, Intel oneTBB, FMT, and Thrust with CUB.
You are expected to build this on an x86 machine with CUDA drivers installed.

```sh
cmake -B build_release
cmake --build build_release --config Release
build_release/reduce_bench # To run all available benchmarks on default array size
build_release/reduce_bench --benchmark_filter="" # Control Google Benchmark params
PARALLEL_REDUCTIONS_LENGTH=1000 build_release/reduce_bench # Try different array size
```

Need a more fine-grained control to run only CUDA-based backends?

```sh
cmake -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -B build_release
cmake --build build_release --config Release
build_release/reduce_bench --benchmark_filter=cuda
```

To debug or introspect, the procedure is similar:

```sh
cmake -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug --config Debug
```

And then run your favorite debugger.

Optional backends:

- To enable [Intel OpenCL](https://github.com/intel/compute-runtime/blob/master/README.md) on CPUs: `apt-get install intel-opencl-icd`.
- To run on integrated Intel GPU, follow [this guide](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/prerequisites.html).

## Results

Different hardware would obviously yield different results, but the general trends and observations are:

- Accumulating over 100M `float` values generally requires `double` precision or Kahan-like numerical tricks to avoid instability.
- Carefully unrolled `for`-loop is easier for compiler to vectorize and faster than `std::accumulate`.
- For `float`, `double`, and even Kahan-like schemes, hand-written AVX2 code is faster than autovectorization.
- Parallel `std::reduce` for large collections is obviously faster than serial `std::accumulate`, but you may not feel the difference between `std::execution::par` and `std::execution::par_unseq` on CPU.
- CUB is always faster than Thrust, and even for trivial types and large jobs the difference can be 50%.

On an Nvidia DGX-H100 node one may expect the following results:

```sh
$ build_release/reduce_bench
You did not feed the size of arrays, so we will use a 1GB array!
2024-05-06T00:11:14+00:00
Running build_release/reduce_bench
Run on (160 X 2100 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x160)
  L1 Instruction 32 KiB (x160)
  L2 Unified 4096 KiB (x80)
  L3 Unified 16384 KiB (x2)
Load Average: 3.23, 19.01, 13.71
----------------------------------------------------------------------------------------------------------------
Benchmark                                                      Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------
unrolled<f32>/min_time:10.000/real_time                142624550 ns    142624045 ns           96 bytes/s=7.52845G/s error,%=50
unrolled<f64>/min_time:10.000/real_time                142010374 ns    142010067 ns           94 bytes/s=7.56101G/s error,%=0
std::accumulate<f32>/min_time:10.000/real_time         190675394 ns    190674841 ns           73 bytes/s=5.63126G/s error,%=93.75
std::accumulate<f64>/min_time:10.000/real_time         189993910 ns    189992814 ns           72 bytes/s=5.65145G/s error,%=0
std::reduce<par, f32>/min_time:10.000/real_time          3654622 ns      3637280 ns         2969 bytes/s=293.804G/s error,%=0
std::reduce<par, f64>/min_time:10.000/real_time          3477476 ns      3462021 ns         3903 bytes/s=308.77G/s error,%=100
std::reduce<par_unseq, f32>/min_time:10.000/real_time    3463821 ns      3445200 ns         4023 bytes/s=309.988G/s error,%=0
std::reduce<par_unseq, f64>/min_time:10.000/real_time    3431538 ns      3401924 ns         4061 bytes/s=312.904G/s error,%=100
openmp<f32>/min_time:10.000/real_time                    4739561 ns      4724086 ns         2649 bytes/s=226.549G/s error,%=65.5651u
avx2<f32>/min_time:10.000/real_time                    118085908 ns    118085361 ns          117 bytes/s=9.09289G/s error,%=50
avx2<f32kahan>/min_time:10.000/real_time               143006790 ns    143003137 ns           99 bytes/s=7.50833G/s error,%=0
avx2<f64>/min_time:10.000/real_time                    123543746 ns    123533957 ns          111 bytes/s=8.69119G/s error,%=0
avx2<f32aligned>@threads/min_time:10.000/real_time       6725173 ns      6424451 ns         2084 bytes/s=159.66G/s error,%=1.25033
avx2<f64>@threads/min_time:10.000/real_time              6729024 ns      6366678 ns         2092 bytes/s=159.569G/s error,%=1.25001
sse<f32aligned>@threads/min_time:10.000/real_time        6263549 ns      5330562 ns         2155 bytes/s=171.427G/s error,%=1.25021
cub@cuda/min_time:10.000/real_time                        357435 ns       357428 ns        39175 bytes/s=3.00402T/s error,%=0
warps@cuda/min_time:10.000/real_time                      488410 ns       488400 ns        28671 bytes/s=2.19844T/s error,%=0
thrust@cuda/min_time:10.000/real_time                     503945 ns       503928 ns        27466 bytes/s=2.13067T/s error,%=0
```