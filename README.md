# Parallel Reductions Benchmark for CPUs & GPUs

![Parallel Reductions Benchmark](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/ParallelReductionsBenchmark.jpg?raw=true)

One of the canonical examples when designing parallel algorithms is implementing parallel tree-like reductions, which is a special case of accumulating a bunch of numbers located in a continuous block of memory.
In modern C++, most developers would call `std::accumulate(array.begin(), array.end(), 0)`, and in Python, it's just a `sum(array)`.
Implementing those operations with high utilization in many-core systems is surprisingly non-trivial and depends heavily on the hardware architecture.
Moreover, on arrays with billions of elements, the default `float` error mounts, and the results become inaccurate unless a [Kahan-like scheme](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) is used.

This repository contains several educational examples showcasing the performance differences between different solutions:

- Single-threaded but SIMD-accelerated code:
  - SSE, AVX, AVX-512 on x86.
  - 🔜 NEON and SVE on Arm.
- OpenMP `reduction` clause.
- Thrust with its `thrust::reduce`.
- CUDA kernels with and w/out warp-reductions.
- OpenCL kernels, eight of them.
- Parallel STL `<algorithm>` in GCC with Intel oneTBB.

Previously, it also compared ArrayFire, Halide, and Vulkan queues for SPIR-V kernels and SyCL.
Examples were collected from early 2010s until 2019 and later updated in 2022.

- [Lecture Slides](https://drive.google.com/file/d/16AicAl99t3ZZFnza04Wnw_Vuem0w8lc7/view?usp=sharing) from 2019.
- [CppRussia Talk](https://youtu.be/AA4RI6o0h1U) in Russia in 2019.
- [JetBrains Talk](https://youtu.be/BUtHOftDm_Y) in Germany & Russia in 2019.

## Build & Run

The following script will, by default, generate a 1GB array of numbers and reduce them using every available backend.
All the classical Google Benchmark arguments are supported, including `--benchmark_filter=opencl`.
All the library dependencies, including GTest, GBench, Intel oneTBB, FMT, and Thrust with CUB, will be automatically fetched.
You are expected to build this on an x86 machine with CUDA drivers installed.

```sh
cmake -B build_release
cmake --build build_release --config Release
build_release/reduce_bench                                 # Run all benchmarks
build_release/reduce_bench --benchmark_filter="cuda"       # Only CUDA-related
PARALLEL_REDUCTIONS_LENGTH=1024 build_release/reduce_bench # Set a different input size
```

Need a more fine-grained control to run only CUDA-based backends?

```sh
cmake -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -B build_release
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

Different hardware would yield different results, but the general trends and observations are:

- Accumulating over 100M `float` values generally requires `double` precision or Kahan-like numerical tricks to avoid instability.
- Carefully unrolled `for`-loop is easier for the compiler to vectorize and faster than `std::accumulate`.
- For `float`, `double`, and even Kahan-like schemes, hand-written AVX2 code is faster than auto-vectorization.
- Parallel `std::reduce` for extensive collections is naturally faster than serial `std::accumulate`, but you may not feel the difference between `std::execution::par` and `std::execution::par_unseq` on CPU.
- CUB is always faster than Thrust, and even for trivial types and large jobs, the difference can be 50%.

### Nvidia DGX-H100

On Nvidia DGX-H100 nodes, with GCC 12 and NVCC 12.1, one may expect the following results:

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
unrolled<f32>/min_time:10.000/real_time                149618549 ns    149615366 ns           95 bytes/s=7.17653G/s error,%=50
unrolled<f64>/min_time:10.000/real_time                146594731 ns    146593719 ns           95 bytes/s=7.32456G/s error,%=0
std::accumulate<f32>/min_time:10.000/real_time         194089563 ns    194088811 ns           72 bytes/s=5.5322G/s error,%=93.75
std::accumulate<f64>/min_time:10.000/real_time         192657883 ns    192657360 ns           74 bytes/s=5.57331G/s error,%=0
std::reduce<par, f32>/min_time:10.000/real_time          3749938 ns      3727477 ns         2778 bytes/s=286.336G/s error,%=0
std::reduce<par, f64>/min_time:10.000/real_time          3921280 ns      3916897 ns         3722 bytes/s=273.824G/s error,%=100
std::reduce<par_unseq, f32>/min_time:10.000/real_time    3884794 ns      3864061 ns         3644 bytes/s=276.396G/s error,%=0
std::reduce<par_unseq, f64>/min_time:10.000/real_time    3889332 ns      3866968 ns         3585 bytes/s=276.074G/s error,%=100
openmp<f32>/min_time:10.000/real_time                    5061544 ns      5043250 ns         2407 bytes/s=212.137G/s error,%=65.5651u
sse<f32aligned>@threads/min_time:10.000/real_time        5986350 ns      5193690 ns         2343 bytes/s=179.365G/s error,%=1.25021
avx2<f32>/min_time:10.000/real_time                    110796474 ns    110794861 ns          127 bytes/s=9.69112G/s error,%=50
avx2<f32kahan>/min_time:10.000/real_time               134144762 ns    134137771 ns          105 bytes/s=8.00435G/s error,%=0
avx2<f64>/min_time:10.000/real_time                    115791797 ns    115790878 ns          121 bytes/s=9.27304G/s error,%=0
avx2<f32aligned>@threads/min_time:10.000/real_time       5958283 ns      5041060 ns         2358 bytes/s=180.21G/s error,%=1.25033
avx2<f64>@threads/min_time:10.000/real_time              5996481 ns      5123440 ns         2337 bytes/s=179.062G/s error,%=1.25001
cub@cuda/min_time:10.000/real_time                        356488 ns       356482 ns        39315 bytes/s=3.012T/s error,%=0
warps@cuda/min_time:10.000/real_time                      486387 ns       486377 ns        28788 bytes/s=2.20759T/s error,%=0
thrust@cuda/min_time:10.000/real_time                     500941 ns       500919 ns        27512 bytes/s=2.14345T/s error,%=0
```

Observations:

- 286 GB/s upper bound on the CPU.
- 2.2 TB/s using vanilla CUDA approaches.
- 3 TB/s using CUB.

### AWS Zen4 `m7a.metal-48xl`

On AWS Zen4 `m7a.metal-48xl` instances with GCC 12, one may expect the following results:

```sh
$ build_release/reduce_bench
You did not feed the size of arrays, so we will use a 1GB array!
2025-01-18T11:26:46+00:00
Running build_release/reduce_bench
Run on (192 X 3701.95 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x192)
  L1 Instruction 32 KiB (x192)
  L2 Unified 1024 KiB (x192)
  L3 Unified 32768 KiB (x24)
Load Average: 4.54, 2.78, 4.94
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
----------------------------------------------------------------------------------------------------------------
Benchmark                                                      Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------
unrolled<f32>/min_time:10.000/real_time                 30546168 ns     30416147 ns          461 bytes/s=35.1514G/s error,%=50
unrolled<f64>/min_time:10.000/real_time                 31563095 ns     31447017 ns          442 bytes/s=34.0189G/s error,%=0
std::accumulate<f32>/min_time:10.000/real_time         219734340 ns    219326135 ns           64 bytes/s=4.88655G/s error,%=93.75
std::accumulate<f64>/min_time:10.000/real_time         219853985 ns    219429612 ns           64 bytes/s=4.88389G/s error,%=0
openmp<f32>/min_time:10.000/real_time                    5749979 ns      5709315 ns         1996 bytes/s=186.738G/s error,%=149.012u
std::reduce<par, f32>/min_time:10.000/real_time          2913596 ns      2827125 ns         4789 bytes/s=368.528G/s error,%=0
std::reduce<par, f64>/min_time:10.000/real_time          2899901 ns      2831183 ns         4874 bytes/s=370.268G/s error,%=0
std::reduce<par_unseq, f32>/min_time:10.000/real_time    3026168 ns      2940291 ns         4461 bytes/s=354.819G/s error,%=0
std::reduce<par_unseq, f64>/min_time:10.000/real_time    3053703 ns      2936506 ns         4797 bytes/s=351.62G/s error,%=0
sse<f32aligned>@threads/min_time:10.000/real_time       10132563 ns      9734108 ns         1000 bytes/s=105.969G/s error,%=0.520837
avx2<f32>/min_time:10.000/real_time                     32225620 ns     32045487 ns          435 bytes/s=33.3195G/s error,%=50
avx2<f32kahan>/min_time:10.000/real_time               110283627 ns    110023814 ns          127 bytes/s=9.73619G/s error,%=0
avx2<f64>/min_time:10.000/real_time                     55559986 ns     55422069 ns          247 bytes/s=19.3258G/s error,%=0
avx2<f32aligned>@threads/min_time:10.000/real_time       9612120 ns      9277454 ns         1467 bytes/s=111.707G/s error,%=0.521407
avx2<f64>@threads/min_time:10.000/real_time             10091882 ns      9708706 ns         1389 bytes/s=106.397G/s error,%=0.520837
avx512<f32streamed>/min_time:10.000/real_time           55713332 ns     55615555 ns          243 bytes/s=19.2726G/s error,%=50
avx512<f32streamed>@threads/min_time:10.000/real_time    9701513 ns      9383267 ns         1435 bytes/s=110.678G/s error,%=50.2604
avx512<f32unrolled>/min_time:10.000/real_time               48203352 ns     48085623 ns          228 bytes/s=22.2753G/s error,%=50
avx512<f32unrolled>@threads/min_time:10.000/real_time        9275968 ns      8955543 ns         1508 bytes/s=115.755G/s error,%=50.2604
avx512<f32interleaving>/min_time:10.000/real_time           40012581 ns     39939290 ns          352 bytes/s=26.8351G/s error,%=50
avx512<f32interleaving>@threads/min_time:10.000/real_time    9477545 ns      9168739 ns         1488 bytes/s=113.293G/s error,%=50.2581
```

One of the interesting observations is the effect of latency hiding, interleaving the operations executing on different ports of the same CPU.
It is evident when benchmarking AVX-512 kernels on very small arrays:

```sh
--------------------------------------------------------------------------------------------------------------------
Benchmark                                                          Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------------------------------
avx512<f32streamed>/min_time:10.000/real_time                   11.5 ns         11.5 ns   1213284345 bytes/s=355.396G/s
avx512<f32unrolled>/min_time:10.000/real_time                   13.2 ns         13.2 ns   1062115528 bytes/s=310.782G/s
avx512<f32interleaving>/min_time:10.000/real_time               8.21 ns         8.20 ns   1699079374 bytes/s=499.047G/s
```

The reason this happens is that on Zen4:

- Addition instructions like `vaddps zmm, zmm, zmm` and `vaddpd zmm, zmm, zmm` execute on ports 2 and 3.
- Fused-Multiply-Add instructions like `vfmadd132ps zmm, zmm, zmm` execute on ports 0 and 1.

So if the CPU can fetch enough data in time, we can have at least 4 ports simultaneously busy, and the latency of the operation is hidden.
