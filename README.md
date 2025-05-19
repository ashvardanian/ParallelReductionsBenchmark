# Parallel Reductions Benchmark for CPUs & GPUs

![Parallel Reductions Benchmark](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/ParallelReductionsBenchmark.jpg?raw=true)

One of the canonical examples when designing parallel algorithms is implementing parallel tree-like reductions, which is a special case of accumulating a bunch of numbers located in a continuous block of memory.
In modern C++, most developers would call `std::accumulate(array.begin(), array.end(), 0)`, and in Python, it's just a `sum(array)`.
Implementing those operations with high utilization in many-core systems is surprisingly non-trivial and depends heavily on the hardware architecture.
This repository contains several educational examples showcasing the performance differences between different solutions:

- Single-threaded but SIMD-accelerated code:
  - SSE, AVX, AVX-512 on x86.
  - NEON and SVE on Arm.
- OpenMP `reduction` clause.
- Thrust with its `thrust::reduce`.
- CUB with its `cub::DeviceReduce::Sum`.
- CUDA kernels with and w/out [warp-primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/).
- CUDA kernels with [Tensor-Core](https://www.nvidia.com/en-gb/data-center/tensor-cores/) acceleration.
- [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) and cuBLAS strided vector and matrix routines.
- OpenCL kernels, eight of them.
- Parallel STL `<algorithm>` in GCC with Intel oneTBB.

Notably:

- on arrays with billions of elements, the default `float` error mounts, and the results become inaccurate unless a [Kahan-like scheme](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) is used.
- to minimize the overhead [Translation Lookaside Buffer](https://en.wikipedia.org/wiki/Translation_lookaside_buffer) __(TLB)__ misses, the arrays are aligned to the OS page size and are allocated in [huge pages on Linux](https://wiki.debian.org/Hugepages), if possible.
- to reduce the memory access latency on many-core  [Non-Uniform Memory Access](https://en.wikipedia.org/wiki/Non-uniform_memory_access) __(NUMA)__ systems, `libnuma` and `pthread` help maximize data affinity.
- to "hide" latency on wide CPU registers (like `ZMM`), expensive Assembly instructions executed on different [CPU ports](https://easyperf.net/blog/2018/03/21/port-contention#utilizing-full-capacity-of-the-load-instructions) are interleaved.

---

The examples in this repository were originally written in early 2010s and were updated in 2019, 2022, and 2025.
Previously, it also included ArrayFire, Halide, and Vulkan queues for SPIR-V kernels and SyCL.

- [Lecture Slides](https://drive.google.com/file/d/16AicAl99t3ZZFnza04Wnw_Vuem0w8lc7/view?usp=sharing) from 2019.
- [CppRussia Talk](https://youtu.be/AA4RI6o0h1U) in Russia in 2019.
- [JetBrains Talk](https://youtu.be/BUtHOftDm_Y) in Germany & Russia in 2019.

## Build & Run

This repository is a CMake project designed to be built on Linux with GCC, Clang, or NVCC.
You may need to install the following dependencies for complete functionality:

```sh
sudo apt install libblas-dev            # For OpenBLAS on Linux
sudo apt install libnuma1 libnuma-dev   # For NUMA allocators on Linux
sudo apt install cuda-toolkit           # This may not be as easy 😈
```

The following script will, by default, generate a 1GB array of numbers and reduce them using every available backend.
All the classical Google Benchmark arguments are supported, including `--benchmark_filter=opencl`.
All the library dependencies, including GTest, GBench, Intel oneTBB, FMT, and Thrust with CUB, will be automatically fetched.
You are expected to build this on an x86 machine with CUDA drivers installed.

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release         # Generate the build files
cmake --build build_release --config Release -j            # Build the project
build_release/reduce_bench                                 # Run all benchmarks
build_release/reduce_bench --benchmark_filter="cuda"       # Only CUDA-related
PARALLEL_REDUCTIONS_LENGTH=1024 build_release/reduce_bench # Set a different input size
```

Need a more fine-grained control to run only CUDA-based backends?

```sh
cmake -D CMAKE_CUDA_COMPILER=nvcc -D CMAKE_C_COMPILER=gcc-12 -D CMAKE_CXX_COMPILER=g++-12 -B build_release
cmake --build build_release --config Release -j
build_release/reduce_bench --benchmark_filter=cuda
```

Want to use the non-default Clang distribution on macOS?
OpenBLAS will be superseded by Apple's `Accelerate.framework`, but LLVM and OpenMP should ideally be pulled from Homebrew:

```sh
brew install llvm libomp
cmake -B build_release \
  -D CMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++ \
  -D OpenMP_ROOT=$(brew --prefix llvm)          \
  -D CMAKE_BUILD_RPATH=$(brew --prefix llvm)/lib \
  -D CMAKE_INSTALL_RPATH=$(brew --prefix llvm)/lib
cmake --build build_release --config Release -j
build_release/reduce_bench
```

To debug or introspect, the procedure is similar:

```sh
cmake -D CMAKE_CUDA_COMPILER=nvcc -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ -D CMAKE_BUILD_TYPE=Debug -B build_debug
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

Running build_release/reduce_bench
Run on (160 X 2100 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x160)
  L1 Instruction 32 KiB (x160)
  L2 Unified 4096 KiB (x80)
  L3 Unified 16384 KiB (x2)
Load Average: 3.23, 19.01, 13.71
--------------------------------------------------------------------------------------
Benchmark                            Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------
unrolled<f32>                149618549 ns    149615366 ns           95 bytes/s=7.17653G/s error,%=50
unrolled<f64>                146594731 ns    146593719 ns           95 bytes/s=7.32456G/s error,%=0
std::accumulate<f32>         194089563 ns    194088811 ns           72 bytes/s=5.5322G/s error,%=93.75
std::accumulate<f64>         192657883 ns    192657360 ns           74 bytes/s=5.57331G/s error,%=0
openmp<f32>                    5061544 ns      5043250 ns         2407 bytes/s=212.137G/s error,%=65.5651u
std::reduce<par, f32>          3749938 ns      3727477 ns         2778 bytes/s=286.336G/s error,%=0
std::reduce<par, f64>          3921280 ns      3916897 ns         3722 bytes/s=273.824G/s error,%=100
std::reduce<par_unseq, f32>    3884794 ns      3864061 ns         3644 bytes/s=276.396G/s error,%=0
std::reduce<par_unseq, f64>    3889332 ns      3866968 ns         3585 bytes/s=276.074G/s error,%=100
sse<f32aligned>@threads        5986350 ns      5193690 ns         2343 bytes/s=179.365G/s error,%=1.25021
avx2<f32>                    110796474 ns    110794861 ns          127 bytes/s=9.69112G/s error,%=50
avx2<f32kahan>               134144762 ns    134137771 ns          105 bytes/s=8.00435G/s error,%=0
avx2<f64>                    115791797 ns    115790878 ns          121 bytes/s=9.27304G/s error,%=0
avx2<f32aligned>@threads       5958283 ns      5041060 ns         2358 bytes/s=180.21G/s error,%=1.25033
avx2<f64>@threads              5996481 ns      5123440 ns         2337 bytes/s=179.062G/s error,%=1.25001
cub@cuda                        356488 ns       356482 ns        39315 bytes/s=3.012T/s error,%=0
warps@cuda                      486387 ns       486377 ns        28788 bytes/s=2.20759T/s error,%=0
thrust@cuda                     500941 ns       500919 ns        27512 bytes/s=2.14345T/s error,%=0
```

Observations:

- 286 GB/s upper bound on the CPU.
- 2.2 TB/s using vanilla CUDA approaches.
- 3 TB/s using CUB.

On Nvidia H200 GPUs, the numbers are even higher:

```sh
-----------------------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations UserCounters...
-----------------------------------------------------------------------------------
cuda/cub                     254609 ns       254607 ns        54992 bytes/s=4.21723T/s error,%=0
cuda/thrust                  319709 ns       316368 ns        43846 bytes/s=3.3585T/s error,%=0
cuda/thrust/interleaving     318598 ns       314996 ns        43956 bytes/s=3.37021T/s error,%=0
```

### AWS Zen4 `m7a.metal-48xl`

On AWS Zen4 `m7a.metal-48xl` instances with GCC 12, one may expect the following results:

```sh
$ build_release/reduce_bench
You did not feed the size of arrays, so we will use a 1GB array!

Running build_release/reduce_bench
Run on (192 X 3701.95 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x192)
  L1 Instruction 32 KiB (x192)
  L2 Unified 1024 KiB (x192)
  L3 Unified 32768 KiB (x24)
Load Average: 4.54, 2.78, 4.94
------------------------------------------------------------------------------------------
Benchmark                                Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------------
unrolled<f32>                     30546168 ns     30416147 ns          461 bytes/s=35.1514G/s error,%=50
unrolled<f64>                     31563095 ns     31447017 ns          442 bytes/s=34.0189G/s error,%=0
std::accumulate<f32>             219734340 ns    219326135 ns           64 bytes/s=4.88655G/s error,%=93.75
std::accumulate<f64>             219853985 ns    219429612 ns           64 bytes/s=4.88389G/s error,%=0
openmp<f32>                        5749979 ns      5709315 ns         1996 bytes/s=186.738G/s error,%=149.012u
std::reduce<par, f32>              2913596 ns      2827125 ns         4789 bytes/s=368.528G/s error,%=0
std::reduce<par, f64>              2899901 ns      2831183 ns         4874 bytes/s=370.268G/s error,%=0
std::reduce<par_unseq, f32>        3026168 ns      2940291 ns         4461 bytes/s=354.819G/s error,%=0
std::reduce<par_unseq, f64>        3053703 ns      2936506 ns         4797 bytes/s=351.62G/s error,%=0
sse<f32aligned>@threads           10132563 ns      9734108 ns         1000 bytes/s=105.969G/s error,%=0.520837
avx2<f32>                         32225620 ns     32045487 ns          435 bytes/s=33.3195G/s error,%=50
avx2<f32kahan>                   110283627 ns    110023814 ns          127 bytes/s=9.73619G/s error,%=0
avx2<f64>                         55559986 ns     55422069 ns          247 bytes/s=19.3258G/s error,%=0
avx2<f32aligned>@threads           9612120 ns      9277454 ns         1467 bytes/s=111.707G/s error,%=0.521407
avx2<f64>@threads                 10091882 ns      9708706 ns         1389 bytes/s=106.397G/s error,%=0.520837
avx512<f32streamed>               55713332 ns     55615555 ns          243 bytes/s=19.2726G/s error,%=50
avx512<f32streamed>@threads        9701513 ns      9383267 ns         1435 bytes/s=110.678G/s error,%=50.2604
avx512<f32unrolled>               48203352 ns     48085623 ns          228 bytes/s=22.2753G/s error,%=50
avx512<f32unrolled>@threads        9275968 ns      8955543 ns         1508 bytes/s=115.755G/s error,%=50.2604
avx512<f32interleaving>           40012581 ns     39939290 ns          352 bytes/s=26.8351G/s error,%=50
avx512<f32interleaving>@threads    9477545 ns      9168739 ns         1488 bytes/s=113.293G/s error,%=50.2581
```

Observations:

- 370 GB/s can be reached in dual-socket DDR5 setups with 12 channel memory.
- Using Kahan-like schemes is 3x slower than pure `float` and 2x slower than `double`.

One of the interesting observations is the effect of [latency hiding, interleaving the operations executing on different ports of the same CPU](https://ashvardanian.com/posts/cpu-ports).
It is evident when benchmarking AVX-512 kernels on very small arrays:

```sh
-------------------------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------
avx512/f32/streamed                    19.4 ns         19.4 ns    724081506 bytes/s=211.264G/s
avx512/f32/unrolled                    15.1 ns         15.1 ns    934282388 bytes/s=271.615G/s
avx512/f32/interleaving                12.3 ns         12.3 ns   1158791855 bytes/s=332.539G/s
```

The reason this happens is that on Zen4:

- Addition instructions like `vaddps zmm, zmm, zmm` and `vaddpd zmm, zmm, zmm` execute on ports 2 and 3.
- Fused-Multiply-Add instructions like `vfmadd132ps zmm, zmm, zmm` execute on ports 0 and 1.

So if the CPU can fetch enough data in time, we can have at least 4 ports simultaneously busy, and the latency of the operation is hidden.

### AWS Graviton4 `c8g.metal-24xl`

On AWS Graviton4 `c8g.metal-24xl` instances with GCC 12, one may expect the following results:

```sh
$ build_release/reduce_bench 
You did not feed the size of arrays, so we will use a 1GB array!
Page size: 4096 bytes
Cache line size: 64 bytes
Dataset size: 268435456 elements
Dataset alignment: 64 bytes
Dataset allocation type: mmap
Dataset NUMA nodes: 1

Running build_release/reduce_bench
Run on (96 X 2000 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB (x96)
  L1 Instruction 64 KiB (x96)
  L2 Unified 2048 KiB (x96)
  L3 Unified 36864 KiB (x1)
Load Average: 5.76, 6.38, 2.75
-------------------------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------
unrolled/f32                 38034000 ns     38033650 ns          368 bytes/s=28.2311G/s error,%=50
unrolled/f64                 72851731 ns     72852189 ns          192 bytes/s=14.7387G/s error,%=0
std::accumulate/f32         192162701 ns    192164003 ns           73 bytes/s=5.58767G/s error,%=93.75
std::accumulate/f64         192266754 ns    192268708 ns           73 bytes/s=5.58465G/s error,%=0
serial/f32/av::fork_union     1889686 ns      1889604 ns         7320 bytes/s=568.212G/s error,%=0
serial/f64/av::fork_union     1935453 ns      1935360 ns         7309 bytes/s=554.775G/s error,%=0
serial/f32/openmp             2244099 ns      2108568 ns         4723 bytes/s=478.473G/s error,%=71.5256u
std::reduce<par>/f32          1950894 ns      1950842 ns         7129 bytes/s=550.384G/s error,%=0
std::reduce<par>/f64          1959062 ns      1953907 ns         7121 bytes/s=548.09G/s error,%=0
std::reduce<par_unseq>/f32    1956428 ns      1949906 ns         7139 bytes/s=548.828G/s error,%=0
std::reduce<par_unseq>/f64    1953465 ns      1952599 ns         7117 bytes/s=549.66G/s error,%=0
neon/f32                     48248562 ns     48249488 ns          290 bytes/s=22.2544G/s error,%=75
neon/f32/av::fork_union       1890173 ns      1887574 ns         7354 bytes/s=568.065G/s error,%=0
neon/f32/std::threads         3321599 ns      3181368 ns         4221 bytes/s=323.261G/s error,%=1.04167
neon/f32/openmp               1901684 ns      1899327 ns         7263 bytes/s=564.627G/s error,%=23.8419u
sve/f32                      50048126 ns     50049059 ns          280 bytes/s=21.4542G/s error,%=75
sve/f32/av::fork_union        1898117 ns      1897862 ns         7329 bytes/s=565.688G/s error,%=0
sve/f32/std::threads          3347690 ns      3203386 ns         4190 bytes/s=320.741G/s error,%=1.04167
sve/f32/openmp                1909972 ns      1901816 ns         7274 bytes/s=562.177G/s error,%=23.8419u
```

Amazon's Graviton CPUs configured with a single NUMA node and Arm's native support for "weak memory model" make it the perfect ground for studying the cost of various concurrency synchronization primitives.
For that, we can launch the benchmark with a tiny input, such as just 1 scalar per core, and measure the overall latency of dispatching all threads, blocking, and afterwards aggregating partial results.
Assuming, some of the work scheduling happens at a cache-line granularity, instead of 1 scalar per core, we take 1 cache-line per core.

> 64 bytes / 4 bytes per scalar * 96 cores = 1536 scalars.

```sh
$ PARALLEL_REDUCTIONS_LENGTH=1536 build_release/reduce_bench --benchmark_filter="(sve/f32)(.*)(openmp|fork_union|taskflow)"

------------------------------------------------------------------------------------
Benchmark                          Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------
sve/f32/std::threads         2047275 ns      2008267 ns        13751 bytes/s=3.00106M/s error,%=0
sve/f32/tf::taskflow          109782 ns       106764 ns       254660 bytes/s=55.9655M/s error,%=0
sve/f32/av::fork_union         13136 ns        13136 ns      2117597 bytes/s=467.714M/s error,%=0
sve/f32/openmp                 10494 ns        10256 ns      2848849 bytes/s=585.492M/s error,%=0
```

### Apple M2 Pro

```sh
$ build_release/reduce_bench
You did not feed the size of arrays, so we will use a 1GB array!
Page size: 16384 bytes
Cache line size: 128 bytes
Dataset size: 268435456 elements
Dataset alignment: 128 bytes
Dataset allocation type: malloc
Dataset NUMA nodes: 1

Running build_release/reduce_bench
Run on (12 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x12)
Load Average: 2.85, 2.81, 3.73
------------------------------------------------------------------------------------
Benchmark                          Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------
unrolled/f32                30964307 ns     30957398 ns          450 bytes/s=34.6768G/s error,%=50
unrolled/f64                29709300 ns     29570448 ns          469 bytes/s=36.1416G/s error,%=0
std::accumulate/f32        230808586 ns    230802100 ns           60 bytes/s=4.65209G/s error,%=93.75
std::accumulate/f64        230730119 ns    230729517 ns           60 bytes/s=4.65367G/s error,%=0
serial/f32/av::fork_union    9916316 ns      9401053 ns         1394 bytes/s=108.28G/s error,%=745.058n
serial/f64/av::fork_union    9681207 ns      9152610 ns         1450 bytes/s=110.91G/s error,%=0
serial/f32/openmp           25585366 ns     21820168 ns          518 bytes/s=41.967G/s error,%=25
neon/f32                    58821387 ns     58819675 ns          234 bytes/s=18.2543G/s error,%=75
neon/f32/av::fork_union     10396624 ns      9999804 ns         1365 bytes/s=103.278G/s error,%=8.9407u
neon/f32/std::threads        9973662 ns      6937423 ns         1370 bytes/s=107.658G/s error,%=8.33334
neon/f32/openmp             10295897 ns      6866715 ns         1367 bytes/s=104.288G/s error,%=8.9407u
```
