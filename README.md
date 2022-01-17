# Parallel GPU Reductions, from Unum with ❤️

It contains various educational examples and benchmarks of GPU backends.
The older versions also included data-parallel operations and dense matrix multiplications (GEMM), but now it's just fast reductions.
Aside from basline `std::accumulate` it compares:

* [AVX2] manually written single-threaded, but SIMD-parallel code.
* [OpenMP]() `reduction` clause.
* [Thrust]() with it's `thrust::reduce`.
* [CUDA]() kernels with block-reductions, warp-reductions and even `wmma::` Tensor Core extensions.
* [OpenCL](tree/master/Shared/OpenCL) kernels, eight of them.
* [Vulkan](tree/master/Shared/Vulkan) queues for SPIR-V kernels.
* [SyCL](tree/master/Shared/TriSYCL).

Previously it also compared [ArrayFire] and [Halide].
Examples were collected from early 2010s until 2019, and later updated in 2022.

* [Lecture Slides](blob/master/Presentation.pdf) from 2019.
* [CppRussia Talk](https://youtu.be/AA4RI6o0h1U) in Russia in 2019.
* [JetBrains Talk](https://youtu.be/BUtHOftDm_Y) in Germany & Russia in 2019.
* [C++ Armenia Talk]() in 2022.

## Build & Run on 1 Line

Following script will, by default, generate a 1GB array of numbers, and reduce them using every available backend.
All the classical Google Benchmark arguments are supported, including `--benchmark_filter=opencl`.
All the needed library dependencies will be automatically fetched.

```sh
mkdir -p release && cd release && cmake .. && make && ./reduce_bench && cd ..
```

To debug or introspect, procedure is similar:

```sh
mkdir -p debug && cd debug && cmake -DCMAKE_BUILD_TYPE=Debug .. && make && cd ..
```

And then run your favorite debugger.
