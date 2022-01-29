#include <vector>

#include <benchmark/benchmark.h>
#include <fmt/core.h>

#include "reduce_cpu.hpp"
#include "reduce_cuda.hpp"
#include "reduce_opencl.hpp"

using namespace unum;
namespace bm = benchmark;
static float *dataset_begin = nullptr;
static float *dataset_end = nullptr;

template <typename accumulator_at> void generic(bm::State &state, accumulator_at &&accumulator) {
    size_t const dataset_size = dataset_end - dataset_begin;
    double const sum_expected = dataset_size * 1.0;
    double sum = 0;
    double error = 0;
    for (auto _ : state) {
        sum = accumulator();
        bm::DoNotOptimize(sum);
        error = std::abs(sum_expected - sum) / sum_expected;
    }

    if (state.thread_index() == 0) {
        auto total_ops = state.iterations() * dataset_size;
        state.counters["bytes/s"] = bm::Counter(total_ops * sizeof(float), bm::Counter::kIsRate);
        state.counters["error,%"] = bm::Counter(error * 100);
    }
}

template <typename accumulator_at> void automatic(bm::State &state) {
    accumulator_at acc{dataset_begin, dataset_end};
    generic(state, acc);
}

int main(int argc, char **argv) {

    // Parse configuration parameters.
    size_t elements = 0;
    if (argc <= 1) {
        fmt::print("You did not feed the size of arrays, so we will use a 1GB array!\n");
        elements = 1024ull * 1024ull * 1024ull / sizeof(float);
    } else {
        elements = static_cast<size_t>(std::atol(argv[1]));
    }
    std::vector<__m256> dataset;
    dataset.resize(elements / 8);
    dataset_begin = reinterpret_cast<float *>(dataset.data());
    dataset_end = dataset_begin + elements;
    std::fill(dataset_begin, dataset_end, 1.f);

    // Log available backends
    auto ocl_targets = opencl_targets();
    for (auto const &tgt : ocl_targets)
        fmt::print("- OpenCL: {} ({}), {}, {}\n", tgt.device_name, tgt.device_version, tgt.driver_version,
                   tgt.language_version);

    bm::RegisterBenchmark("memcpy", &automatic<memcpy_t>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("memcpy@threadpool", &automatic<threadpool_memcpy_t>)->MinTime(10)->UseRealTime();

    // Generic CPU benchmarks
    bm::RegisterBenchmark("std::accumulate<f32>", &automatic<stl_accumulate_gt<float>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("std::accumulate<f64>", &automatic<stl_accumulate_gt<double>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("std::reduce<par, f32>", &automatic<stl_par_reduce_gt<float>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("std::reduce<par, f64>", &automatic<stl_par_reduce_gt<double>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("std::reduce<par_unseq, f32>", &automatic<stl_parunseq_reduce_gt<float>>)
        ->MinTime(10)
        ->UseRealTime();
    bm::RegisterBenchmark("std::reduce<par_unseq, f64>", &automatic<stl_parunseq_reduce_gt<double>>)
        ->MinTime(10)
        ->UseRealTime();
    bm::RegisterBenchmark("openmp<f32>", &automatic<openmp_t>)->MinTime(10)->UseRealTime();

    // x86
    bm::RegisterBenchmark("avx2<f32>", &automatic<avx2_f32_t>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("avx2<f32kahan>", &automatic<avx2_f32kahan_t>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("avx2<f64>", &automatic<avx2_f64_t>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("avx2<f32aligned>@threadpool", &automatic<threadpool_gt<avx2_f32aligned_t>>)
        ->MinTime(10)
        ->UseRealTime();
    bm::RegisterBenchmark("avx2<f64>@threadpool", &automatic<threadpool_gt<avx2_f64_t>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("sse<f32aligned>@threadpool", &automatic<threadpool_gt<sse_f32aligned_t>>)
        ->MinTime(10)
        ->UseRealTime();

    // CUDA
    if (cuda_device_count()) {
        bm::RegisterBenchmark("cub@cuda", &automatic<cuda_cub_t>)->MinTime(10)->UseRealTime();
        bm::RegisterBenchmark("warps@cuda", &automatic<cuda_warps_t>)->MinTime(10)->UseRealTime();
        bm::RegisterBenchmark("thrust@cuda", &automatic<cuda_thrust_t>)->MinTime(10)->UseRealTime();
    } else
        fmt::print("No CUDA capable devices found!\n");

    // OpenCL
    // for (auto tgt : ocl_targets) {
    //     for (auto kernel_name : opencl_t::kernels_k) {
    //         for (auto group_size : opencl_wg_sizes) {
    //             auto name = fmt::format("opencl-{} split by {} on {}", kernel_name, group_size, tgt.device_name);
    //             bm::RegisterBenchmark(name.c_str(),
    //                                   [=](bm::State &state) {
    //                                       opencl_t ocl(dataset.data(), dataset.data() + dataset.size(), tgt,
    //                                       group_size,
    //                                       kernel_name);
    //                                       generic(state, ocl);
    //                                   })->MinTime(10)->UseRealTime();
    //         }
    //     }
    // }

    bm::Initialize(&argc, argv);
    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}