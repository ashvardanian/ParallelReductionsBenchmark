#include <cstdlib> // Accessing environment variables
#include <new>     // `std::launder`

#include <benchmark/benchmark.h>
#include <fmt/core.h>

#include "reduce_cpu.hpp"

#if defined(__OPENCL__)
#include "reduce_opencl.hpp"
#endif

#if defined(__CUDACC__)
#include "reduce_cuda.hpp"
#endif

using namespace ashvardanian::reduce;

namespace bm = benchmark;
static volatile float *dataset_begin = nullptr;
static volatile float *dataset_end = nullptr;

template <typename accumulator_at> void generic(bm::State &state, accumulator_at &&accumulator) {
    size_t const dataset_size = dataset_end - dataset_begin;
    double const sum_expected = dataset_size * 1.0;
    double sum = 0;
    for (auto _ : state) {
        sum = accumulator();
        bm::DoNotOptimize(sum);
    }

    if (state.thread_index() == 0) {
        auto error = std::abs(sum_expected - sum) / sum_expected;
        auto total_ops = state.iterations() * dataset_size;
        state.counters["bytes/s"] = bm::Counter(total_ops * sizeof(float), bm::Counter::kIsRate);
        state.counters["error,%"] = bm::Counter(error * 100);
    }
}

template <typename accumulator_at> void make(bm::State &state) {
    accumulator_at acc{(float *)(dataset_begin), (float *)(dataset_end)};
    generic(state, acc);
}

template <typename at> std::unique_ptr<at[]> alloc_aligned(size_t alignment, size_t length) {
    at *raw = 0;
    int error = posix_memalign((void **)&raw, alignment, sizeof(at) * length);
    (void)error;
    return std::unique_ptr<at[]>{raw};
}

int main(int argc, char **argv) {

    // Parse configuration parameters.
    size_t elements = 0;
    char const *elements_env_variable = std::getenv("PARALLEL_REDUCTIONS_LENGTH");

    if (elements_env_variable) {
        elements = static_cast<size_t>(std::atol(elements_env_variable));
        if (elements == 0) {
            fmt::print("Inappropriate `PARALLEL_REDUCTIONS_LENGTH` value!\n");
            return 1;
        }
    } else {
        fmt::print("You did not feed the size of arrays, so we will use a 1GB array!\n");
        elements = 1024ull * 1024ull * 1024ull / sizeof(float);
    }

    auto dataset = alloc_aligned<float>(64, elements);
    dataset_begin = dataset.get();
    dataset_end = dataset_begin + elements;
    std::fill(dataset_begin, dataset_end, 1.f);

    // Log available backends
#if defined(__OPENCL__)
    auto ocl_targets = opencl_targets();
    for (auto const &tgt : ocl_targets)
        fmt::print("- OpenCL: {} ({}), {}, {}\n", tgt.device_name, tgt.device_version, tgt.driver_version,
                   tgt.language_version);
#endif

    // Memset is only useful as a baseline, but running it will corrupt our buffer
    // bm::RegisterBenchmark("memset", &make<memset_t>)->MinTime(10)->UseRealTime();
    // bm::RegisterBenchmark("memset@threads", &make<threads_gt<memset_t>>)->MinTime(10)->UseRealTime();

    // Generic CPU benchmarks
    bm::RegisterBenchmark("unrolled<f32>", &make<unrolled_gt<float>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("unrolled<f64>", &make<unrolled_gt<double>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("std::accumulate<f32>", &make<stl_accumulate_gt<float>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("std::accumulate<f64>", &make<stl_accumulate_gt<double>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("std::reduce<par, f32>", &make<stl_par_reduce_gt<float>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("std::reduce<par, f64>", &make<stl_par_reduce_gt<double>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("std::reduce<par_unseq, f32>", &make<stl_parunseq_reduce_gt<float>>)
        ->MinTime(10)
        ->UseRealTime();
    bm::RegisterBenchmark("std::reduce<par_unseq, f64>", &make<stl_parunseq_reduce_gt<double>>)
        ->MinTime(10)
        ->UseRealTime();
    bm::RegisterBenchmark("openmp<f32>", &make<openmp_t>)->MinTime(10)->UseRealTime();

    // x86
#if defined(__AVX2__)
    bm::RegisterBenchmark("avx2<f32>", &make<avx2_f32_t>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("avx2<f32kahan>", &make<avx2_f32kahan_t>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("avx2<f64>", &make<avx2_f64_t>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("avx2<f32aligned>@threads", &make<threads_gt<avx2_f32aligned_t>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("avx2<f64>@threads", &make<threads_gt<avx2_f64_t>>)->MinTime(10)->UseRealTime();
    bm::RegisterBenchmark("sse<f32aligned>@threads", &make<threads_gt<sse_f32aligned_t>>)->MinTime(10)->UseRealTime();
#endif

// CUDA
#if defined(__CUDACC__)
    if (cuda_device_count()) {
        bm::RegisterBenchmark("cub@cuda", &make<cuda_cub_t>)->MinTime(10)->UseRealTime();
        bm::RegisterBenchmark("warps@cuda", &make<cuda_warps_t>)->MinTime(10)->UseRealTime();
        bm::RegisterBenchmark("thrust@cuda", &make<cuda_thrust_t>)->MinTime(10)->UseRealTime();
    } else
        fmt::print("No CUDA capable devices found!\n");
#endif

        // OpenCL
#if defined(__OPENCL__)
    for (auto tgt : ocl_targets) {
        for (auto kernel_name : opencl_t::kernels_k) {
            for (auto group_size : opencl_wg_sizes) {
                auto name = fmt::format("opencl-{} split by {} on {}", kernel_name, group_size, tgt.device_name);
                bm::RegisterBenchmark(name.c_str(),
                                      [=](bm::State &state) {
                                          opencl_t ocl(dataset.data(), dataset.data() + dataset.size(), tgt, group_size,
                                                       kernel_name);
                                          generic(state, ocl);
                                      })
                    ->MinTime(10)
                    ->UseRealTime();
            }
        }
    }
#endif

    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv))
        return 1;

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}