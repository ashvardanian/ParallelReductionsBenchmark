#include <vector>

#include <benchmark/benchmark.h>
#include <fmt/core.h>

#include "reduce_cpu.hpp"
#include "reduce_cuda.hpp"
#include "reduce_opencl.hpp"

using namespace av;
namespace bm = benchmark;
static std::vector<float> dataset;

template <typename accumulator_at> void generic(bm::State &state, accumulator_at&& accumulator) {
    double const sum_expected = dataset.size() * 1.0;
    double sum = 0;
    double error = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(sum = accumulator());
        error = std::abs(sum_expected - sum)/sum_expected;
    }

    auto total_ops = state.iterations() * dataset.size();
    state.counters["adds/s"] = bm::Counter(total_ops, bm::Counter::kIsRate);
    state.counters["bytes/s"] = bm::Counter(total_ops * sizeof(float), bm::Counter::kIsRate);
    state.counters["error,%"] = bm::Counter(error * 100);
}

template <typename accumulator_at> 
void automatic(bm::State &state) {
    std::fill(dataset.begin(), dataset.end(), 1.f);
    accumulator_at acc {dataset.data(), dataset.data() + dataset.size()};
    generic(state, acc);
}

int main(int argc, char **argv) {

    // Parse configuration parameters.
    size_t elements = 0;
    if (argc <= 1) {
        fmt::print("You did not feed the size of arrays, so we will use a 1GB array!\n");
        elements = 1024 * 1024 * 1024 / sizeof(float);
    } else {
        elements = static_cast<size_t>(std::atol(argv[1]));
    }
    dataset.resize(elements);
    std::fill(dataset.begin(), dataset.end(), 1.f);

    // Register and run all the benchmarks.
    bm::RegisterBenchmark("cpu_baseline:f32", &automatic<cpu_baseline_gt<float>>)->MinTime(10);
    bm::RegisterBenchmark("cpu_baseline:f64", &automatic<cpu_baseline_gt<double>>)->MinTime(10);
    bm::RegisterBenchmark("cpu_avx2:f32", &automatic<cpu_avx2_f32_t>)->MinTime(10);
    bm::RegisterBenchmark("cpu_avx2:f32kahan", &automatic<cpu_avx2_kahan_t>)->MinTime(10);
    bm::RegisterBenchmark("cpu_avx2:f64", &automatic<cpu_avx2_f64_t>)->MinTime(10);
    bm::RegisterBenchmark("cpu_openmp", &automatic<cpu_openmp_t>)->MinTime(10);

    // Log available backends.
    auto ocl_targets = opencl_targets();
    for (auto const &tgt : ocl_targets)
        fmt::print("- OpenCL: {} ({}), {}, {}\n", tgt.device_name, tgt.device_version, tgt.driver_version,
                   tgt.language_version);

    if (cuda_device_count()) {
        bm::RegisterBenchmark("cuda_thrust", &automatic<cuda_thrust_t>)->MinTime(10);
        bm::RegisterBenchmark("cuda_tensors", &automatic<cuda_tensors_t>)->MinTime(10);
        bm::RegisterBenchmark("cuda_warps", &automatic<cuda_warps_t>)->MinTime(10);
        // bm::RegisterBenchmark("cuda_blocks", &automatic<cuda_blocks_t>)->MinTime(10);
    }
    else
        fmt::print("No CUDA capable devices found!\n");

    
    for (auto tgt : ocl_targets) {
        for (auto kernel_name : opencl_t::kernels_k) {
            for (auto group_size : opencl_wg_sizes) {
                auto name = fmt::format("opencl-{} split by {} on {}", kernel_name, group_size, tgt.device_name);
                bm::RegisterBenchmark(name.c_str(), [=](bm::State &state) {
                    opencl_t ocl(dataset.data(), dataset.data() + dataset.size(), tgt, group_size, kernel_name);
                    generic(state, ocl);
                })->MinTime(10);
            }
        }
    }


    bm::Initialize(&argc, argv);
    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}