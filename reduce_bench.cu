#include <vector>

#include <benchmark/benchmark.h>
#include <fmt/core.h>

#include "reduce_cpu.hpp"
#include "reduce_cuda.hpp"
#include "reduce_opencl.hpp"

using namespace av;
namespace bm = benchmark;
std::vector<float> dataset;

template <typename accumulator_at> void generic(bm::State &state, accumulator_at &&accumulator) {
    for (auto _ : state)
        bm::DoNotOptimize(accumulator());

    state.counters["bytes/s"] = bm::Counter(state.iterations() * dataset.size() * sizeof(float), bm::Counter::kIsRate);
}

template <typename accumulator_at> void automatic(bm::State &state) {
    accumulator_at acc{dataset.data(), dataset.data() + dataset.size()};
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
    std::fill(dataset.begin(), dataset.end(), float(0.5));

    // Log available backends.
    auto ocl_targets = opencl_targets();
    for (auto const &tgt : ocl_targets)
        fmt::print("- OpenCL: {} ({}), {}, {}\n", tgt.device_name, tgt.device_version, tgt.driver_version,
                   tgt.language_version);

    if (cuda_device_count()) {
        bm::RegisterBenchmark("cuda_thrust", &automatic<cuda_thrust_t>)->MinTime(10);
        bm::RegisterBenchmark("cuda_blocks", &automatic<cuda_gt<cuda_kernel_t::blocks_k>>)->MinTime(10);
        bm::RegisterBenchmark("cuda_warps", &automatic<cuda_gt<cuda_kernel_t::warps_k>>)->MinTime(10);
    }
    else
        fmt::print("No CUDA capable devices found!\n");

    // Register and run all the benchmarks.
    bm::RegisterBenchmark("cpu_baseline", &automatic<cpu_baseline_t>)->MinTime(10);
    bm::RegisterBenchmark("cpu_avx2", &automatic<cpu_avx2_t>)->MinTime(10);
    bm::RegisterBenchmark("cpu_openmp", &automatic<cpu_openmp_t>)->MinTime(10);

    std::vector<size_t> group_sizes = {8, 32, 128};
    for (auto tgt : ocl_targets) {
        for (auto kernel_name : opencl_t::kernels_k) {
            for (auto group_size : group_sizes) {
                auto name = fmt::format("opencl-{} on {} split by {}", kernel_name, tgt.device_name, group_size);
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