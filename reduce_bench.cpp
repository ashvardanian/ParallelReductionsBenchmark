/**
 *  @date 04/09/2019
 *  @file reduce_bench.cpp
 *  @brief Benchmarking parallel reductions
 *  @author Ash Vardanian
 */
#include <cstdlib>   // `std::getenv`
#include <cstring>   // `std::memset`
#include <memory>    // `std::uniue_ptr`
#include <new>       // `std::launder`
#include <stdexcept> // `std::bad_alloc`

/**
 *  Platform-specific includes for memory allocation and binding.
 *  On Linux we use `mmap` and `numa_*` functions to allocate memory and bind it to NUMA nodes.
 *  On Apple we use `sysctlbyname` to query the cache line size and page size.
 *  On Windows we use `GetSystemInfo` and `VirtualAlloc`.
 */
#if defined(__linux__)
#include <sys/mman.h> // `mmap`
#include <unistd.h>   // `sysconf`
#if defined(LIBNUMA)
#include <numa.h> // `numa_available`, `numa_alloc_onnode`, `numa_free`
#endif
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h> // `sysctlbyname`
#include <unistd.h>     // `sysconf`
#endif

#if defined(_WIN32)
#include <windows.h> // `GetSystemInfo`, `VirtualAlloc`
#endif

#include <benchmark/benchmark.h>
#include <fmt/core.h>

/**
 *  Platform-specific includes:
 *  - CPU kernels with AVX2, AVX-512, and OpenMP acceleration
 *  - CUDA kernels with CUB, Thrust, and manual implementations
 *  - OpenCL kernels with manual implementations
 *  - Dysfunctional Metal kernels for Apple devices
 */
#include "reduce_cpu.hpp"

#if defined(__OPENCL__)
#include "reduce_opencl.hpp"
#endif

#if defined(__CUDACC__)
#include "reduce_cuda.cuh"
#endif

#if defined(__APPLE__) && 0 // TODO: Fix compilation
#include "reduce_metal.h"
#endif

namespace bm = benchmark;
using namespace ashvardanian::reduce;

/**
 *  @brief  Wraps the memory allocated for the benchmark either from `malloc` or `mmap`.
 *          Their deallocation mechanisms differ, so we need to keep track of the type.
 */
struct dataset_t {
    float *begin_ = nullptr;
    float *end_ = nullptr;
    enum class type_t { unknown, malloc, mmap } type_ = type_t::unknown;

    dataset_t() noexcept = default;
    dataset_t(dataset_t const &) = delete;
    dataset_t(float *b, float *e, type_t t) noexcept : begin_(b), end_(e), type_(t) {}

    std::size_t size() const noexcept { return end_ - begin_; }
    float *data() const noexcept { return begin_; }

    ~dataset_t() noexcept {
        switch (type_) {
        case type_t::malloc: std::free(begin_); break;
        case type_t::mmap: munmap(begin_, size() * sizeof(float)); break;
        default: break;
        }
        begin_ = nullptr;
        end_ = nullptr;
        type_ = type_t::unknown;
    }
};

/**
 *  @brief  Runs the main loop of the benchmark, reporting the bandwidth and @b error,
 *          that is not typical in
 */
template <typename accumulator_, typename... accumulator_args_>
void run(bm::State &state, dataset_t const &dataset, accumulator_args_ &&...args) {

    std::size_t const n = dataset.size();
    double const sum_expected = n * 1.0;
    double sum = 0;

    accumulator_ accumulator(dataset.data(), dataset.data() + n, std::forward<accumulator_args_>(args)...);
    for (auto _ : state) bm::DoNotOptimize(sum = accumulator());

    // Only log stats from the main thread
    if (state.thread_index() != 0) return;
    auto error = std::abs(sum_expected - sum) / sum_expected;
    auto total_ops = state.iterations() * n;
    state.counters["bytes/s"] = bm::Counter(total_ops * sizeof(float), bm::Counter::kIsRate);
    state.counters["error,%"] = bm::Counter(error * 100);
    state.SetComplexityN(n);
}

template <typename accumulator_, typename... accumulator_args_>
auto register_(std::string const &name, accumulator_ &&, dataset_t const &data, accumulator_args_ &&...args) {
    using accumulator = std::decay_t<accumulator_>;
    return bm::RegisterBenchmark(
               name, [&](bm::State &s) { run<accumulator>(s, data, std::forward<accumulator_args_>(args)...); })
        ->MinTime(10)
        ->UseRealTime();
}

/**
 *  @brief Detects and returns the cache-line alignment in bytes.
 *  @return The cache-line size in bytes, or a default of 64 or 128 bytes.
 */
std::size_t alignment_cache_line() {

#if defined(__linux__)
    // Some distributions define `_SC_LEVEL1_DCACHE_LINESIZE`;
    // if not, we can define it ourselves or rely on a fallback.
#if defined(_SC_LEVEL1_DCACHE_LINESIZE)
    long sysconf_res = ::sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (sysconf_res > 0) return static_cast<std::size_t>(sysconf_res);
#endif // defined(_SC_LEVEL1_DCACHE_LINESIZE)
    return 64;

#elif defined(__APPLE__)
    // On macOS, query the cache line size via `sysctl`
    std::size_t sysctl_result = 0;
    std::size_t sysctl_result_len = sizeof(sysctl_result);
    if (sysctlbyname("hw.cachelinesize", &sysctl_result, &sysctl_result_len, nullptr, 0) == 0) return sysctl_result;
    return 128;
#else // Windows, FreeBSD, or other
    return 64;
#endif
}

/**
 *  @brief Detects and returns the system page size for RAM.
 *  @return The page size in bytes, or a default of 4096 bytes.
 */
std::size_t alignment_ram_page() {
#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
    long sysconf_res = ::sysconf(_SC_PAGESIZE);
    if (sysconf_res > 0) return static_cast<std::size_t>(sysconf_res);
    return 4096;
#elif defined(_WIN32)
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    return static_cast<std::size_t>(system_info.dwPageSize);
#else
    return 4096;
#endif
}

/**
 *  @brief Allocates a dataset of floats using various strategies.
 *
 *  On Linux, this function attempts the following approaches:
 *  1. `mmap` with huge pages if supported @b (MAP_HUGETLB).
 *  2. `std::aligned_alloc` aligned to the system page size with optional @b `madvise(MADV_HUGEPAGE)`.
 *  If NUMA is available (libNUMA on Linux), memory is distributed across NUMA nodes.
 *
 *  @param elements Number of float elements to allocate.
 *  @return dataset_t A dataset wrapper holding the pointer and type of allocation.
 *  @throws std::bad_alloc if allocation fails.
 */
dataset_t make_dataset(                           //
    std::size_t needed_elements,                  //
    [[maybe_unused]] std::size_t alignment_cache, //
    [[maybe_unused]] std::size_t alignment_page) {

    std::size_t needed_bytes = needed_elements * sizeof(float);
    dataset_t::type_t allocation_type = dataset_t::type_t::unknown;
    float *memory = nullptr;

#if defined(__linux__)
    // Try to allocate with mmap + huge pages
    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
#if defined(MAP_HUGETLB)
    mmap_flags |= MAP_HUGETLB;
#endif
    void *mmap_memory = ::mmap(nullptr, needed_bytes, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);

    if (mmap_memory != MAP_FAILED) {
        memory = reinterpret_cast<float *>(mmap_memory);
        allocation_type = dataset_t::type_t::mmap;
    }
    else {
        // Fallback to `std::aligned_alloc` with RAM page alignment.
        // It requires the size to be a multiple of alignment.
        std::size_t aligned_size = round_up_to_multiple(needed_bytes, alignment_page);
        memory = static_cast<float *>(std::aligned_alloc(alignment_page, aligned_size));
        if (!memory) throw std::bad_alloc();
        allocation_type = dataset_t::type_t::malloc;

#if defined(MADV_HUGEPAGE)
        // Suggest transparent huge pages
        ::madvise(memory, needed_bytes, MADV_HUGEPAGE);
#endif
    }

    // If `libnuma` is available, bind memory across NUMA nodes
#if defined(LIBNUMA)
    if (numa_available() != -1) {
        int num_nodes = numa_num_configured_nodes();
        if (num_nodes <= 1) {
            std::size_t chunk_size = needed_elements / num_nodes;
            for (int i = 0; i < num_nodes; ++i) {
                float *chunk_start = memory + i * chunk_size;
                std::size_t chunk_elems =
                    (i == num_nodes - 1) ? (needed_elements - (chunk_size * (num_nodes - 1))) : (chunk_size);
                numa_tonode_memory(chunk_start, chunk_elems * sizeof(float), i);
            }
        }
    }
#endif // LIBNUMA
#else  // Not Linux:
    std::size_t aligned_size = round_up_to_multiple(needed_bytes, alignment_page);
    memory = static_cast<float *>(std::aligned_alloc(alignment_page, aligned_size));
    if (!memory) throw std::bad_alloc();
    allocation_type = dataset_t::type_t::malloc;
#endif

    // Initialize the allocated memory to zero to make sure it's not a copy-on-write mapping
    std::memset(memory, 0, needed_bytes);
    return dataset_t {memory, memory + needed_elements, allocation_type};
}

/**
 *  @brief Sets all elements in the provided range to the value 1.0f.
 *         Can be used as a synthetic baseline for the throughput of writes.
 *
 *  All other kernels have a similar signature of constructors and the `operator()`.
 */
class memset_t {
    float *const begin_ = nullptr;
    float *const end_ = nullptr;

  public:
    memset_t() = default;
    memset_t(float *b, float *e) noexcept : begin_(b), end_(e) {}

    float operator()() noexcept {
        std::memset(begin_, 1, end_ - begin_);
        return 1;
    }
};

int main(int argc, char **argv) {

    // Parse configuration parameters.
    std::size_t elements = 0;
    char const *elements_env_variable = std::getenv("PARALLEL_REDUCTIONS_LENGTH");

    if (elements_env_variable) {
        elements = static_cast<std::size_t>(std::atol(elements_env_variable));
        if (elements == 0) {
            fmt::print("Inappropriate `PARALLEL_REDUCTIONS_LENGTH` value!\n");
            return 1;
        }
    }
    else {
        fmt::print("You did not feed the size of arrays, so we will use a 1GB array!\n");
        elements = 1024ull * 1024ull * 1024ull / sizeof(float);
    }

    std::size_t const alignment_cache = alignment_cache_line();
    std::size_t const alignment_page = alignment_ram_page();
    fmt::print("Cache line size: {} bytes\n", alignment_cache);
    fmt::print("Page size: {} bytes\n", alignment_page);

    dataset_t dataset = make_dataset(elements, alignment_cache, alignment_page);
    std::fill_n(dataset.data(), dataset.size(), 1.f);

    // Log available backends
#if defined(__OPENCL__)
    auto ocl_targets = opencl_targets();
    for (auto const &tgt : ocl_targets)
        fmt::print( //
            "- OpenCL: {} ({}), {}, {}\n", tgt.device_name, tgt.device_version, tgt.driver_version,
            tgt.language_version);
#endif // defined(__OPENCL__)

    // Memset is only useful as a baseline, but running it will corrupt our buffer
    // register_("memset", memset_t {}, dataset);
    // register_("memset@threads", threads_gt<memset_t> {}, dataset);

    // Generic CPU benchmarks
    register_("unrolled<f32>", unrolled_gt<float> {}, dataset);
    register_("unrolled<f64>", unrolled_gt<double> {}, dataset);
    register_("std::accumulate<f32>", stl_accumulate_gt<float> {}, dataset);
    register_("std::accumulate<f64>", stl_accumulate_gt<double> {}, dataset);
    register_("openmp<f32>", openmp_t {}, dataset);

#if defined(__cpp_lib_execution)
    register_("std::reduce<par, f32>", stl_par_reduce_gt<float> {}, dataset);
    register_("std::reduce<par, f64>", stl_par_reduce_gt<double> {}, dataset);
    register_("std::reduce<par_unseq, f32>", stl_par_unseq_reduce_gt<float> {}, dataset);
    register_("std::reduce<par_unseq, f64>", stl_par_unseq_reduce_gt<double> {}, dataset);
#endif // defined(__cpp_lib_execution)

    // x86 SSE
#if defined(__SSE__)
    register_("sse<f32aligned>@threads", threads_gt<sse_f32aligned_t> {}, dataset);
#endif // defined(__SSE__)

    // x86 AVX2
#if defined(__AVX2__)
    register_("avx2<f32>", avx2_f32_t {}, dataset);
    register_("avx2<f32kahan>", avx2_f32kahan_t {}, dataset);
    register_("avx2<f64>", avx2_f64_t {}, dataset);
    register_("avx2<f32aligned>@threads", threads_gt<avx2_f32aligned_t> {}, dataset);
    register_("avx2<f64>@threads", threads_gt<avx2_f64_t> {}, dataset);
#endif // defined(__AVX2__)

    // x86 AVX-512
#if defined(__AVX512F__)
    register_("avx512<f32streamed>", avx512_f32streamed_t {}, dataset);
    register_("avx512<f32streamed>@threads", threads_gt<avx512_f32streamed_t> {}, dataset);
    register_("avx512<f32unrolled>", avx512_f32unrolled_t {}, dataset);
    register_("avx512<f32unrolled>@threads", threads_gt<avx512_f32unrolled_t> {}, dataset);
    register_("avx512<f32interleaving>", avx512_f32interleaving_t {}, dataset);
    register_("avx512<f32interleaving>@threads", threads_gt<avx512_f32interleaving_t> {}, dataset);
#endif // defined(__AVX512F__)

    // CUDA
#if defined(__CUDACC__)
    if (cuda_device_count()) {
        register_("cub@cuda", cuda_cub_t {}, dataset);
        register_("warps@cuda", cuda_warps_t {}, dataset);
        register_("thrust@cuda", cuda_thrust_t {}, dataset);
    }
    else { fmt::print("No CUDA capable devices found!\n"); }
#endif // defined(__CUDACC__)

    // OpenCL
#if defined(__OPENCL__)
    for (auto tgt : ocl_targets) {
        for (auto kernel_name : opencl_t::kernels_k) {
            for (auto group_size : opencl_wg_sizes) {
                auto name = fmt::format("opencl-{} split by {} on {}", kernel_name, group_size, tgt.device_name);
                register_(name, opencl_t {}, data, tgt, group_size, kernel_name);
            }
        }
    }
#endif // defined(__OPENCL__)

    // Apple's Metal Performance Shaders
#if defined(__APPLE__) && 0
    register_("metal<f32>", metal_t {}, dataset);
#endif // defined(__APPLE__)

    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv)) return 1;

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}