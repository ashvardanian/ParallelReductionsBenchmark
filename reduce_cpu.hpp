#include <immintrin.h> // AVX2 intrinsics
#include <numeric>     // `std::accumulate`

namespace av {

struct cpu_baseline_t {
    float const *begin = nullptr;
    float const *end = nullptr;
    float operator()() const noexcept { return std::accumulate(begin, end, float(0)); }
};

/**
 * @brief Single-threaded, but SIMD parallel reductions,
 * that accumulate 256 bits worth of data on every logic step.
 *
 * Links:
 * https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX,AVX2&text=add_ps
 * https://stackoverflow.com/a/23190168
 */
struct cpu_avx2_t {

    float const *begin = nullptr;
    float const *end = nullptr;

    static inline float _mm256_reduce_add_ps(__m256 x) noexcept {
        auto x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
        auto x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        auto x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        return _mm_cvtss_f32(x32);
    }

    float operator()() const noexcept {
        auto it = begin;

        // SIMD-parallel summation stage
        auto running_sums = _mm256_set1_ps(0);
        for (; it + 8 < end; it += 8)
            running_sums = _mm256_add_ps(_mm256_load_ps(it), running_sums);

        // Serial summation
        auto running_sum = _mm256_reduce_add_ps(running_sums);
        for (; it != end; ++it)
            running_sum += *it;

        return running_sum;
    }
};

/**
 * @brief OpenMP on-CPU multi-core reductions acceleration.
 * https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-reduction.html
 */
struct cpu_openmp_t {

    float const *begin = nullptr;
    float const *end = nullptr;

    float operator()() const noexcept {
        float sum = 0;
        size_t const n = end - begin;
        auto const ptr = begin;
#pragma omp parallel for reduction(+ : sum)
        for (size_t i = 0; i < n; i++)
            sum = sum + ptr[i];
        return sum;
    }
};

} // namespace av