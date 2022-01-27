#pragma once
#include <execution>   // `std::execution::par_unseq`
#include <immintrin.h> // AVX2 intrinsics
#include <numeric>     // `std::accumulate`
#include <thread>      // `std::thread`
#include <omp.h>       // `#pragma omp`

namespace av {

template <typename accumulator_at = float> struct cpu_baseline_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    accumulator_at operator()() const noexcept { return std::accumulate(begin_, end_, accumulator_at(0)); }
};

template <typename accumulator_at = float> struct cpu_par_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    accumulator_at operator()() const noexcept {
        return std::reduce(std::execution::par, begin_, end_, accumulator_at(0));
    }
};

template <typename accumulator_at = float> struct cpu_par_unseq_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    accumulator_at operator()() const noexcept {
        return std::reduce(std::execution::par_unseq, begin_, end_, accumulator_at(0));
    }
};

/**
 * @brief Single-threaded, but SIMD parallel reductions,
 * that accumulate 256 bits worth of data on every logic step.
 *
 * Links:
 * https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX,AVX2&text=add_ps
 * https://stackoverflow.com/a/23190168
 */
struct cpu_avx2_f32_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    static inline float _mm256_reduce_add_ps(__m256 x) noexcept {
        auto x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
        auto x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        auto x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        return _mm_cvtss_f32(x32);
    }

    float operator()() const noexcept {
        auto it = begin_;

        // SIMD-parallel summation stage
        auto running_sums = _mm256_set1_ps(0);
        for (; it + 8 < end_; it += 8)
            running_sums = _mm256_add_ps(_mm256_loadu_ps(it), running_sums);

        // Serial summation
        auto running_sum = _mm256_reduce_add_ps(running_sums);
        for (; it != end_; ++it)
            running_sum += *it;

        return running_sum;
    }
};

/**
 * @brief Improvement over `cpu_avx2_f32_t`, that uses Kahan
 * stable summation algorithm to compensate floating point
 * error.
 * https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 */
struct cpu_avx2_kahan_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    float operator()() const noexcept {
        auto it = begin_;

        // SIMD-parallel summation stage
        auto running_sums = _mm256_set1_ps(0);
        auto compensations = _mm256_set1_ps(0);
        auto t = _mm256_set1_ps(0);
        auto y = _mm256_set1_ps(0);
        for (; it + 8 < end_; it += 8) {
            y = _mm256_sub_ps(_mm256_loadu_ps(it), compensations);
            t = _mm256_add_ps(running_sums, y);
            compensations = _mm256_sub_ps(_mm256_sub_ps(t, running_sums), y);
            running_sums = t;
        }

        // Serial summation
        auto running_sum = cpu_avx2_f32_t::_mm256_reduce_add_ps(running_sums);
        for (; it != end_; ++it)
            running_sum += *it;

        return running_sum;
    }
};

struct cpu_avx2_f64_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    double operator()() const noexcept {
        auto it = begin_;

        // SIMD-parallel summation stage
        auto running_sums = _mm256_set1_pd(0);
        for (; it + 4 < end_; it += 4)
            running_sums = _mm256_add_pd(_mm256_cvtps_pd(_mm_loadu_ps(it)), running_sums);

        // Serial summation
        running_sums = _mm256_hadd_pd(running_sums, running_sums);
        running_sums = _mm256_hadd_pd(running_sums, running_sums);
        auto running_sum = _mm256_cvtsd_f64(running_sums);
        for (; it != end_; ++it)
            running_sum += *it;

        return running_sum;
    }
};

struct cpu_avx2_f64_by32_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    double operator()() const noexcept {
        auto it = begin_;
        size_t const threads_cnt = 32;
        size_t step = (end_ - begin_) / threads_cnt;

        // SIMD-parallel summation stage
        std::vector<std::thread> threads;
        std::vector<double> running_sums(threads_cnt, 0.0);
        for (size_t i = 0; i < threads_cnt; ++i) {
            auto cpu_avx2_f64_sum = [](auto begin, size_t step, double& result) {
                cpu_avx2_f64_t cpu_avx2_f64 = {begin, begin + step};
                result = cpu_avx2_f64();
            };
            threads.push_back(std::thread(cpu_avx2_f64_sum, it, step, std::ref(running_sums[i])));
            it += step;
        }

        double running_sum = 0;
        for (size_t i = 0; i < threads_cnt; ++i) {
            threads[i].join();
            running_sum += running_sums[i];
        }

        for (; it != end_; ++it)
            running_sum += *it;

        return running_sum;
    }
};

/**
 * @brief OpenMP on-CPU multi-core reductions acceleration.
 * https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-reduction.html
 */
struct cpu_openmp_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    float operator()() const noexcept {
        float sum = 0;
        size_t const n = end_ - begin_;
#pragma omp parallel for default(shared) reduction(+ : sum)
        for (size_t i = 0; i != n; i++)
            sum += begin_[i];
        return sum;
    }
};

} // namespace av