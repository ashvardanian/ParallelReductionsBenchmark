#pragma once
#include <execution>             // `std::execution::par_unseq`
#include <immintrin.h>           // AVX2 intrinsics
#include <numeric>               // `std::accumulate`
#include <omp.h>                 // `#pragma omp`
#include <taskflow/taskflow.hpp> // Reusable thread-pools
#include <thread>                // `std::thread`

namespace unum {

inline size_t total_cores() { return std::thread::hardware_concurrency() / 2; }

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
 * that accumulate 256 bits worth of data on every logic thread.
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

struct cpu_avx2_f64threads_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;
    std::vector<std::thread> threads_;
    std::vector<double> sums_;

    cpu_avx2_f64threads_t(float const *b, float const *e) : begin_(b), end_(e) {
        threads_.reserve(total_cores());
        sums_.resize(total_cores());
    }

    struct thread_task_t {
        float const *const begin_;
        float const *const end_;
        double &output_;
        inline void operator()() const noexcept { output_ = cpu_avx2_f64_t{begin_, end_}(); }
    };

    double operator()() {
        auto it = begin_;
        size_t const count_per_thread = (end_ - begin_) / sums_.size();

        // Start the child threads
        for (size_t i = 0; i + 1 < sums_.size(); ++i, it += count_per_thread)
            threads_.emplace_back(thread_task_t{it, it + count_per_thread, sums_[i]});

        // This thread lives by its own rules :)
        double running_sum = 0;
        thread_task_t{it, it + count_per_thread, running_sum}();

        // Accumulate sums from child threads.
        for (size_t i = 1; i < sums_.size(); ++i) {
            threads_[i - 1].join();
            running_sum += sums_[i];
        }

        threads_.clear();
        return running_sum;
    }
};

/**
 * @brief A logical improvement over other parallel CPU implementation,
 * as it explicitly reuses the threads instead of creating new ones,
 * but doesn't improve performance in practice.
 *
 * https://taskflow.github.io/taskflow/ParallelIterations.html
 */
struct cpu_avx2_f64threadpool_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;
    tf::Executor executor_;
    tf::Taskflow taskflow_;
    std::vector<double> sums_;

    cpu_avx2_f64threadpool_t(float const *b, float const *e)
        : begin_(b), end_(e), executor_(total_cores()), sums_(total_cores()) {
        size_t const count_total = end_ - begin_;
        size_t const count_per_thread = count_total / sums_.size();
        taskflow_.for_each_index(0ul, sums_.size(), 1ul, [&](size_t i) {
            auto b = begin_ + i * count_per_thread;
            sums_[i] = cpu_avx2_f64_t{b, b + count_per_thread}();
        });
    }

    double operator()() {

        executor_.run(taskflow_).wait();

        double running_sum = 0;
        for (double &s : sums_)
            running_sum += s;
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

    cpu_openmp_t(float const *b, float const *e) : begin_(b), end_(e) {
        omp_set_dynamic(0);
        omp_set_num_threads(total_cores());
    }

    float operator()() const noexcept {
        float sum = 0;
        size_t const n = end_ - begin_;
#pragma omp parallel for default(shared) reduction(+ : sum)
        for (size_t i = 0; i != n; i++)
            sum += begin_[i];
        return sum;
    }
};

} // namespace unum