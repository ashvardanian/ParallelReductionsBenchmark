#pragma once
#include <cstring>   // `std::memcpy`
#include <execution> // `std::execution::par_unseq`
#include <numeric>   // `std::accumulate`, `std::reduce`
#include <omp.h>     // `#pragma omp`
#include <thread>    // `std::thread`

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h> // x86 intrinsics
#endif

namespace ashvardanian::reduce {

/// Returns half the number of hardware concurrency units (typically cores).
inline static size_t total_cores() { return std::thread::hardware_concurrency() / 2; }

/// Returns the optimal number of cores for parallel processing (same as total_cores).
inline static size_t optimal_cores() { return total_cores(); }

/// Sets all elements in the provided range to the value 1.0f.
struct memset_t {
    float *const begin_ = nullptr;
    float *const end_ = nullptr;

    memset_t(float *b, float *e) : begin_(b), end_(e) {}
    float operator()() {
        std::memset(begin_, 1, end_ - begin_);
        return 1;
    }
};

/// Computes the sum of a sequence of float values using an unrolled @b `for`-loop.
template <typename accumulator_at = float> struct unrolled_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    accumulator_at operator()() const noexcept {
        accumulator_at sums[8];
        std::fill(sums, sums + 8, 0);
        float const *it = begin_;
        for (; it + 8 < end_; it += 8) {
            sums[0] += it[0];
            sums[1] += it[1];
            sums[2] += it[2];
            sums[3] += it[3];
            sums[4] += it[4];
            sums[5] += it[5];
            sums[6] += it[6];
            sums[7] += it[7];
        }
        for (; it != end_; ++it)
            sums[0] += *it;
        return sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
    }
};

/// Computes the sum of a sequence of float values using @b `std::accumulate`.
template <typename accumulator_at = float> struct stl_accumulate_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    accumulator_at operator()() const noexcept { return std::accumulate(begin_, end_, accumulator_at(0)); }
};

/// Computes the sum of a sequence of float values using parallel std::reduce with execution policy std::execution::par.
template <typename accumulator_at = float> struct stl_par_reduce_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    accumulator_at operator()() const noexcept {
        return std::reduce(std::execution::par, begin_, end_, accumulator_at(0), std::plus<accumulator_at>());
    }
};

/// Computes the sum of a sequence of float values using parallel std::reduce with execution policy
/// std::execution::par_unseq for non-blocking parallelism.
template <typename accumulator_at = float> struct stl_parunseq_reduce_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    accumulator_at operator()() const noexcept {
        return std::reduce(std::execution::par_unseq, begin_, end_, accumulator_at(0), std::plus<accumulator_at>());
    }
};

/// Computes the sum of a sequence of float values using SIMD @b SSE intrinsics,
/// processing 128 bits of data on every logic thread.
struct sse_f32aligned_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    float operator()() const noexcept {
        auto it = begin_;
        auto a = _mm_set1_ps(0);
        auto const last_sse_ptr = begin_ + ((end_ - begin_) / 4 - 1) * 4;
        while (it != last_sse_ptr) {
            a = _mm_add_ps(a, _mm_load_ps(it));
            it += 4;
        }
        a = _mm_hadd_ps(a, a);
        a = _mm_hadd_ps(a, a);
        return _mm_cvtss_f32(a);
    }
};

#if defined(__AVX2__)

/// Reduces a __m256 vector to a single float by horizontal addition.
inline static float _mm256_reduce_add_ps(__m256 x) noexcept {
    x = _mm256_add_ps(x, _mm256_permute2f128_ps(x, x, 1));
    x = _mm256_hadd_ps(x, x);
    x = _mm256_hadd_ps(x, x);
    return _mm256_cvtss_f32(x);
}

/// Computes the sum of a sequence of float values using SIMD @b AVX2 intrinsics,
/// processing 256 bits of data on every logic thread.
///
/// Links:
/// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX,AVX2&text=add_ps
/// https://stackoverflow.com/a/23190168
struct avx2_f32_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    inline float operator()() const noexcept {
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

/// Computes the sum of a sequence of float values using SIMD @b AVX2 intrinsics,
/// but unlike the `avx2_f32_t` it uses Kahan stable summation algorithm to compensate floating point error.
/// https://en.wikipedia.org/wiki/Kahan_summation_algorithm
struct avx2_f32kahan_t {

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
        auto running_sum = _mm256_reduce_add_ps(running_sums);
        for (; it != end_; ++it)
            running_sum += *it;

        return running_sum;
    }
};

/// Computes the sum of a sequence of float values using SIMD @b AVX2 intrinsics,
/// but unlike the `avx2_f32_t` it uses double precision to compensate floating point error.
struct avx2_f64_t {

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

/// Computes the sum of a sequence of float values using SIMD @b AVX2 intrinsics,
/// processing 256 bits of data on every logic thread, but unlike the `avx2_f32_t`
/// it only performs aligned memory accesses.
struct avx2_f32aligned_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    float operator()() const noexcept {
        auto it = begin_;
        auto a = _mm256_set1_ps(0);
        auto const last_avx_ptr = begin_ + ((end_ - begin_) / 8 - 1) * 8;
        while (it != last_avx_ptr) {
            a = _mm256_add_ps(a, _mm256_load_ps(it));
            it += 8;
        }
        return _mm256_reduce_add_ps(a);
    }
};

#endif

#if defined(__AVX512F__)

/// Computes the sum of a sequence of float values using SIMD @b AVX-512 intrinsics,
/// using streaming loads and bidirectional accumulation into 2 separate ZMM registers.
struct avx512_f32streamed_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    float operator()() const noexcept {
        auto it_begin = begin_;
        auto it_end = end_;

        __m512 acc1 = _mm512_set1_ps(0.0f); // Accumulator for forward direction
        __m512 acc2 = _mm512_set1_ps(0.0f); // Accumulator for reverse direction

        // Process in chunks of 32 floats in each direction
        for (; it_end - it_begin >= 64; it_begin += 32, it_end -= 32) {
            acc1 = _mm512_add_ps(acc1, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_begin))));
            acc2 = _mm512_add_ps(acc2, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_end - 32))));
        }
        if (it_end - it_begin >= 32) {
            acc1 = _mm512_add_ps(acc1, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_begin))));
            it_begin += 32;
        }

        // Combine the accumulators
        __m512 acc = _mm512_add_ps(acc1, acc2);
        float sum = _mm512_reduce_add_ps(acc);
        while (it_begin < it_end)
            sum += *it_begin++;
        return sum;
    }
};

/// Computes the sum of a sequence of float values using SIMD @b AVX-512 intrinsics,
/// using caching loads and bidirectional traversal using all the available ZMM registers.
struct avx512_f32unrolled_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    float operator()() const noexcept {
        auto it_begin = begin_;
        auto it_end = end_;

        // We have a grand-total of 32 floats in a ZMM register.
        // We want to keep half of them free for loading buffers, and the rest can be used for accumulation:
        // 8 in the forward direction, 8 in the reverse direction, and 16 for the accumulator.
        __m512 fwd0 = _mm512_set1_ps(0.0f), rev0 = _mm512_set1_ps(0.0f);
        __m512 fwd1 = _mm512_set1_ps(0.0f), rev1 = _mm512_set1_ps(0.0f);
        __m512 fwd2 = _mm512_set1_ps(0.0f), rev2 = _mm512_set1_ps(0.0f);
        __m512 fwd3 = _mm512_set1_ps(0.0f), rev3 = _mm512_set1_ps(0.0f);
        __m512 fwd4 = _mm512_set1_ps(0.0f), rev4 = _mm512_set1_ps(0.0f);
        __m512 fwd5 = _mm512_set1_ps(0.0f), rev5 = _mm512_set1_ps(0.0f);
        __m512 fwd6 = _mm512_set1_ps(0.0f), rev6 = _mm512_set1_ps(0.0f);
        __m512 fwd7 = _mm512_set1_ps(0.0f), rev7 = _mm512_set1_ps(0.0f);

        // Process in chunks of 32 floats x 8 ZMM registers = 256 floats in each direction
        for (; it_end - it_begin >= 512; it_begin += 256, it_end -= 256) {
            fwd0 = _mm512_add_ps(fwd0, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 32 * 0))));
            fwd1 = _mm512_add_ps(fwd1, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 32 * 1))));
            fwd2 = _mm512_add_ps(fwd2, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 32 * 2))));
            fwd3 = _mm512_add_ps(fwd3, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 32 * 3))));
            fwd4 = _mm512_add_ps(fwd4, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 32 * 4))));
            fwd5 = _mm512_add_ps(fwd5, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 32 * 5))));
            fwd6 = _mm512_add_ps(fwd6, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 32 * 6))));
            fwd7 = _mm512_add_ps(fwd7, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 32 * 7))));
            rev0 = _mm512_add_ps(rev0, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 32 * (1 + 0)))));
            rev1 = _mm512_add_ps(rev1, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 32 * (1 + 1)))));
            rev2 = _mm512_add_ps(rev2, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 32 * (1 + 2)))));
            rev3 = _mm512_add_ps(rev3, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 32 * (1 + 3)))));
            rev4 = _mm512_add_ps(rev4, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 32 * (1 + 4)))));
            rev5 = _mm512_add_ps(rev5, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 32 * (1 + 5)))));
            rev6 = _mm512_add_ps(rev6, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 32 * (1 + 6)))));
            rev7 = _mm512_add_ps(rev7, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 32 * (1 + 7)))));
        }
        for (; it_end - it_begin >= 32; it_begin += 32)
            fwd1 = _mm512_add_ps(fwd1, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_begin))));

        // Combine the accumulators
        __m512 fwd = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(fwd0, fwd1), _mm512_add_ps(fwd2, fwd3)),
                                   _mm512_add_ps(_mm512_add_ps(fwd4, fwd5), _mm512_add_ps(fwd5, fwd7)));
        __m512 rev = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(rev0, rev1), _mm512_add_ps(rev2, rev3)),
                                   _mm512_add_ps(_mm512_add_ps(rev4, rev5), _mm512_add_ps(rev5, rev7)));
        __m512 acc = _mm512_add_ps(fwd, rev);
        float sum = _mm512_reduce_add_ps(acc);
        while (it_begin < it_end)
            sum += *it_begin++;
        return sum;
    }
};

#endif

#pragma region Multi Core

/// Computes the sum of a sequence of float values using @b OpenMP on-CPU multi-core reductions acceleration.
/// https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-reduction.html
struct openmp_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

    openmp_t(float const *b, float const *e) : begin_(b), end_(e) {
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

/// Computes the sum of a sequence of float values using @b std::thread on-CPU multi-core reductions acceleration.
/// https://en.cppreference.com/w/cpp/thread/thread
template <typename serial_at = stl_accumulate_gt<float>> struct threads_gt {

    float *const begin_ = nullptr;
    float *const end_ = nullptr;
    std::vector<std::thread> threads_;
    std::vector<double> sums_;

    threads_gt(float *b, float *e) : begin_(b), end_(e), sums_() {
        auto cores = total_cores();
        threads_.reserve(cores);
        sums_.resize(cores);
    }

    size_t count_per_thread() const { return (end_ - begin_) / sums_.size(); }

    struct thread_task_t {
        float *const begin_;
        float *const end_;
        double &output_;
        inline void operator()() const noexcept { output_ = serial_at{begin_, end_}(); }
    };

    double operator()() {
        auto it = begin_;
        size_t const batch_size = count_per_thread();

        // Start the child threads
        for (size_t i = 0; i + 1 < sums_.size(); ++i, it += batch_size)
            threads_.emplace_back(thread_task_t{it, it + batch_size, sums_[i]});

        // This thread lives by its own rules :)
        double running_sum = 0;
        thread_task_t{it, it + batch_size, running_sum}();

        // Accumulate sums from child threads.
        for (size_t i = 1; i < sums_.size(); ++i) {
            threads_[i - 1].join();
            running_sum += sums_[i];
        }

        threads_.clear();
        return running_sum;
    }
};

} // namespace ashvardanian::reduce