/**
 *  @brief Parallel reduction with SIMD and multicore acceleration
 *  @file reduce_cpu.hpp
 *  @author Ash Vardanian
 *  @date 04/09/2019
 */
#pragma once
#include <cstring>   // `std::memcpy`
#include <execution> // `std::execution::par_unseq`
#include <numeric>   // `std::accumulate`, `std::reduce`
#include <thread>    // `std::thread`

#if defined(_OPENMP)
#include <omp.h> // `omp_set_num_threads`
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h> // x86 intrinsics
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h> // ARM NEON intrinsics
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h> // ARM SVE intrinsics
#endif

#include <fork_union.hpp>

namespace ashvardanian {

/**
 *  @brief Returns the current number of logical cores on the CPU.
 *         On x86 this is the number of threads, not the number of physical cores
 *         due to Simultaneous Multi-Threading @b (SMT) or Hyper-Threading (HT).
 */
inline static std::size_t total_cores() { return std::thread::hardware_concurrency(); }

/**
 *  @brief Divides a value by another value and rounds it up to the nearest integer.
 *         Example: `round_up_to_multiple(5, 3) == 2`
 */
inline static std::size_t divide_round_up(std::size_t value, std::size_t multiple) noexcept {
    return ((value + multiple - 1) / multiple);
}

/**
 *  @brief Rounds a value up to the nearest multiple of another value.
 *         Example: `round_up_to_multiple(5, 3) == 6`
 */
inline static std::size_t round_up_to_multiple(std::size_t value, std::size_t multiple) noexcept {
    return divide_round_up(value, multiple) * multiple;
}

#pragma region - Serial and Autovectorized

/**
 *  @brief Computes the sum of a sequence of float values using an unrolled @b `for`-loop,
 *         accumulating into 8 separate registers and summing them at the end.
 *         It's a common approach to simplify compilers' auto-vectorization job.
 */
template <typename accumulator_at = float>
class unrolled_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    unrolled_gt() = default;
    unrolled_gt(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

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
        for (; it != end_; ++it) sums[0] += *it;
        return sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
    }
};

/**
 *  @brief Computes the sum of a sequence of float values using @b `std::accumulate`,
 *         Standard Library's serial reduction algorithm. It shouldn't perform better
 *         than `unrolled_gt`.
 */
template <typename accumulator_at = float>
class stl_accumulate_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    stl_accumulate_gt() = default;
    stl_accumulate_gt(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    accumulator_at operator()() const noexcept { return std::accumulate(begin_, end_, accumulator_at(0)); }
};

#if defined(__cpp_lib_execution)

/**
 *  @brief Computes the sum of a sequence of float values using STL's parallel @b `std::reduce`
 *         with execution policy @b `std::execution::par`.
 */
template <typename accumulator_at = float>
class stl_par_reduce_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    stl_par_reduce_gt() = default;
    stl_par_reduce_gt(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    accumulator_at operator()() const noexcept {
        return std::reduce(std::execution::par, begin_, end_, accumulator_at(0), std::plus<accumulator_at>());
    }
};

/**
 *  @brief Computes the sum of a sequence of float values using parallel `std::reduce` with execution
 *         policy @b `std::execution::par_unseq` for non-blocking parallelism.
 */
template <typename accumulator_at = float>
class stl_par_unseq_reduce_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    stl_par_unseq_reduce_gt() = default;
    stl_par_unseq_reduce_gt(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    accumulator_at operator()() const noexcept {
        return std::reduce(std::execution::par_unseq, begin_, end_, accumulator_at(0), std::plus<accumulator_at>());
    }
};

#endif // defined(__cpp_lib_execution)

#pragma endregion - Serial and Autovectorized

#pragma region - Handwritten SIMD Kernels
#pragma region x86

#if defined(__SSE__)

/**
 *  @brief Computes the sum of a sequence of float values using SIMD @b SSE intrinsics,
 *         processing 128 bits of data on every logic thread. It's the largest register
 *         size universally available across x86, Arm, and RISC-V architectures, making
 *         it a good baseline for portable SIMD code.
 */
class sse_f32aligned_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    sse_f32aligned_t() = default;
    sse_f32aligned_t(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    float operator()() const noexcept {
        auto const count_sse = (end_ - begin_) / 4;
        auto const last_sse_ptr = begin_ + count_sse * 4;
        auto it = begin_;

        auto running_sums = _mm_setzero_ps();
        for (; it != last_sse_ptr; it += 4) running_sums = _mm_add_ps(running_sums, _mm_load_ps(it));

        auto running_sum = 0.f;
        for (; it != end_; ++it) running_sum += *it;

        running_sums = _mm_hadd_ps(running_sums, running_sums);
        running_sums = _mm_hadd_ps(running_sums, running_sums);
        return _mm_cvtss_f32(running_sums) + running_sum;
    }
};

#endif // defined(__SSE__)

#if defined(__AVX2__)

/**
 *  @brief Reduces a `__m256` vector to a single float using horizontal additions
 *         in 3 tree-like steps as opposed to 7 sequential additions.
 *  @see   https://stackoverflow.com/a/23190168
 */
inline static float _mm256_reduce_add_ps(__m256 x) noexcept {
    x = _mm256_add_ps(x, _mm256_permute2f128_ps(x, x, 1));
    x = _mm256_hadd_ps(x, x);
    x = _mm256_hadd_ps(x, x);
    return _mm256_cvtss_f32(x);
}

/**
 *  @brief Computes the sum of a sequence of float values using SIMD @b AVX2 intrinsics,
 *         processing 256 bits of data on every logic thread. Available on most CPUs after 2013.
 *  @see   https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX,AVX2&text=add_ps
 */
class avx2_f32_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    avx2_f32_t() = default;
    avx2_f32_t(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    float operator()() const noexcept {
        auto it = begin_;

        // SIMD-parallel summation stage
        auto running_sums = _mm256_setzero_ps();
        for (; it + 8 < end_; it += 8) running_sums = _mm256_add_ps(_mm256_loadu_ps(it), running_sums);

        // Serial summation
        auto running_sum = _mm256_reduce_add_ps(running_sums);
        for (; it != end_; ++it) running_sum += *it;

        return running_sum;
    }
};

/**
 *  @brief Computes the sum of a sequence of float values using SIMD @b AVX2 intrinsics,
 *         but unlike the `avx2_f32_t` it uses Kahan stable summation algorithm to compensate
 *         floating point error.
 *  @see   https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 */
class avx2_f32kahan_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    avx2_f32kahan_t() = default;
    avx2_f32kahan_t(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    float operator()() const noexcept {
        auto it = begin_;

        // SIMD-parallel summation stage
        auto running_sums = _mm256_setzero_ps();
        auto compensations = _mm256_setzero_ps();
        auto t = _mm256_setzero_ps();
        auto y = _mm256_setzero_ps();
        for (; it + 8 < end_; it += 8) {
            y = _mm256_sub_ps(_mm256_loadu_ps(it), compensations);
            t = _mm256_add_ps(running_sums, y);
            compensations = _mm256_sub_ps(_mm256_sub_ps(t, running_sums), y);
            running_sums = t;
        }

        // Serial summation
        auto running_sum = _mm256_reduce_add_ps(running_sums);
        for (; it != end_; ++it) running_sum += *it;

        return running_sum;
    }
};

/**
 *  @brief Computes the sum of a sequence of float values using SIMD @b AVX2 intrinsics,
 *         but unlike the `avx2_f32_t` it uses double precision to compensate floating point error.
 */
class avx2_f64_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    avx2_f64_t() = default;
    avx2_f64_t(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    double operator()() const noexcept {
        auto it = begin_;

        // SIMD-parallel summation stage
        auto running_sums = _mm256_set1_pd(0);
        for (; it + 4 < end_; it += 4) running_sums = _mm256_add_pd(_mm256_cvtps_pd(_mm_loadu_ps(it)), running_sums);

        // Serial summation
        running_sums = _mm256_hadd_pd(running_sums, running_sums);
        running_sums = _mm256_hadd_pd(running_sums, running_sums);
        auto running_sum = _mm256_cvtsd_f64(running_sums);
        for (; it != end_; ++it) running_sum += *it;

        return running_sum;
    }
};

/**
 *  @brief Computes the sum of a sequence of float values using SIMD @b AVX2 intrinsics,
 *         processing 256 bits of data on every logic thread, but unlike the `avx2_f32_t`
 *         it only performs aligned memory accesses.
 */
class avx2_f32aligned_t {

    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    avx2_f32aligned_t() = default;
    avx2_f32aligned_t(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    float operator()() const noexcept {
        auto it = begin_;
        auto running_sums = _mm256_setzero_ps();
        auto count_avx = (end_ - begin_) / 8;
        auto const last_avx_ptr = begin_ + count_avx * 8;
        for (; it != last_avx_ptr; it += 8) running_sums = _mm256_add_ps(running_sums, _mm256_load_ps(it));
        auto running_sum = 0.f;
        for (; it != end_; ++it) running_sum += *it;
        return _mm256_reduce_add_ps(running_sums) + running_sum;
    }
};

#endif // defined(__AVX2__)

#if defined(__AVX512F__)

/**
 *  @brief Computes the sum of a sequence of float values using SIMD @b AVX-512 intrinsics,
 *         using @b non-temporal streaming loads and @b bidirectional accumulation into two
 *         separate ZMM registers.
 *
 *  On both Intel and AMD this instruction has 3-4 cycle latency and can generally be executed
 *  on 2 ports, so we shouldn't get benefits from unrolling the loop further.
 */
class avx512_f32streamed_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    avx512_f32streamed_t() = default;
    avx512_f32streamed_t(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    float operator()() const noexcept {
        auto it_begin = begin_;
        auto it_end = end_;

        __m512 acc1 = _mm512_setzero_ps(); // Accumulator for forward direction
        __m512 acc2 = _mm512_setzero_ps(); // Accumulator for reverse direction
        static_assert(sizeof(__m512) / sizeof(float) == 16, "AVX-512 register size is 16 floats");

        // Process in chunks of 16 floats in each direction = 32 per cycle
        for (; it_end - it_begin >= 32; it_begin += 16, it_end -= 16) {
            acc1 = _mm512_add_ps(acc1, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_begin))));
            acc2 = _mm512_add_ps(acc2, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_end - 16))));
        }
        if (it_end - it_begin >= 16) {
            acc1 = _mm512_add_ps(acc1, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_begin))));
            it_begin += 16;
        }

        // Combine the accumulators
        __m512 acc = _mm512_add_ps(acc1, acc2);
        float sum = _mm512_reduce_add_ps(acc);
        while (it_begin < it_end) sum += *it_begin++;
        return sum;
    }
};

/**
 *  @brief Computes the sum of a sequence of float values using SIMD @b AVX-512 intrinsics,
 *         using @b caching loads and @b bidirectional traversal using @b all the available
 *         ZMM registers. Shouldn't perform better than `avx512_f32streamed_t`.
 */
class avx512_f32unrolled_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    avx512_f32unrolled_t() = default;
    avx512_f32unrolled_t(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    float operator()() const noexcept {
        auto it_begin = begin_;
        auto it_end = end_;

        // We have a grand-total of 32 floats in a ZMM register.
        // We want to keep half of them free for loading buffers, and the rest can be used for accumulation:
        // 8 in the forward direction, 8 in the reverse direction, and 16 for the accumulator.
        __m512 fwd0 = _mm512_setzero_ps(), rev0 = _mm512_setzero_ps();
        __m512 fwd1 = _mm512_setzero_ps(), rev1 = _mm512_setzero_ps();
        __m512 fwd2 = _mm512_setzero_ps(), rev2 = _mm512_setzero_ps();
        __m512 fwd3 = _mm512_setzero_ps(), rev3 = _mm512_setzero_ps();
        __m512 fwd4 = _mm512_setzero_ps(), rev4 = _mm512_setzero_ps();
        __m512 fwd5 = _mm512_setzero_ps(), rev5 = _mm512_setzero_ps();
        __m512 fwd6 = _mm512_setzero_ps(), rev6 = _mm512_setzero_ps();
        __m512 fwd7 = _mm512_setzero_ps(), rev7 = _mm512_setzero_ps();
        static_assert(sizeof(__m512) / sizeof(float) == 16, "AVX-512 register size is 16 floats");

        // Process in chunks of 16 floats x 8 ZMM registers = 128 floats in each direction = 256 per cycle
        for (; it_end - it_begin >= 256; it_begin += 128, it_end -= 128) {
            fwd0 = _mm512_add_ps(fwd0, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 16 * 0))));
            fwd1 = _mm512_add_ps(fwd1, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 16 * 1))));
            fwd2 = _mm512_add_ps(fwd2, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 16 * 2))));
            fwd3 = _mm512_add_ps(fwd3, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 16 * 3))));
            fwd4 = _mm512_add_ps(fwd4, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 16 * 4))));
            fwd5 = _mm512_add_ps(fwd5, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 16 * 5))));
            fwd6 = _mm512_add_ps(fwd6, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 16 * 6))));
            fwd7 = _mm512_add_ps(fwd7, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_begin + 16 * 7))));
            rev0 = _mm512_add_ps(rev0, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 16 * (1 + 0)))));
            rev1 = _mm512_add_ps(rev1, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 16 * (1 + 1)))));
            rev2 = _mm512_add_ps(rev2, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 16 * (1 + 2)))));
            rev3 = _mm512_add_ps(rev3, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 16 * (1 + 3)))));
            rev4 = _mm512_add_ps(rev4, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 16 * (1 + 4)))));
            rev5 = _mm512_add_ps(rev5, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 16 * (1 + 5)))));
            rev6 = _mm512_add_ps(rev6, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 16 * (1 + 6)))));
            rev7 = _mm512_add_ps(rev7, _mm512_castsi512_ps(_mm512_load_si512((void *)(it_end - 16 * (1 + 7)))));
        }
        for (; it_end - it_begin >= 16; it_begin += 16)
            fwd1 = _mm512_add_ps(fwd1, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_begin))));

        // Combine the accumulators
        __m512 fwd = _mm512_add_ps( //
            _mm512_add_ps(_mm512_add_ps(fwd0, fwd1), _mm512_add_ps(fwd2, fwd3)),
            _mm512_add_ps(_mm512_add_ps(fwd4, fwd5), _mm512_add_ps(fwd5, fwd7)));
        __m512 rev = _mm512_add_ps( //
            _mm512_add_ps(_mm512_add_ps(rev0, rev1), _mm512_add_ps(rev2, rev3)),
            _mm512_add_ps(_mm512_add_ps(rev4, rev5), _mm512_add_ps(rev5, rev7)));
        __m512 acc = _mm512_add_ps(fwd, rev);
        float sum = _mm512_reduce_add_ps(acc);
        while (it_begin < it_end) sum += *it_begin++;
        return sum;
    }
};

/**
 *  @brief  Computes the sum of a sequence of float values using SIMD @b AVX-512 intrinsics,
 *          using @b non-temporal streaming loads and @b bidirectional accumulation into two
 *          separate ZMM registers, and @b interleaving the additions and FMAs, executing on
 *          different CPU @b ports on AMD Zen4.
 *
 *  This hides the latency of expensive operations and improves hardware utilization. This
 *  kernel should be the fastest on new AMD machines!
 */
class avx512_f32interleaving_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    avx512_f32interleaving_t() = default;
    avx512_f32interleaving_t(float const *b, float const *e) : begin_(b), end_(e) {}

    float operator()() const noexcept {
        auto it_begin = begin_;
        auto it_end = end_;

        __m512 acc1 = _mm512_setzero_ps(); // Accumulator for forward direction addition
        __m512 acc2 = _mm512_setzero_ps(); // Accumulator for reverse direction addition
        __m512 acc3 = _mm512_setzero_ps(); // Accumulator for forward direction FMAs
        __m512 acc4 = _mm512_setzero_ps(); // Accumulator for reverse direction FMAs
        __m512 ones = _mm512_set1_ps(1.0f);
        static_assert(sizeof(__m512) / sizeof(float) == 16, "AVX-512 register size is 16 floats");

        // Process in chunks of 32 floats in each direction
        for (; it_end - it_begin >= 64; it_begin += 32, it_end -= 32) {
            acc1 = _mm512_add_ps(acc1, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_begin))));
            acc2 = _mm512_add_ps(acc2, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_end - 16))));
            acc3 = _mm512_fmadd_ps(ones, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_begin + 32))), acc3);
            acc4 = _mm512_fmadd_ps(ones, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_end - 16))), acc4);
        }
        while (it_end - it_begin >= 16) {
            acc1 = _mm512_add_ps(acc1, _mm512_castsi512_ps(_mm512_stream_load_si512((void *)(it_begin))));
            it_begin += 16;
        }

        // Combine the accumulators
        __m512 acc = _mm512_add_ps(_mm512_add_ps(acc1, acc2), _mm512_add_ps(acc2, acc3));
        float sum = _mm512_reduce_add_ps(acc);
        while (it_begin < it_end) sum += *it_begin++;
        return sum;
    }
};

#endif // defined(__AVX512F__)

#pragma endregion x86
#pragma region ARM
#if defined(__ARM_NEON)

/**
 *  @brief Computes the sum of a sequence of float values using SIMD @b NEON intrinsics,
 *         processing 128 bits (4 floats) per vector.
 */
class neon_f32_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    neon_f32_t() = default;
    neon_f32_t(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    float operator()() const noexcept {
        auto const count_neon = (end_ - begin_) / 4;
        auto const last_neon_ptr = begin_ + count_neon * 4;
        auto it = begin_;

        float32x4_t running_sums = vdupq_n_f32(0.f);
        for (; it != last_neon_ptr; it += 4) running_sums = vaddq_f32(running_sums, vld1q_f32(it));

        float running_sum = vaddvq_f32(running_sums);
        for (; it != end_; ++it) running_sum += *it;
        return running_sum;
    }
};

#endif // defined(__ARM_NEON)

#if defined(__ARM_FEATURE_SVE)

/**
 *  @brief Computes the sum of a sequence of float values using SIMD @b SVE intrinsics,
 *         processing multiple entries per cycle.
 */
class sve_f32_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    sve_f32_t() = default;
    sve_f32_t(float const *b, float const *e) noexcept : begin_(b), end_(e) {}

    float operator()() const noexcept {
        auto const sve_register_width = svcntw();
        auto const input_size = static_cast<std::size_t>(end_ - begin_);

        svfloat32_t running_sums = svdup_f32(0.f);
        for (std::size_t start_offset = 0; start_offset < input_size; start_offset += sve_register_width) {
            svbool_t progress_vec = svwhilelt_b32(start_offset, input_size);
            running_sums = svadd_f32_m(progress_vec, running_sums, svld1(progress_vec, begin_ + start_offset));
        }

        // No need to handle the tail separately
        float const running_sum = svaddv(svptrue_b32(), running_sums);
        return running_sum;
    }
};

#endif // defined(__ARM_FEATURE_SVE__)

#pragma endregion ARM
#pragma endregion Handwritten SIMD Kernels

#pragma region - Multicore

#if defined(_OPENMP)

/**
 *  @brief Computes the sum of a sequence of float values using @b OpenMP on-CPU
 *         for multi-core reductions acceleration.
 *  @see   https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-reduction.html
 */
class openmp_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

  public:
    openmp_t() = default;
    openmp_t(float const *b, float const *e) : begin_(b), end_(e) {
        omp_set_dynamic(0);
        omp_set_num_threads(total_cores());
    }

    float operator()() const noexcept {
        float sum = 0;
        std::size_t const n = end_ - begin_;
#pragma omp parallel for default(shared) reduction(+ : sum)
        for (std::size_t i = 0; i != n; i++) sum += begin_[i];
        return sum;
    }
};

/**
 *  @brief Computes the sum of a sequence of float values using @b OpenMP on-CPU
 *         for multi-core parallelism, combined with the given @b SIMD vectorization.
 *  @see   https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-reduction.html
 */
template <typename serial_at = stl_accumulate_gt<float>>
class openmp_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;
    std::size_t const total_cores_ = 0;
    std::vector<double> sums_;

  public:
    openmp_gt() = default;
    openmp_gt(float const *b, float const *e) : begin_(b), end_(e), total_cores_(total_cores()), sums_(total_cores_) {
        omp_set_dynamic(0);
        omp_set_num_threads(total_cores_);
    }

    double operator()() {
        auto const input_size = static_cast<std::size_t>(end_ - begin_);
        auto const chunk_size = divide_round_up(input_size, total_cores_);
#pragma omp parallel
        {
            std::size_t const thread_id = static_cast<std::size_t>(omp_get_thread_num());
            std::size_t const start = std::min(thread_id * chunk_size, input_size);
            std::size_t const stop = std::min(start + chunk_size, input_size);
            double local_sum = serial_at {begin_ + start, begin_ + stop}();
            sums_[thread_id] = local_sum;
        }
        return std::accumulate(sums_.begin(), sums_.end(), 0.0);
    }
};

#endif

/**
 *  @brief Computes the sum of a sequence of float values using @b `std::thread` on-CPU
 *         multi-core reductions acceleration.
 *  @see   https://en.cppreference.com/w/cpp/thread/thread
 */
template <typename serial_at = stl_accumulate_gt<float>>
class threads_gt {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;
    std::vector<std::thread> threads_;
    std::vector<double> sums_;

    struct thread_task_t {
        float const *const begin_;
        float const *const end_;
        double &output_;

        thread_task_t(float const *b, float const *e, double &out) noexcept : begin_(b), end_(e), output_(out) {}
        inline void operator()() const noexcept { output_ = serial_at {begin_, end_}(); }
    };

  public:
    threads_gt() = default;
    threads_gt(float const *b, float const *e) : begin_(b), end_(e), sums_() {
        auto cores = total_cores();
        threads_.reserve(cores);
        sums_.resize(cores);
    }

    std::size_t count_per_thread() const noexcept {
        constexpr std::size_t entries_per_zmm_register = 64 / sizeof(float);
        std::size_t balanced_split = divide_round_up(end_ - begin_, sums_.size());
        return round_up_to_multiple(balanced_split, entries_per_zmm_register);
    }

    double operator()() {
        auto it = begin_;
        std::size_t const batch_size = count_per_thread();

        // Start the child threads
        for (std::size_t i = 0; i + 1 < sums_.size(); ++i, it += batch_size)
            threads_.emplace_back(thread_task_t {it, it + batch_size, sums_[i]});

        // This thread lives by its own rules and may end up processing a smaller batch :)
        double running_sum = 0;
        thread_task_t {it, end_, running_sum}();

        // Accumulate sums from child threads.
        for (std::size_t i = 1; i < sums_.size(); ++i) {
            threads_[i - 1].join();
            running_sum += sums_[i];
        }

        threads_.clear();
        return running_sum;
    }
};

/**
 *  @brief Computes the sum of a sequence of float values using @b `std::thread` on-CPU
 *         multi-core reductions acceleration, reusing a fixed-size thread pool.
 *  @see   https://github.com/ashvardanian/fork_union
 */
template <typename serial_at = stl_accumulate_gt<float>>
class fork_union_gt {
    using pool_t = ::ashvardanian::fork_union_t;
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;
    pool_t pool_;
    std::vector<double> sums_;

  public:
    fork_union_gt() = default;
    fork_union_gt(float const *b, float const *e) : begin_(b), end_(e), sums_() {
        auto cores = total_cores();
        if (!pool_.try_spawn(cores)) throw std::runtime_error("Failed to fork threads");
        sums_.resize(cores);
    }

    double operator()() {
        auto const input_size = static_cast<std::size_t>(end_ - begin_);
        pool_.for_each_slice(input_size, [this](pool_t::task_t first_task, std::size_t slice_length) noexcept {
            auto const slice_begin = begin_ + first_task.task_index;
            sums_[first_task.thread_index] = serial_at {slice_begin, slice_begin + slice_length}();
        });
        return std::accumulate(sums_.begin(), sums_.end(), 0.0);
    }
};

#pragma endregion - Multicore

} // namespace ashvardanian