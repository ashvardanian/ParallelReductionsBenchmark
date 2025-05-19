/**
 *  @brief BLAS-based reductions
 *  @file reduce_blas.hpp
 *  @author Ash Vardanian
 *  @date 19/01/2025
 */
#pragma once
#include <limits>    // `std::numeric_limits`
#include <stdexcept> // `std::length_error`

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h> // Apple's BLAS/CBLAS
#else
#include <cblas.h> // OpenBLAS, MKL, etc.
#endif

namespace ashvardanian {

/**
 *  @brief Using BLAS dot-product interface to accumulate a vector.
 *
 *  BLAS interfaces have a convenient "stride" parameter that can be used to
 *  apply the kernel to various data layouts. Similarly, if we set the stride
 *  to @b zero, we can fool the kernels into thinking that a scalar is a vector.
 */
class blas_dot_t {
    float const *const begin_ = nullptr;
    float const *const end_ = nullptr;

#if defined(CBLAS_INDEX)
    using blas_dim_t = CBLAS_INDEX;
#else
    using blas_dim_t = blasint;
#endif

  public:
    blas_dot_t() = default;
    blas_dot_t(float const *b, float const *e) : begin_(b), end_(e) {
        constexpr std::size_t max_length_k = static_cast<std::size_t>(std::numeric_limits<blas_dim_t>::max());
        if (end_ - begin_ > max_length_k) throw std::length_error("BLAS not configured for 64-bit sizes");
    }

    float operator()() const noexcept {
        float repeated_ones[1];
        repeated_ones[0] = 1.0f;
        return cblas_sdot(end_ - begin_, begin_, 1, &repeated_ones[0], 0);
    }
};

} // namespace ashvardanian