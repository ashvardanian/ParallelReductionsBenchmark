#pragma once
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace av {

/**
 * @brief Using CUDA warp-level primitives on Nvidia Kepler GPUs and newer.
 *
 * Reading:
 * https://stackoverflow.com/a/25584577
 * https://stackoverflow.com/q/12733084
 * https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
 * https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-shuffle/
 * https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 */
__global__ void cuda_block_sum(float *d_sum, float *d_data) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    temp[tid] = d_data[tid + blockIdx.x * blockDim.x];
    for (int d = blockDim.x >> 1; d >= 1; d >>= 1) {
        __syncthreads();
        if (tid < d)
            temp[tid] += temp[tid + d];
    }
    if (tid == 0)
        d_sum[blockIdx.x] = temp[0];
}

struct cuda_t {
    thrust::device_vector<float> on_gpu;
    cuda_t(float const *b, float const *e) : on_gpu(b, e) {}
    float operator()(float const *, float const *) const;
};

/**
 * @brief Using Thrust tempaltes library for parallel reductions
 * on Nvidia GPUs, whithout explicitly writing a single line of CUDA.
 * https://docs.nvidia.com/cuda/thrust/index.html#reductions
 */
struct cuda_thrust_t {
    thrust::device_vector<float> on_gpu;
    cuda_thrust_t(float const *b, float const *e) : on_gpu(b, e) {}
    float operator()() const noexcept {
        return thrust::reduce(on_gpu.begin(), on_gpu.end(), float(0), thrust::plus<float>());
    }
};

} // namespace av