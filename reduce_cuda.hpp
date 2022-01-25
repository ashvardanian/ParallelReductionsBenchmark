#pragma once
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

namespace av {

static constexpr int warp_size_k = 32;

/**
 * @brief Using CUDA warp-level primitives on Nvidia Kepler GPUs and newer.
 *
 * Reading:
 * https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-
 * https://stackoverflow.com/inputs/25584577
 * https://stackoverflow.com/q/12733084
 * https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
 * https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-shuffle/
 * https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 */
template <int block_size_ak = 1024>
__global__ void reduce_2step_blocks(const float *inputs, int input_size, float *outputs) {
    int const thread_idx = threadIdx.x;
    int gthread_idx = thread_idx + blockIdx.x * block_size_ak;
    int const grid_size = block_size_ak * gridDim.x;
    float sum = 0;
    for (int i = gthread_idx; i < input_size; i += grid_size)
        sum += inputs[i];

    __shared__ float shared_buffer[block_size_ak];
    shared_buffer[thread_idx] = sum;
    __syncthreads();
    for (int size = block_size_ak >> 1; size > 0; size >>= 1) {
        if (thread_idx < size)
            shared_buffer[thread_idx] += shared_buffer[thread_idx + size];
        __syncthreads();
    }
    if (thread_idx == 0)
        outputs[blockIdx.x] = shared_buffer[0];
}

inline __device__ float warp_reduce(float volatile *shared_buffer) {
    int lane_in_warp = threadIdx.x & (warp_size_k - 1);
    if (lane_in_warp < 16) {
        shared_buffer[lane_in_warp] += shared_buffer[lane_in_warp + 16];
        shared_buffer[lane_in_warp] += shared_buffer[lane_in_warp + 8];
        shared_buffer[lane_in_warp] += shared_buffer[lane_in_warp + 4];
        shared_buffer[lane_in_warp] += shared_buffer[lane_in_warp + 2];
        shared_buffer[lane_in_warp] += shared_buffer[lane_in_warp + 1];
    }
    return shared_buffer[0];
}

template <int block_size_ak = 1024>
__global__ void reduce_2step_warps(float const *inputs, int input_size, float *out) {
    int idx = threadIdx.x;
    float sum = 0;
    for (int i = idx; i < input_size; i += block_size_ak)
        sum += inputs[i];
    __shared__ float r[block_size_ak];
    r[idx] = sum;
    warp_reduce(&r[idx & ~(warp_size_k - 1)]);
    __syncthreads();
    if (idx < warp_size_k) {
        r[idx] = idx * warp_size_k < block_size_ak ? r[idx * warp_size_k] : 0;
        warp_reduce(r);
        if (idx == 0)
            *out = r[0];
    }
}

inline __device__ float registers_reduce(float value) {
    value += __shfl_down(value, 1);
    value += __shfl_down(value, 2);
    value += __shfl_down(value, 4);
    value += __shfl_down(value, 8);
    value += __shfl_down(value, 16);
    return __shfl(value, 0);
}

template <int block_size_ak = 1024>
__global__ void reduce_2step_registers(float const *inputs, int input_size, float *out) {
    int idx = threadIdx.x;
    float sum = 0;
    for (int i = idx; i < input_size; i += block_size_ak)
        sum += inputs[i];
    __shared__ float r[block_size_ak];
    r[idx] = sum;
    warp_reduce(&r[idx & ~(warp_size_k - 1)]);
    __syncthreads();
    if (idx < warp_size_k) {
        r[idx] = idx * warp_size_k < block_size_ak ? r[idx * warp_size_k] : 0;
        warp_reduce(r);
        if (idx == 0)
            *out = r[0];
    }
}

inline static size_t cuda_device_count() {
    int count;
    auto error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess)
        return 0;
    return static_cast<size_t>(count);
}

enum class cuda_kernel_t {
    blocks_k,
    warps_k,
    registers_k,
};

template <cuda_kernel_t kernel_ak> struct cuda_gt {
    static constexpr int grid_size_k = 64;
    static constexpr int block_size_k = 1024;
    thrust::device_vector<float> gpu_inputs;
    thrust::device_vector<float> gpu_partial_sums;
    thrust::host_vector<float> cpu_partial_sums;

    cuda_gt(float const *b, float const *e)
        : gpu_inputs(b, e), gpu_partial_sums(grid_size_k), cpu_partial_sums(grid_size_k) {}

    float operator()() {
        // Accumulate partial results, then reduce them further to inputs single scalar.
        auto kernel = kernel_ak == cuda_kernel_t::blocks_k ? &reduce_2step_blocks<block_size_k>
                                                           : &reduce_2step_warps<block_size_k>;
        kernel<<<grid_size_k, block_size_k>>>(gpu_inputs.data().get(), gpu_inputs.size(),
                                              gpu_partial_sums.data().get());
        kernel<<<1, block_size_k>>>(gpu_partial_sums.data().get(), grid_size_k, gpu_partial_sums.data().get());
        cudaDeviceSynchronize();
        cpu_partial_sums = gpu_partial_sums;
        return cpu_partial_sums[0];
    }
};

/**
 * @brief Using Thrust tempaltes library for parallel reductions
 * on Nvidia GPUs, whithout explicitly writing inputs single line of CUDA.
 * https://docs.nvidia.com/cuda/thrust/index.html#reductions
 */
struct cuda_thrust_t {
    thrust::device_vector<float> gpu_inputs;
    cuda_thrust_t(float const *b, float const *e) : gpu_inputs(b, e) {}
    float operator()() const noexcept {
        return thrust::reduce(gpu_inputs.begin(), gpu_inputs.end(), float(0), thrust::plus<float>());
    }
};

} // namespace av