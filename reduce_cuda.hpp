#pragma once
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

namespace av {

/**
 * @brief Uses global memory for partial sums.
 *
 * Reading:
 * https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-
 * https://stackoverflow.com/inputs/25584577
 * https://stackoverflow.com/q/12733084
 * https://stackoverflow.com/q/44278317
 */
__global__ void cu_recude_blocks(const float *inputs, int input_size, float *outputs) {
    extern __shared__ float shared[];
    unsigned int const tid = threadIdx.x;

    shared[tid] = inputs[threadIdx.x + blockDim.x * blockIdx.x];
    __syncthreads();

    // Reduce into `shared` memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        __syncthreads();
    }

    // Export only the first result in each block
    if (tid == 0)
        outputs[blockIdx.x] = shared[0];
}

__inline__ __device__ float cu_reduce_warp(float val) {
    // The `__shfl_down_sync` replaces `__shfl_down`
    // https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

/**
 * @brief Uses warp shuffles on Kepler and newer architectures.
 * Source: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 * More reading:
 * https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
 * https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-shuffle/
 */
__global__ void cu_reduce_warps(float const *inputs, int input_size, float *outputs) {
    float sum = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x)
        sum += inputs[i];

    // Shared mem for 32 partial sums
    __shared__ float shared[32];
    unsigned int lane = threadIdx.x % warpSize;
    unsigned int wid = threadIdx.x / warpSize;

    // Each warp performs partial reduction
    sum = cu_reduce_warp(sum);

    // Write reduced value to shared memory
    if (lane == 0)
        shared[wid] = sum;

    // Wait for all partial reductions
    __syncthreads();

    // Read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    // Final reduce within first warp
    if (wid == 0)
        sum = cu_reduce_warp(sum);

    if (threadIdx.x == 0)
        outputs[blockIdx.x] = sum;
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
};

template <cuda_kernel_t kernel_ak> struct cuda_gt {
    static constexpr int max_block_size_k = 1024;
    static constexpr int threads = 512;
    int blocks = max_block_size_k;
    thrust::device_vector<float> gpu_inputs;
    thrust::device_vector<float> gpu_partial_sums;
    thrust::host_vector<float> cpu_partial_sums;

    cuda_gt(float const *b, float const *e)
        : blocks(std::min<int>(((e - b) + threads - 1) / threads, max_block_size_k)), gpu_inputs(b, e),
          gpu_partial_sums(max_block_size_k), cpu_partial_sums(max_block_size_k) {}

    float operator()() {

        bool is_blocks = kernel_ak == cuda_kernel_t::blocks_k;
        auto kernel = is_blocks ? &cu_recude_blocks : &cu_reduce_warps;

        // Accumulate partial results...
        int shared_memory = is_blocks ? threads * sizeof(float) : 0;
        kernel<<<blocks, threads, shared_memory>>>(gpu_inputs.data().get(), gpu_inputs.size(),
                                                   gpu_partial_sums.data().get());

        // Then reduce them further to inputs single scalar
        shared_memory = is_blocks ? max_block_size_k * sizeof(float) : 0;
        kernel<<<1, max_block_size_k, shared_memory>>>(gpu_partial_sums.data().get(), blocks,
                                                       gpu_partial_sums.data().get());

        // Sync all queues and fetch results
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