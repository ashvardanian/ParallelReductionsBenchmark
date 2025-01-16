#pragma once
#include <cuda_runtime_api.h>
#include <mma.h> // `wmma::`

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

using namespace nvcuda;

namespace ashvardanian::reduce {

/// Base class for CUDA-based reductions.
struct cuda_base_t {
    static constexpr int max_block_size_k = 1024;
    static constexpr int threads_k = 512;

    int blocks = max_block_size_k;
    thrust::device_vector<float> gpu_inputs;
    thrust::device_vector<float> gpu_partial_sums;
    thrust::host_vector<float> cpu_partial_sums;

    cuda_base_t(float const *b, float const *e)
        : blocks(std::min<int>(((e - b) + threads_k - 1) / threads_k, max_block_size_k)), gpu_inputs(b, e),
          gpu_partial_sums(max_block_size_k), cpu_partial_sums(max_block_size_k) {}
};

__global__ void cu_reduce_blocks(float const *inputs, unsigned int input_size, float *outputs) {
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

/// CUDA-based reduction using slow `global` memory for partial sums.
///
/// https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-
/// https://stackoverflow.com/inputs/25584577
/// https://stackoverflow.com/q/12733084
/// https://stackoverflow.com/q/44278317
struct cuda_blocks_t : public cuda_base_t {

    cuda_blocks_t(float const *b, float const *e) : cuda_base_t(b, e) {}

    float operator()() {

        // Accumulate partial results...
        int shared_memory = threads_k * sizeof(float);
        cu_reduce_blocks<<<blocks, threads_k, shared_memory>>>( //
            gpu_inputs.data().get(), gpu_inputs.size(), gpu_partial_sums.data().get());

        // Then reduce them further to inputs single scalar
        shared_memory = max_block_size_k * sizeof(float);
        cu_reduce_blocks<<<1, max_block_size_k, shared_memory>>>( //
            gpu_partial_sums.data().get(), blocks, gpu_partial_sums.data().get());

        // Sync all queues and fetch results
        cudaDeviceSynchronize();
        cpu_partial_sums = gpu_partial_sums;
        return cpu_partial_sums[0];
    }
};

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

__global__ void cu_reduce_warps(float const *inputs, unsigned int input_size, float *outputs) {
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

/// CUDA-based reductions using fast warp shuffles on Kepler and newer architectures.
/// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
/// https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
/// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-shuffle/
struct cuda_warps_t : public cuda_base_t {

    cuda_warps_t(float const *b, float const *e) : cuda_base_t(b, e) {}

    float operator()() {

        // Accumulate partial results...
        cu_reduce_warps<<<blocks, threads_k>>>( //
            gpu_inputs.data().get(), gpu_inputs.size(), gpu_partial_sums.data().get());

        // Then reduce them further to inputs single scalar
        cu_reduce_warps<<<1, max_block_size_k>>>( //
            gpu_partial_sums.data().get(), blocks, gpu_partial_sums.data().get());

        // Sync all queues and fetch results
        cudaDeviceSynchronize();
        cpu_partial_sums = gpu_partial_sums;
        return cpu_partial_sums[0];
    }
};

inline static size_t cuda_device_count() {
    int count;
    auto error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess)
        return 0;
    return static_cast<size_t>(count);
}

/// Uses CUDA Thrust library for parallel reductions on Nvidia GPUs.
/// https://docs.nvidia.com/cuda/thrust/index.html#reductions
struct cuda_thrust_t {
    thrust::device_vector<float> gpu_inputs;
    cuda_thrust_t(float const *b, float const *e) : gpu_inputs(b, e) {}
    float operator()() const {
        return thrust::reduce(gpu_inputs.begin(), gpu_inputs.end(), float(0), thrust::plus<float>());
    }
};

/// Uses CUB on Nvidia GPUs for faster global reductions.
/// https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html#aa4adabeb841b852a7a5ecf4f99a2daeb
struct cuda_cub_t {
    thrust::device_vector<float> gpu_inputs;
    thrust::device_vector<uint8_t> temporary;
    thrust::device_vector<float> gpu_sums;
    thrust::host_vector<float> cpu_sums;

    cuda_cub_t(float const *b, float const *e) : gpu_inputs(b, e), gpu_sums(1), cpu_sums(1) {
        // CUB can't handle large arrays with over 2 billion elements!
        assert(gpu_inputs.size() < std::numeric_limits<int>::max());
    }

    float operator()() {

        auto num_items = static_cast<int>(gpu_inputs.size());
        auto d_in = gpu_inputs.data().get();
        auto d_out = gpu_sums.data().get();
        cudaError_t error;

        // Determine temporary device storage requirements
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        error = cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
        assert(error == cudaSuccess);
        assert(temp_storage_bytes > 0);

        // Allocate temporary storage, if needed
        if (temp_storage_bytes > temporary.size())
            temporary.resize(temp_storage_bytes);
        d_temp_storage = temporary.data().get();

        error = cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
        assert(error == cudaSuccess);
        cudaDeviceSynchronize();

        cpu_sums = gpu_sums;
        return cpu_sums[0];
    }
};

} // namespace ashvardanian::reduce