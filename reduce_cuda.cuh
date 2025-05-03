/**
 *  @brief Pure CUDA, CUB, and Thrust-based reductions
 *  @file reduce_cuda.cuh
 *  @author Ash Vardanian
 *  @date 04/09/2019
 */
#pragma once
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <cub/cub.cuh>

namespace ashvardanian {

std::size_t cuda_device_count() noexcept {
    int count;
    auto error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) return 0;
    return static_cast<std::size_t>(count);
}

__global__ void cuda_blocks_kernel(float const *inputs, std::size_t input_size, float *outputs);

/**
 *  @brief CUDA-based reduction using `shared` memory for partial sums.
 *
 *  This kernels uses the default CUDA "blocks" and "threads" semantics for
 *  scheduling. Threads within the same block accumulate different parts of
 *  the input into a `__shared__` buffer and then perform a tree-like reduction
 *  within it.
 *
 *  The NVIDIA H100 GPU supports shared memory capacities of 0, 8, 16, 32, 64,
 *  100, 132, 164, 196 and 228 KB per SM. CUDA reserves 1 KB of shared memory
 *  per thread block. Hence, the H100 GPU enables a single thread block to
 *  address up to 227 KB of shared memory.
 *
 *  @see Hopper Tuning Guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
 *  @see https://stackoverflow.com/q/12733084
 *  @see https://stackoverflow.com/q/44278317
 */
class cuda_blocks_t {
    unsigned int thread_blocks_;
    unsigned int threads_per_block_;

    thrust::device_vector<float> gpu_inputs_;
    mutable thrust::device_vector<float> gpu_partial_sums_;
    mutable thrust::host_vector<float> cpu_partial_sums_;

  public:
    /**
     *  @param threads_per_block Should be 32 or its multiple to saturate all cores in a warp.
     *  @param thread_blocks Should be greater or equal to the number of Streaming Multiprocessors
     *  on the current GPU. 256 is a good default.
     */
    cuda_blocks_t(float const *b, float const *e, std::size_t threads_per_block = 64, std::size_t thread_blocks = 256)
        : thread_blocks_(thread_blocks), threads_per_block_(threads_per_block), gpu_inputs_(b, e),
          gpu_partial_sums_(thread_blocks), cpu_partial_sums_(1) {}

    float operator()() const {

        // Accumulate partial results...
        unsigned int shared_memory = threads_per_block_ * sizeof(float);
        cuda_blocks_kernel<<<thread_blocks_, threads_per_block_, shared_memory>>>( //
            gpu_inputs_.data().get(), gpu_inputs_.size(), gpu_partial_sums_.data().get());

        // Then reduce them further to inputs single scalar
        shared_memory = threads_per_block_ * sizeof(float);
        cuda_blocks_kernel<<<1, threads_per_block_, shared_memory>>>( //
            gpu_partial_sums_.data().get(), thread_blocks_, gpu_partial_sums_.data().get());

        // Sync all queues and fetch results
        cudaDeviceSynchronize();
        cpu_partial_sums_ = gpu_partial_sums_;
        return cpu_partial_sums_[0];
    }
};

__global__ void cuda_blocks_kernel(float const *inputs, std::size_t input_size, float *outputs) {
    extern __shared__ float shared[]; //? This will be sized at runtime using the third kernel argument

    unsigned int const total_threads = blockDim.x * gridDim.x;
    unsigned int const thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int const thread_in_block = threadIdx.x;

    // This specific thread will accumulate `input_size / total_threads` elements
    // starting with `thread_id` entry and walking forward with a `total_threads`
    // stride until the end is reached.
    float strided_sum = 0;
    for (std::size_t i = thread_id; i < input_size; i += total_threads) strided_sum += inputs[i];
    shared[thread_in_block] = strided_sum;
    __syncthreads();

    // Accumulate all entries within current block to one using a tree-like reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_in_block < s) shared[thread_in_block] += shared[thread_in_block + s];
        __syncthreads();
    }

    // Export only the first result in each block
    if (thread_in_block == 0) outputs[blockIdx.x] = shared[0];
}

__global__ void cuda_warps_kernel(float const *inputs, std::size_t input_size, float *outputs);

/**
 *  @brief CUDA-based reductions using fast warp shuffles on Kepler and newer architectures.
 *
 *  @see Faster Parallel Reductions on Kepler: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 *  @see Using CUDA Warp-Level Primitives: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
 *  @see Do The Kepler Shuffle: https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-shuffle/
 */
class cuda_warps_t {

    unsigned int thread_blocks_;

    thrust::device_vector<float> gpu_inputs_;
    mutable thrust::device_vector<float> gpu_partial_sums_;
    mutable thrust::host_vector<float> cpu_partial_sums_;

  public:
    cuda_warps_t() = default;
    cuda_warps_t(float const *b, float const *e, std::size_t thread_blocks = 256)
        : thread_blocks_(thread_blocks), gpu_inputs_(b, e), gpu_partial_sums_(thread_blocks), cpu_partial_sums_(1) {}

    float operator()() const {
        constexpr unsigned int threads_per_block = 32;

        // Accumulate partial results...
        cuda_warps_kernel<<<thread_blocks_, threads_per_block>>>( //
            gpu_inputs_.data().get(), gpu_inputs_.size(), gpu_partial_sums_.data().get());

        // Then reduce them further to inputs single scalar
        cuda_warps_kernel<<<1, threads_per_block>>>( //
            gpu_partial_sums_.data().get(), thread_blocks_, gpu_partial_sums_.data().get());

        // Sync all queues and fetch results
        cudaDeviceSynchronize();
        cpu_partial_sums_ = gpu_partial_sums_;
        return cpu_partial_sums_[0];
    }
};

__inline__ __device__ float cuda_warp_reduce(float val) noexcept {
    // The `__shfl_down_sync` replaces `__shfl_down`
    // https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__global__ void cuda_warps_kernel(float const *inputs, std::size_t input_size, float *outputs) {
    unsigned int const total_threads = blockDim.x * gridDim.x;
    unsigned int const thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int const thread_in_block = threadIdx.x;

    // This specific thread will accumulate `input_size / total_threads` elements
    // starting with `thread_id` entry and walking forward with a `total_threads`
    // stride until the end is reached.
    float sum = 0;
    for (std::size_t i = thread_id; i < input_size; i += total_threads) sum += inputs[i];

    // Shared memory for 32 partial sums
    __shared__ float shared[32];
    unsigned int lane = thread_in_block % warpSize; // In our case, generally equal to `threadIdx.x
    unsigned int wid = thread_in_block / warpSize;  // In our case, generally equal to 0

    // Each warp performs partial reduction
    sum = cuda_warp_reduce(sum);

    // Write reduced value to shared memory
    if (lane == 0) shared[wid] = sum;

    // Wait for all partial reductions
    __syncthreads();

    // Read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    // Final reduce within first warp
    if (wid == 0) sum = cuda_warp_reduce(sum);
    if (threadIdx.x == 0) outputs[blockIdx.x] = sum;
}

/**
 *  @brief Uses CUDA @b Thrust library for parallel reductions on Nvidia GPUs.
 *  @see Thrust Reductions Docs: https://docs.nvidia.com/cuda/thrust/index.html#reductions
 */
class cuda_thrust_t {
    thrust::device_vector<float> gpu_inputs_;

  public:
    cuda_thrust_t() = default;
    cuda_thrust_t(float const *b, float const *e) : gpu_inputs_(b, e) {}
    float operator()() const {
        return thrust::reduce(gpu_inputs_.begin(), gpu_inputs_.end(), float(0), thrust::plus<float>());
    }
};

/**
 *  @brief Uses CUDA @b Thrust library for parallel reductions on Nvidia GPUs, interleaving
 *         additions and Fused Multiply-Add @b (FMA) instructions.
 *  @see Thrust Reductions Docs: https://docs.nvidia.com/cuda/thrust/index.html#reductions
 */
class cuda_thrust_fma_t {
    thrust::device_vector<float> gpu_inputs_;

  public:
    struct pair_t {
        float even = 0;
        float odd = 0;
    };
    struct interleaving_add_t {
        __device__ pair_t operator()(pair_t const &a, pair_t const &b) const noexcept {
            return {a.even + b.even, fmaf(a.odd, 1.f, b.odd)};
        }
    };

    cuda_thrust_fma_t() = default;
    cuda_thrust_fma_t(float const *b, float const *e) : gpu_inputs_(b, e) {}
    float operator()() const {
        auto floats_data = gpu_inputs_.data().get();
        auto pairs_data = thrust::device_pointer_cast<pair_t const>(reinterpret_cast<pair_t const *>(floats_data));
        auto pair = thrust::reduce(pairs_data, pairs_data + gpu_inputs_.size() / 2, pair_t {}, interleaving_add_t {});
        return pair.even + pair.odd;
    }
};

/**
 *  @brief Uses @b CUB on Nvidia GPUs for faster global reductions.
 *  @see Device-wide Reduction Docs:
 *  https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html#aa4adabeb841b852a7a5ecf4f99a2daeb
 *  @see 64-bit indexing issues: https://github.com/NVIDIA/cccl/issues/744
 */
class cuda_cub_t {
    thrust::device_vector<float> gpu_inputs_;
    mutable thrust::device_vector<char> temporary_;
    mutable thrust::device_vector<float> gpu_sums_;
    mutable thrust::host_vector<float> cpu_sums_;

  public:
    cuda_cub_t() = default;
    cuda_cub_t(float const *b, float const *e) : gpu_inputs_(b, e), gpu_sums_(1), cpu_sums_(1) {
        // CUB can't handle large arrays with over 2 billion elements!
        assert(gpu_inputs_.size() < std::numeric_limits<int>::max());

        // Determine temporary device storage requirements
        auto num_items = static_cast<int>(gpu_inputs_.size());
        auto gpu_inputs_ptr = gpu_inputs_.data().get();
        auto gpu_sums_ptr = gpu_sums_.data().get();
        void *temporary_ptr = nullptr;
        std::size_t temporary_bytes = 0;
        cudaError_t error =
            cub::DeviceReduce::Sum(temporary_ptr, temporary_bytes, gpu_inputs_ptr, gpu_sums_ptr, num_items);
        assert(error == cudaSuccess);
        assert(temporary_bytes > 0);
        temporary_.resize(temporary_bytes);
    }

    float operator()() const {

        auto num_items = static_cast<int>(gpu_inputs_.size());
        float const *gpu_inputs_ptr = gpu_inputs_.data().get();
        float *gpu_sums_ptr = gpu_sums_.data().get();
        void *temporary_ptr = temporary_.data().get();
        std::size_t temporary_bytes = temporary_.size(); //! Must be mutable for CUB

        cudaError_t error =
            cub::DeviceReduce::Sum(temporary_ptr, temporary_bytes, gpu_inputs_ptr, gpu_sums_ptr, num_items);
        assert(error == cudaSuccess);
        cudaDeviceSynchronize();

        cpu_sums_ = gpu_sums_;
        return cpu_sums_[0];
    }
};

} // namespace ashvardanian