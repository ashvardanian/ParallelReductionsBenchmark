#pragma once
#include <cuda_runtime_api.h>
#include <mma.h> // `wmma::`

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

using namespace nvcuda;

namespace av {

struct cuda_base_t {
    static constexpr int max_block_size_k = 1024;
    static constexpr int threads = 512;

    int blocks = max_block_size_k;
    thrust::device_vector<float> gpu_inputs;
    thrust::device_vector<float> gpu_partial_sums;
    thrust::host_vector<float> cpu_partial_sums;

    cuda_base_t(float const *b, float const *e)
        : blocks(std::min<int>(((e - b) + threads - 1) / threads, max_block_size_k)), gpu_inputs(b, e),
          gpu_partial_sums(max_block_size_k), cpu_partial_sums(max_block_size_k) {}
};

__global__ void cu_recude_blocks(const float *inputs, unsigned int input_size, float *outputs) {
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

/**
 * @brief Uses global memory for partial sums.
 *
 * Reading:
 * https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-
 * https://stackoverflow.com/inputs/25584577
 * https://stackoverflow.com/q/12733084
 * https://stackoverflow.com/q/44278317
 */
struct cuda_blocks_t : public cuda_base_t {

    cuda_blocks_t(float const *b, float const *e) : cuda_base_t(b, e) {}

    float operator()() {

        // Accumulate partial results...
        int shared_memory = threads * sizeof(float);
        cu_recude_blocks<<<blocks, threads, shared_memory>>>(gpu_inputs.data().get(), gpu_inputs.size(),
                                                             gpu_partial_sums.data().get());

        // Then reduce them further to inputs single scalar
        shared_memory = max_block_size_k * sizeof(float);
        cu_recude_blocks<<<1, max_block_size_k, shared_memory>>>(gpu_partial_sums.data().get(), blocks,
                                                                 gpu_partial_sums.data().get());

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

/**
 * @brief Uses warp shuffles on Kepler and newer architectures.
 * Source: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 * More reading:
 * https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
 * https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-shuffle/
 */
struct cuda_warps_t : public cuda_base_t {

    cuda_warps_t(float const *b, float const *e) : cuda_base_t(b, e) {}

    float operator()() {

        // Accumulate partial results...
        cu_reduce_warps<<<blocks, threads>>>(gpu_inputs.data().get(), gpu_inputs.size(), gpu_partial_sums.data().get());

        // Then reduce them further to inputs single scalar
        cu_reduce_warps<<<1, max_block_size_k>>>(gpu_partial_sums.data().get(), blocks, gpu_partial_sums.data().get());

        // Sync all queues and fetch results
        cudaDeviceSynchronize();
        cpu_partial_sums = gpu_partial_sums;
        return cpu_partial_sums[0];
    }
};

__global__ void cu_reduce_tensors(float const *inputs, unsigned int input_size, float *sums_per_row,
                                  unsigned int columns) {

    // Tile using a 2D grid
    unsigned int first_row_of_block = blockIdx.x * blockDim.x;
    unsigned int thread_within_block = threadIdx.x;
    unsigned int constexpr side_k = 16;
    __shared__ float shared[side_k][side_k];

    // Fill shared memory with the pattern we are going to use as a multiplier
    shared[thread_within_block][0] = 1;
    for (unsigned int i = 1; i != side_k; ++i)
        shared[thread_within_block][i] = 0;

    // Declare the fragments
    // The only tile size currently supported is 16.
    using tf32_t = wmma::precision::tf32;
    wmma::fragment<wmma::matrix_a, 16, 16, 8, tf32_t, wmma::row_major> inputs_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, tf32_t, wmma::row_major> multiplier_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> rows_block_sums_frag;

    // The accumulator fragment will only have 1 column worth of relevant data.
    // Only that first column will be exported to `sums_per_row`.
    wmma::fill_fragment(rows_block_sums_frag, 0.0f);
    wmma::load_matrix_sync(multiplier_frag, &shared[0][0], side_k);

    // Loop over k
    for (int i = 0; i < columns; i += side_k) {
        // `mptr` must be a 256-bit aligned pointer pointing to the first element of the matrix in memory.
        auto input = inputs + first_row_of_block * columns;

        // `ldm` describes the stride in elements between consecutive rows (for row major layout) or
        // columns (for column major layout) and must be a multiple of 8 for `__half` element type or
        // multiple of 4 for `float` element type.
        wmma::load_matrix_sync(inputs_frag, input, side_k);

        // Perform the matrix multiplication
        wmma::mma_sync(rows_block_sums_frag, inputs_frag, multiplier_frag, rows_block_sums_frag);
    }

    // Temporarily dump back to shared memory and export one element into global.
    wmma::store_matrix_sync(&shared[0][0], rows_block_sums_frag, side_k, wmma::mem_row_major);
    sums_per_row[first_row_of_block + thread_within_block] = shared[thread_within_block][0];
}

/**
 * @brief Uses Nvidia Tensor Cores for reductions.
 *
 * Supported types and tile sizes:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-type-sizes
 * https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
 * https://www.microway.com/knowledge-center-articles/in-depth-comparison-of-nvidia-ampere-gpu-accelerators/
 *
 * Real (f32) matrices are only available in 16x16x8 form.
 * https://developer.nvidia.com/blog/using-tensor-cores-in-cuda-fortran/
 *
 * Reading:
 * https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
 * https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu
 */
struct cuda_tensors_t {

    // We review this input array as a wide 2D matrix.
    // Every group of threads operates on its set of rows.
    // Accumulating values in each row.
    // If each block of threads operates on X rows,
    // we will end up with X partial sums.
    static constexpr unsigned int rows_per_block_k = 16;
    static constexpr unsigned int count_rows_k = 1024;
    static constexpr unsigned int count_blocks_k = count_rows_k / rows_per_block_k;

    thrust::device_vector<float> gpu_inputs;
    thrust::device_vector<float> sums_per_row;

    cuda_tensors_t(float const *b, float const *e) : gpu_inputs(b, e), sums_per_row(count_rows_k) {}

    float operator()() {

        unsigned int const entries_total = gpu_inputs.size();
        unsigned int const entries_per_row = entries_total / count_rows_k;

        // Accumulate partial results...
        cu_reduce_tensors<<<count_blocks_k, rows_per_block_k>>>(gpu_inputs.data().get(), entries_total,
                                                                sums_per_row.data().get(), entries_per_row);

        // Sync all queues and fetch results
        cudaDeviceSynchronize();
        return thrust::reduce(sums_per_row.begin(), sums_per_row.end(), float(0), thrust::plus<float>());
    }
};

inline static size_t cuda_device_count() {
    int count;
    auto error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess)
        return 0;
    return static_cast<size_t>(count);
}

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