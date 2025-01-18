/**
 *  @date 04/09/2019
 *  @file reduce_cublas.cuh
 *  @brief cuBLAS-based reductions
 *  @author Ash Vardanian
 */
#pragma once
#include <cublas_api.h>
#include <cuda_runtime_api.h>
#include <mma.h> // `wmma::`

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

using namespace nvcuda;

namespace ashvardanian::reduce {

struct cuda_blas_t {

    // We review this input array as a wide 2D matrix.
    // Every group of threads operates on its set of rows.
    // Accumulating values in each row.
    // If each block of threads operates on X rows,
    // we will end up with X partial sums.
    static constexpr unsigned int tile_side_k = 16;

    thrust::device_vector<float> gpu_inputs;
    thrust::device_vector<float> ones_matrix;
    thrust::device_vector<float> product_matrix;

    cuda_blas_t(float const *b, float const *e)
        : gpu_inputs(b, e), ones_matrix(b - e), product_matrix(tile_side_k * tile_side_k) {}

    float operator()() {

        int a_rows_num = tile_side_k;
        int dim = gpu_inputs.size() / tile_side_k;
        int b_rows_num = dim;

        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.f;
        float beta = 0.f;
        auto a_device = gpu_inputs.data().get();
        auto b_device = ones_matrix.data().get();
        auto c_device = product_matrix.data().get();

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, b_rows_num, a_rows_num, dim, &alpha, b_device, b_rows_num,
                    a_device, a_rows_num, &beta, c_device, b_rows_num);

        cudaDeviceSynchronize();

        // Accumulate partial results...
        cu_reduce_tensors<<<count_blocks_k, rows_per_block_k>>>(gpu_inputs.data().get(), entries_total,
                                                                sums_per_row.data().get(), entries_per_row);

        // Sync all queues and fetch results
        cudaDeviceSynchronize();
        return thrust::reduce(sums_per_row.begin(), sums_per_row.end(), float(0), thrust::plus<float>());
    }
};

__global__ void cu_reduce_tensors( //
    float const *inputs, unsigned int input_size, float *sums_per_row, unsigned int columns) {

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

} // namespace ashvardanian::reduce