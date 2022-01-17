// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

/**
 *  Most of the algorithms here have follwong properties:
 *  - takes log(n) steps for n input elements,
 *  - uses n threads,
 *  - only works for power-of-2 arrays.
 */
__kernel void reduce_simple(__global float const *inputs, __global float *outputs, ulong const n,
                            __local float *buffer) {
    ulong const idx_global = get_global_id(0);
    ulong const idx_in_block = get_local_id(0);
    buffer[idx_in_block] = (idx_global < n) ? inputs[idx_global] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    ulong block_size = get_local_size(0);
    ulong block_size_half = block_size / 2;
    while (block_size_half > 0) {
        if (idx_in_block < block_size_half) {
            buffer[idx_in_block] += buffer[idx_in_block + block_size_half];
            // Check for uneven block division.
            if ((block_size_half * 2) < block_size) {
                if (idx_in_block == 0)
                    buffer[idx_in_block] += buffer[idx_in_block + (block_size - 1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        block_size = block_size_half;
        block_size_half = block_size / 2;
    }

    if (idx_in_block == 0)
        outputs[get_group_id(0)] = buffer[0];
}

/**
 *  This reduction interleaves which threads are active by using the modulo
 *  operator. This operator is very expensive on GPUs, and the interleaved
 *  inactivity means that no whole warps are active, which is also very
 *  inefficient.
 */
__kernel void reduce_w_modulo(__global float const *inputs, __global float *outputs, ulong const n,
                              __local float *buffer) {
    ulong const idx_in_block = get_local_id(0);
    ulong const idx_global = get_global_id(0);
    buffer[idx_in_block] = (idx_global < n) ? inputs[idx_global] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in the shared memory.
    for (ulong i = 1; i < get_local_size(0); i *= 2) {
        // Modulo arithmetic is slow!
        if ((idx_in_block % (2 * i)) == 0)
            buffer[idx_in_block] += buffer[idx_in_block + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Export this block to global memory.
    if (idx_in_block == 0)
        outputs[get_group_id(0)] = buffer[0];
}

/**
 *  This version uses contiguous threads, but its interleaved
 *  addressing results in many shared memory bank conflicts.
 */
__kernel void reduce_in_shared(__global float const *inputs, __global float *outputs, ulong const n,
                               __local float *buffer) {
    ulong const idx_in_block = get_local_id(0);
    ulong const idx_global = get_global_id(0);
    buffer[idx_in_block] = (idx_global < n) ? inputs[idx_global] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in the shared memory..
    for (ulong i = 1; i < get_local_size(0); i *= 2) {
        ulong const index = 2 * i * idx_in_block;
        if (index < get_local_size(0))
            buffer[index] += buffer[index + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Export this block to global memory.
    if (idx_in_block == 0)
        outputs[get_group_id(0)] = buffer[0];
}

/**
 *  This version uses sequential addressing.
 *  No divergence or bank conflicts.
 */
__kernel void reduce_w_sequential_addressing(__global float const *inputs, __global float *outputs, ulong const n,
                                             __local float *buffer) {
    ulong const idx_in_block = get_local_id(0);
    ulong const idx_global = get_global_id(0);
    buffer[idx_in_block] = (idx_global < n) ? inputs[idx_global] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Do reduction in shared mem.
    for (ulong i = get_local_size(0) / 2; i > 0; i >>= 1) {
        if (idx_in_block < i)
            buffer[idx_in_block] += buffer[idx_in_block + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result for this block to global mem.
    if (idx_in_block == 0)
        outputs[get_group_id(0)] = buffer[0];
}

/**
 *  This version uses n/2 threads - it performs the first level
 *  of reduction when reading from global memory.
 */
__kernel void reduce_bi_step(__global float const *inputs, __global float *outputs, ulong const n,
                             __local float *buffer) {
    ulong const idx_in_block = get_local_id(0);
    ulong const idx_global = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    buffer[idx_in_block] = (idx_global < n) ? inputs[idx_global] : 0;

    // Perform first level of reduction,
    // reading from global memory,
    // writing to shared memory.
    if (idx_global + get_local_size(0) < n)
        buffer[idx_in_block] += inputs[idx_global + get_local_size(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in the shared memory.
    for (ulong i = get_local_size(0) / 2; i > 0; i >>= 1) {
        if (idx_in_block < i)
            buffer[idx_in_block] += buffer[idx_in_block + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Export this block to global memory.
    if (idx_in_block == 0)
        outputs[get_group_id(0)] = buffer[0];
}

/**
 *  Unrolls the last warp to avoid synchronization where
 *  where its not needed.
 */
__kernel void reduce_unrolled(__global float const *inputs, __global float *outputs, ulong const n,
                              __local float *buffer) {
    ulong const idx_in_block = get_local_id(0);
    ulong const idx_global = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    ulong const block_size = get_local_size(0);
    buffer[idx_in_block] = (idx_global < n) ? inputs[idx_global] : 0;

    // Perform first level of reduction,
    // reading from global memory,
    // writing to shared memory.
    if (idx_global + get_local_size(0) < n)
        buffer[idx_in_block] += inputs[idx_global + get_local_size(0)];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in the shared memory.
#pragma unroll 1
    for (ulong i = get_local_size(0) / 2; i > 32; i >>= 1) {
        if (idx_in_block < i)
            buffer[idx_in_block] += buffer[idx_in_block + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (idx_in_block < 32) {
        if (block_size >= 64)
            buffer[idx_in_block] += buffer[idx_in_block + 32];
        if (block_size >= 32)
            buffer[idx_in_block] += buffer[idx_in_block + 16];
        if (block_size >= 16)
            buffer[idx_in_block] += buffer[idx_in_block + 8];
        if (block_size >= 8)
            buffer[idx_in_block] += buffer[idx_in_block + 4];
        if (block_size >= 4)
            buffer[idx_in_block] += buffer[idx_in_block + 2];
        if (block_size >= 2)
            buffer[idx_in_block] += buffer[idx_in_block + 1];
    }

    // Export this block to global memory.
    if (idx_in_block == 0)
        outputs[get_group_id(0)] = buffer[0];
}

/**
 *  This version is completely unrolled. It uses a template parameter to achieve
 *  optimal code for any (power of 2) number of threads. This requires a switch
 *  statement in the host code to handle all the different thread block sizes at
 *  compile time.
 */
__kernel void reduce_unrolled_fully(__global float const *inputs, __global float *outputs, ulong const n,
                                    __local float *buffer) {
    ulong const idx_in_block = get_local_id(0);
    ulong const idx_global = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    ulong const block_size = get_local_size(0);
    buffer[idx_in_block] = (idx_global < n) ? inputs[idx_global] : 0;

    // Perform first level of reduction,
    // reading from global memory,
    // writing to shared memory.
    if (idx_global + block_size < n)
        buffer[idx_in_block] += inputs[idx_global + block_size];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in the shared memory.
    if (block_size >= 512) {
        if (idx_in_block < 256)
            buffer[idx_in_block] += buffer[idx_in_block + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (block_size >= 256) {
        if (idx_in_block < 128)
            buffer[idx_in_block] += buffer[idx_in_block + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (block_size >= 128) {
        if (idx_in_block < 64)
            buffer[idx_in_block] += buffer[idx_in_block + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (idx_in_block < 32) {
        if (block_size >= 64)
            buffer[idx_in_block] += buffer[idx_in_block + 32];
        if (block_size >= 32)
            buffer[idx_in_block] += buffer[idx_in_block + 16];
        if (block_size >= 16)
            buffer[idx_in_block] += buffer[idx_in_block + 8];
        if (block_size >= 8)
            buffer[idx_in_block] += buffer[idx_in_block + 4];
        if (block_size >= 4)
            buffer[idx_in_block] += buffer[idx_in_block + 2];
        if (block_size >= 2)
            buffer[idx_in_block] += buffer[idx_in_block + 1];
    }

    // Export this block to global memory.
    if (idx_in_block == 0)
        outputs[get_group_id(0)] = buffer[0];
}

/**
 *  Uses Brent's Theorem optimization.
 *  This version adds multiple elements per thread sequentially.
 *  This reduces the overall cost of the algorithm while keeping
 *  the work complexity O(n) and the step complexity O(log n).
 */
__kernel void reduce_w_brents_theorem(__global float const *inputs, __global float *outputs, ulong const n,
                                      __local float *buffer) {
    ulong const block_size = get_local_size(0);
    ulong const idx_in_block = get_local_id(0);
    ulong idx_global = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    ulong const grid_size = block_size * 2 * get_num_groups(0);
    buffer[idx_in_block] = (idx_global < n) ? inputs[idx_global] : 0;

    // We reduce multiple elements per thread.
    // The number is determined by the number of active thread blocks (via gridDim).
    // More blocks will result in a larger grid_size and therefore fewer elements per thread.
    while (idx_global < n) {
        buffer[idx_in_block] += inputs[idx_global];
        // Ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays.
        if (idx_global + block_size < n)
            buffer[idx_in_block] += inputs[idx_global + block_size];
        idx_global += grid_size;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in the shared memory.
    if (block_size >= 512) {
        if (idx_in_block < 256)
            buffer[idx_in_block] += buffer[idx_in_block + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (block_size >= 256) {
        if (idx_in_block < 128)
            buffer[idx_in_block] += buffer[idx_in_block + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (block_size >= 128) {
        if (idx_in_block < 64)
            buffer[idx_in_block] += buffer[idx_in_block + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (idx_in_block < 32) {
        if (block_size >= 64)
            buffer[idx_in_block] += buffer[idx_in_block + 32];
        if (block_size >= 32)
            buffer[idx_in_block] += buffer[idx_in_block + 16];
        if (block_size >= 16)
            buffer[idx_in_block] += buffer[idx_in_block + 8];
        if (block_size >= 8)
            buffer[idx_in_block] += buffer[idx_in_block + 4];
        if (block_size >= 4)
            buffer[idx_in_block] += buffer[idx_in_block + 2];
        if (block_size >= 2)
            buffer[idx_in_block] += buffer[idx_in_block + 1];
    }

    // Export this block to global memory.
    if (idx_in_block == 0)
        outputs[get_group_id(0)] = buffer[0];
}
