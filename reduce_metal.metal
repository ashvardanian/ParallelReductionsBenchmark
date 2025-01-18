#include <metal_stdlib>

using namespace metal;

/**
 *  @brief Phase 1: Compute partial sums of the input array.
 *
 *  Each threadgroup reduces its subset of the input into a single partial sum.
 *  That partial sum is stored in `partials[group_id]`.
 */
kernel void reduce_phase1( //
    device float const *inputs [[buffer(0)]], device float *partials [[buffer(1)]],
    constant uint &input_size [[buffer(2)]], uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]], uint group_id [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]]) {

    // Each thread accumulates from the global array in steps of (total # of threads)
    float sum_local = 0.0;
    uint total_threads = grid_size; // total threads across all threadgroups

    for (uint i = gid; i < input_size; i += total_threads)
        sum_local += inputs[i];

    // Use threadgroup shared memory to do an in-group tree reduction
    threadgroup float scratch[256]; // Example: 256 threads per group max
    scratch[tid] = sum_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within the threadgroup
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The first thread in each group writes out the partial sum
    if (tid == 0)
        partials[group_id] = scratch[0];
}

/**
 *  @brief Phase 2: Sum up the threadgroup partial sums into a single scalar.
 */
kernel void reduce_phase2( //
    device float const *partials [[buffer(0)]], device float *outputs [[buffer(1)]],
    constant uint &num_groups [[buffer(2)]], uint gid [[thread_position_in_grid]]) {

    // We'll just have (num_groups) partial sums to add
    float sum_final = 0.0;
    for (uint i = gid; i < num_groups; i += grid_size)
        sum_final += partials[i];

    // Only one thread needs to write out the final scalar
    if (gid == 0)
        outputs[0] = sum_final;
}
