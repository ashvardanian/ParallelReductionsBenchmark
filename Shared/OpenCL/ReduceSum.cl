// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

typedef float sReal;
typedef int sIdx;
typedef size_t bSize;

/**
 *  Most of the algorithms here have follwong properties:
 *  - takes log(xLen) steps for xLen input elements,
 *  - uses xLen threads,
 *  - only works for power-of-2 arrays.
 */

__kernel
void gReduceSimple(__global sReal const * xArr, __global sReal * yArr,
                   sIdx const xLen, __local sReal * mBuffer) {
    sIdx const lIdxGlobal = get_global_id(0);
    sIdx const lIdxInBlock = get_local_id(0);
    mBuffer[lIdxInBlock] = (lIdxGlobal < xLen) ? xArr[lIdxGlobal] : 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    sIdx lBlockSize = get_local_size(0);
    sIdx lBlockSizeHalf = lBlockSize / 2;
    while (lBlockSizeHalf > 0) {
        if (lIdxInBlock < lBlockSizeHalf) {
            mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + lBlockSizeHalf];
            // Check for uneven block division.
            if ((lBlockSizeHalf * 2) < lBlockSize) {
                if (lIdxInBlock == 0)
                    mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + (lBlockSize - 1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        lBlockSize = lBlockSizeHalf;
        lBlockSizeHalf = lBlockSize / 2;
    }
    
    if (lIdxInBlock == 0) yArr[get_group_id(0)] = mBuffer[0];
}

/**
 *  This reduction interleaves which threads are active by using the modulo
 *  operator. This operator is very expensive on GPUs, and the interleaved
 *  inactivity means that no whole warps are active, which is also very
 *  inefficient.
 */
__kernel
void gReduceWModulo(__global sReal const * xArr, __global sReal * yArr,
                    sIdx const xLen, __local sReal * mBuffer) {
    sIdx const lIdxInBlock = get_local_id(0);
    sIdx const lIdxGlobal = get_global_id(0);
    mBuffer[lIdxInBlock] = (lIdxGlobal < xLen) ? xArr[lIdxGlobal] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform reduction in the shared memory.
    for(sIdx lTemp = 1; lTemp < get_local_size(0); lTemp *= 2) {
        // modulo arithmetic is slow!
        if ((lIdxInBlock % (2*lTemp)) == 0) {
            mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + lTemp];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Export this block to global memory.
    if (lIdxInBlock == 0) yArr[get_group_id(0)] = mBuffer[0];
}


/**
 *  This version uses contiguous threads, but its interleaved
 *  addressing results in many shared memory bank conflicts.
 */
__kernel
void gReduceInSharedMem(__global sReal const * xArr, __global sReal * yArr,
                        sIdx const xLen, __local sReal * mBuffer) {
    sIdx const lIdxInBlock = get_local_id(0);
    sIdx const lIdxGlobal = get_global_id(0);
    mBuffer[lIdxInBlock] = (lIdxGlobal < xLen) ? xArr[lIdxGlobal] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform reduction in the shared memory..
    for (sIdx lTemp = 1; lTemp < get_local_size(0); lTemp *= 2)  {
        sIdx const index = 2 * lTemp * lIdxInBlock;
        if (index < get_local_size(0))
            mBuffer[index] += mBuffer[index + lTemp];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Export this block to global memory.
    if (lIdxInBlock == 0) yArr[get_group_id(0)] = mBuffer[0];
}

/**
 *  This version uses sequential addressing.
 *  No divergence or bank conflicts.
 */
__kernel
void gReduceWSequentialAddressing(__global sReal const * xArr, __global sReal * yArr,
                                  sIdx const xLen, __local sReal * mBuffer) {
    sIdx const lIdxInBlock = get_local_id(0);
    sIdx const lIdxGlobal = get_global_id(0);
    mBuffer[lIdxInBlock] = (lIdxGlobal < xLen) ? xArr[lIdxGlobal] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Do reduction in shared mem.
    for(sIdx lTemp = get_local_size(0) / 2; lTemp > 0; lTemp >>= 1)  {
        if (lIdxInBlock < lTemp)
            mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + lTemp];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result for this block to global mem.
    if (lIdxInBlock == 0) yArr[get_group_id(0)] = mBuffer[0];
}

/**
 *  This version uses xLen/2 threads - it performs the first level
 *  of reduction when reading from global memory.
 */
__kernel
void gReduce2Steps(__global sReal const * xArr, __global sReal * yArr,
                   sIdx const xLen, __local sReal * mBuffer) {
    sIdx const lIdxInBlock = get_local_id(0);
    sIdx const lIdxGlobal = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    mBuffer[lIdxInBlock] = (lIdxGlobal < xLen) ? xArr[lIdxGlobal] : 0;
    
    // Perform first level of reduction,
    // reading from global memory,
    // writing to shared memory.
    if (lIdxGlobal + get_local_size(0) < xLen)
        mBuffer[lIdxInBlock] += xArr[lIdxGlobal + get_local_size(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform reduction in the shared memory.
    for (sIdx lTemp = get_local_size(0) / 2; lTemp > 0; lTemp >>= 1)  {
        if (lIdxInBlock < lTemp)
            mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + lTemp];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Export this block to global memory.
    if (lIdxInBlock == 0) yArr[get_group_id(0)] = mBuffer[0];
}

/**
 *  Unrolls the last warp to avoid synchronization where
 *  where its not needed.
 */
__kernel
void gReduceUnrolled(__global sReal const * xArr, __global sReal * yArr,
                     sIdx const xLen, __local sReal * mBuffer) {
    sIdx const lIdxInBlock = get_local_id(0);
    sIdx const lIdxGlobal = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    sIdx const lBlockSize = get_local_size(0);
    mBuffer[lIdxInBlock] = (lIdxGlobal < xLen) ? xArr[lIdxGlobal] : 0;
    
    // Perform first level of reduction,
    // reading from global memory,
    // writing to shared memory.
    if (lIdxGlobal + get_local_size(0) < xLen)
        mBuffer[lIdxInBlock] += xArr[lIdxGlobal+get_local_size(0)];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform reduction in the shared memory.
#pragma unroll 1
    for (sIdx lTemp = get_local_size(0) / 2; lTemp > 32; lTemp >>= 1)  {
        if (lIdxInBlock < lTemp)
            mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + lTemp];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lIdxInBlock < 32) {
        if (lBlockSize >=  64) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + 32]; }
        if (lBlockSize >=  32) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + 16]; }
        if (lBlockSize >=  16) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  8]; }
        if (lBlockSize >=   8) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  4]; }
        if (lBlockSize >=   4) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  2]; }
        if (lBlockSize >=   2) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  1]; }
    }
    
    // Export this block to global memory.
    if (lIdxInBlock == 0) yArr[get_group_id(0)] = mBuffer[0];
}

/**
 *  This version is completely unrolled.  It uses a template parameter to achieve
 *  optimal code for any (power of 2) number of threads.  This requires a switch
 *  statement in the host code to handle all the different thread block sizes at
 *  compile time.
 */
__kernel
void gReduceFullyUnrolled(__global sReal const * xArr, __global sReal * yArr,
                          sIdx const xLen, __local sReal * mBuffer) {
    sIdx const lIdxInBlock = get_local_id(0);
    sIdx const lIdxGlobal = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    sIdx const lBlockSize = get_local_size(0);
    mBuffer[lIdxInBlock] = (lIdxGlobal < xLen) ? xArr[lIdxGlobal] : 0;
    
    // Perform first level of reduction,
    // reading from global memory,
    // writing to shared memory.
    if (lIdxGlobal + lBlockSize < xLen)
        mBuffer[lIdxInBlock] += xArr[lIdxGlobal+lBlockSize];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform reduction in the shared memory.
    if (lBlockSize >= 512) { if (lIdxInBlock < 256) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (lBlockSize >= 256) { if (lIdxInBlock < 128) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (lBlockSize >= 128) { if (lIdxInBlock <  64) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }
    
    if (lIdxInBlock < 32) {
        if (lBlockSize >=  64) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + 32]; }
        if (lBlockSize >=  32) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + 16]; }
        if (lBlockSize >=  16) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  8]; }
        if (lBlockSize >=   8) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  4]; }
        if (lBlockSize >=   4) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  2]; }
        if (lBlockSize >=   2) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  1]; }
    }
    
    // Export this block to global memory.
    if (lIdxInBlock == 0) yArr[get_group_id(0)] = mBuffer[0];
}

/**
 *  Uses Brent's Theorem optimization.
 *  This version adds multiple elements per thread sequentially.
 *  This reduces the overall cost of the algorithm while keeping
 *  the work complexity O(xLen) and the step complexity O(log xLen).
 */
__kernel
void gReduceWBrentsTh(__global sReal const * xArr, __global sReal * yArr,
                      sIdx const xLen, __local sReal * mBuffer) {
    sIdx const lBlockSize = get_local_size(0);
    sIdx const lIdxInBlock = get_local_id(0);
    sIdx lIdxGlobal = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    sIdx const lGridSize = lBlockSize * 2 * get_num_groups(0);
    mBuffer[lIdxInBlock] = (lIdxGlobal < xLen) ? xArr[lIdxGlobal] : 0;
    
    // We reduce multiple elements per thread.
    // The number is determined by the number of active thread blocks (via gridDim).
    // More blocks will result in a larger lGridSize and therefore fewer elements per thread.
    while (lIdxGlobal < xLen) {
        mBuffer[lIdxInBlock] += xArr[lIdxGlobal];
        // Ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays.
        if (lIdxGlobal + lBlockSize < xLen)
            mBuffer[lIdxInBlock] += xArr[lIdxGlobal+lBlockSize];
        lIdxGlobal += lGridSize;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform reduction in the shared memory.
    if (lBlockSize >= 512) { if (lIdxInBlock < 256) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (lBlockSize >= 256) { if (lIdxInBlock < 128) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (lBlockSize >= 128) { if (lIdxInBlock <  64) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }
    
    if (lIdxInBlock < 32) {
        if (lBlockSize >=  64) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + 32]; }
        if (lBlockSize >=  32) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock + 16]; }
        if (lBlockSize >=  16) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  8]; }
        if (lBlockSize >=   8) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  4]; }
        if (lBlockSize >=   4) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  2]; }
        if (lBlockSize >=   2) { mBuffer[lIdxInBlock] += mBuffer[lIdxInBlock +  1]; }
    }
    
    // Export this block to global memory.
    if (lIdxInBlock == 0) yArr[get_group_id(0)] = mBuffer[0];
}
