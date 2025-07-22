"""
reduce_bench.py - Python benchmarks for CUDA DSL/JIT reductions using cuda.cccl

This script benchmarks parallel reductions using the cuda.cccl library, comparing its performance to naive CuPy implementations.

Requirements:
    pip install cuda-cccl cupy numpy

Author: Ansh Singh Sonkhia
Date: July 23, 2025
"""

import time
import numpy as np
import cupy as cp
from cuda import parallel

# Define reduction operation and transform
add = lambda x, y: x + y
def transform(x):
    return -x if x % 2 == 0 else x

# Benchmark parameters
SIZES = [10_000, 100_000, 1_000_000, 10_000_000]
REPEATS = 10


def bench_naive(size):
    seq = cp.arange(1, size + 1)
    cp.cuda.runtime.deviceSynchronize()
    start = time.perf_counter()
    for _ in range(REPEATS):
        result = (seq * (-1) ** (seq + 1)).sum()
        cp.cuda.runtime.deviceSynchronize()
    end = time.perf_counter()
    return (end - start) / REPEATS * 1e6  # microseconds


def bench_cccl(size):
    counts = parallel.CountingIterator(np.int32(1))
    seq = parallel.TransformIterator(counts, transform)
    out = cp.empty(1, cp.int32)
    reducer = parallel.reduce_into(seq, out, add, np.int32(0))
    tmp_storage_size = reducer(None, seq, out, size, np.int32(0))
    tmp_storage = cp.empty(tmp_storage_size, cp.uint8)
    cp.cuda.runtime.deviceSynchronize()
    start = time.perf_counter()
    for _ in range(REPEATS):
        reducer(tmp_storage, seq, out, size, np.int32(0))
        cp.cuda.runtime.deviceSynchronize()
    end = time.perf_counter()
    return (end - start) / REPEATS * 1e6  # microseconds


def main():
    print(f"{'Size':>12} | {'Naive (us)':>12} | {'CCCL (us)':>12} | Speedup")
    print("-" * 50)
    for size in SIZES:
        t_naive = bench_naive(size)
        t_cccl = bench_cccl(size)
        speedup = t_naive / t_cccl if t_cccl > 0 else float('inf')
        print(f"{size:12} | {t_naive:12.2f} | {t_cccl:12.2f} | {speedup:7.2f}x")

if __name__ == "__main__":
    main()
