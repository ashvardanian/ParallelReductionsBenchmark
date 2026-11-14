#!/usr/bin/env python3
"""Python port of `reduce_bench.cpp`, comparing NumPy on the CPU against CuPy and CCCL's `cuda.compute` on the GPU.

Sums a 1 GB array of `float` ones by default, or `PARALLEL_REDUCTIONS_LENGTH` elements, reporting bandwidth and error:

    pip install numpy cupy-cuda13x "cuda-cccl[cu13]"
    PARALLEL_REDUCTIONS_LENGTH=1024 python reduce_bench.py
"""
import os
import time
from functools import partial

import numpy as np
import cupy as cp
import cuda.compute as cc

MIN_SECONDS = 1.0


def numpy_sum(host_inputs):
    output = np.empty(1, dtype=np.float32)
    return lambda: host_inputs.sum(out=output, keepdims=True), output


def cupy_sum(host_inputs):
    device_inputs = cp.asarray(host_inputs)
    output = cp.empty(1, dtype=cp.float32)
    return lambda: device_inputs.sum(out=output, keepdims=True), output


def cccl_sum(host_inputs, op):
    """Reuses one reducer with preallocated scratch space, like `cuda/cub` does in C++."""
    device_inputs = cp.asarray(host_inputs)
    output = cp.empty(1, dtype=cp.float32)
    initial = np.zeros(1, dtype=np.float32)
    reducer = cc.make_reduce_into(d_in=device_inputs, d_out=output, op=op, h_init=initial)
    arguments = dict(d_in=device_inputs, d_out=output, num_items=len(host_inputs), op=op, h_init=initial)
    scratch = cp.empty(reducer(temp_storage=None, **arguments), dtype=cp.uint8)
    return lambda: reducer(temp_storage=scratch, **arguments), output


def plus(a, b):
    return a + b


def time_host(reduce):
    reduce()
    calls, start = 0, time.perf_counter()
    while (seconds := time.perf_counter() - start) < MIN_SECONDS:
        reduce()
        calls += 1
    return seconds / calls


def time_device(reduce):
    reduce()  # The first call compiles and caches the kernels
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    calls = 1
    while True:
        start.record()
        for _ in range(calls):
            reduce()
        stop.record()
        stop.synchronize()
        seconds = cp.cuda.get_elapsed_time(start, stop) / 1e3
        if seconds >= MIN_SECONDS:
            return seconds / calls
        calls *= 2


def report(name, seconds_per_call, total, length):
    bytes_per_second = length * np.dtype(np.float32).itemsize / seconds_per_call
    error = abs(length - total) / length
    print(f"{name:<24}{seconds_per_call * 1e9:>14.0f} ns{bytes_per_second / 1e9:>12.2f} GB/s{error * 100:>12.4g} %")


def main():
    length = int(os.environ.get("PARALLEL_REDUCTIONS_LENGTH", 1024**3 // np.dtype(np.float32).itemsize))
    assert length > 0, "Inappropriate `PARALLEL_REDUCTIONS_LENGTH` value!"
    print(f"Dataset size: {length} elements on {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"{'Benchmark':<24}{'Time':>17}{'Bandwidth':>17}{'Error':>14}")

    host_inputs = np.ones(length, dtype=np.float32)
    backends = [
        ("numpy/f32", time_host, numpy_sum),
        ("cupy/f32", time_device, cupy_sum),
        ("cuda.compute/f32", time_device, partial(cccl_sum, op=cc.OpKind.PLUS)),
        ("cuda.compute/f32/jit", time_device, partial(cccl_sum, op=plus)),
    ]
    for name, timer, backend in backends:
        reduce, output = backend(host_inputs)
        report(name, timer(reduce), float(output[0]), length)


if __name__ == "__main__":
    main()
