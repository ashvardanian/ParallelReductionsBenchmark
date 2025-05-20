//! Rust port of `reduce_bench.cpp` comparing Rust thread-pools,
//! like: `rayon`, `tokio`, `smol`, and `fork_union`.
#![feature(allocator_api)]
use criterion::{criterion_group, criterion_main, Criterion};
use std::alloc::Global;
use std::env;
use std::hint::black_box;
use std::ptr;

use futures::future::join_all;

const MAX_CACHE_LINE_SIZE: usize = 64; // bytes on x86; adjust if needed
const SCALARS_PER_CACHE_LINE: usize = MAX_CACHE_LINE_SIZE / std::mem::size_of::<f32>();

#[inline(always)]
fn divide_round_up(value: usize, divisor: usize) -> usize {
    (value + divisor - 1) / divisor
}

#[inline(always)]
fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    divide_round_up(value, multiple) * multiple
}

#[inline]
fn scalars_per_core(input_size: usize, total_cores: usize) -> usize {
    let chunk = divide_round_up(input_size, total_cores);
    round_up_to_multiple(chunk, SCALARS_PER_CACHE_LINE)
}

/// Manual eight‑way loop‑unrolled accumulation
#[inline(never)]
pub fn sum_unrolled(slice: &[f32]) -> f64 {
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;
    let mut s4 = 0.0f64;
    let mut s5 = 0.0f64;
    let mut s6 = 0.0f64;
    let mut s7 = 0.0f64;

    let chunks = slice.chunks_exact(8);
    for chunk in chunks.clone() {
        s0 += chunk[0] as f64;
        s1 += chunk[1] as f64;
        s2 += chunk[2] as f64;
        s3 += chunk[3] as f64;
        s4 += chunk[4] as f64;
        s5 += chunk[5] as f64;
        s6 += chunk[6] as f64;
        s7 += chunk[7] as f64;
    }
    for &val in chunks.remainder() {
        s0 += val as f64;
    }
    s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
}

/// Reads `PARALLEL_REDUCTIONS_LENGTH`, defaulting to 1 GB of `f32` elements (≈268 435 456)
pub fn elements_from_env() -> usize {
    const ONE_GB_IN_ELEMENTS: usize = 1_073_741_824 / std::mem::size_of::<f32>();
    match env::var("PARALLEL_REDUCTIONS_LENGTH") {
        Ok(val) => {
            let parsed = val.parse::<usize>().unwrap_or(0);
            assert!(
                parsed > 0,
                "Inappropriate `PARALLEL_REDUCTIONS_LENGTH` value: {val}"
            );
            parsed
        }
        Err(_) => ONE_GB_IN_ELEMENTS,
    }
}

/// Allocates and initializes the vector once for the whole benchmark suite.
/// Each entry equals its index modulo 1000, to avoid constant‑folded results.
pub fn prepare_input() -> Vec<f32> {
    let n = elements_from_env();
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push((i % 1000) as f32);
    }
    data
}

/// Parallel reduction through Fork Union, using a custom pool with explicit chunking
pub fn sum_fork_union(pool: &fork_union::ForkUnion, data: &[f32], partial_sums: &mut [f64]) -> f64 {
    let cores = pool.thread_count();
    let chunk_size = scalars_per_core(data.len(), cores);
    let partial_sums_ptr = partial_sums.as_mut_ptr();

    pool.for_each_thread(|thread_index| unsafe {
        let start = thread_index * chunk_size;
        if start >= data.len() {
            return;
        }
        let stop = usize::min(start + chunk_size, data.len());
        let partial_sum = sum_unrolled(&data[start..stop]);
        ptr::write(partial_sums_ptr.add(thread_index), partial_sum);
    });

    partial_sums[..].into_iter().sum()
}

/// Parallel reduction through Rayon, using a custom pool with explicit chunking
pub fn sum_rayon(pool: &rayon::ThreadPool, data: &[f32], partial_sums: &mut [f64]) -> f64 {
    let cores = pool.current_num_threads();
    let chunk_size = scalars_per_core(data.len(), cores);
    let partial_sums_ptr = partial_sums.as_mut_ptr();

    pool.broadcast(|context: rayon::BroadcastContext<'_>| {
        let thread_index = context.index();
        let start = thread_index * chunk_size;
        if start >= data.len() {
            return;
        }
        let stop = std::cmp::min(start + chunk_size, data.len());
        let partial_sum = sum_unrolled(&data[start..stop]);
        unsafe {
            ptr::write(partial_sums_ptr.add(thread_index), partial_sum);
        }
    });

    partial_sums[..].into_iter().sum()
}

/// Parallel reduction through Tokio, which notably itself recommends Rayon for CPU-bound tasks
/// https://docs.rs/tokio/latest/tokio/index.html#cpu-bound-tasks-and-blocking-code
pub fn sum_tokio(pool: &tokio::runtime::Runtime, data: &[f32], partial_sums: &mut [f64]) -> f64 {
    let cores = num_cpus::get();
    let chunk_size = scalars_per_core(data.len(), cores);
    let partial_sums_ptr = partial_sums.as_mut_ptr();

    // Raw parts of the slice – immutable, lives as long as `data`.
    let ptr = data.as_ptr();
    let len = data.len();

    pool.block_on(async move {
        let mut handles = Vec::with_capacity(cores);
        for thread_index in 0..cores {
            let start = thread_index * chunk_size;
            let handle = pool.spawn_blocking(move || unsafe {
                if start >= len {
                    return;
                }
                let stop = std::cmp::min(start + chunk_size, len);
                let slice = std::slice::from_raw_parts(ptr.add(start), stop - start);
                let partial_sum = sum_unrolled(slice);
                ptr::write(partial_sums_ptr.add(thread_index), partial_sum);
            });
            handles.push(handle);
        }
        let _ = join_all(handles).await;
        partial_sums[..].into_iter().sum()
    })
}

/// Parallel reduction through "Smol.rs" toolkit.
/// The `async-executor` is recommended, but it doesn't allow setting the number of threads.
pub fn sum_smol(pool: &async_executor::Executor, data: &[f32], partial_sums: &mut [f64]) -> f64 {
    let cores = num_cpus::get();
    let chunk_size = scalars_per_core(data.len(), cores);
    let partial_sums_ptr = partial_sums.as_mut_ptr();

    let ptr = data.as_ptr();
    let len = data.len();

    pool.run(async move {
        let mut tasks = Vec::with_capacity(cores);
        for thread_index in 0..cores {
            let start = thread_index * chunk_size;
            tasks.push(pool.spawn(async move || unsafe {
                if start >= len {
                    return;
                }
                let stop = std::cmp::min(start + chunk_size, len);
                let slice = std::slice::from_raw_parts(ptr.add(start), stop - start);
                let partial_sum = sum_unrolled(slice);
                ptr::write(partial_sums_ptr.add(thread_index), partial_sum);
            }));
        }
        let _ = join_all(tasks).await;
        partial_sums[..].into_iter().sum()
    })
}

pub fn reduction_bench(c: &mut Criterion) {
    let cores = num_cpus::get();
    let data = prepare_input();
    let mut partial_sums = vec![0.0; cores];

    // Sum with the serial baseline
    c.bench_function("serial", |b| b.iter(|| black_box(sum_unrolled(&data))));

    // Sum with Fork Union
    {
        let pool = fork_union::ForkUnion::try_spawn_in(cores, Global).unwrap();
        c.bench_function("fork_union", |b| {
            b.iter(|| black_box(sum_fork_union(&pool, &data, &mut partial_sums)))
        });
    }

    // Sum with Rayon
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cores)
            .build()
            .unwrap();
        c.bench_function("rayon", |b| {
            b.iter(|| black_box(sum_rayon(&pool, &data, &mut partial_sums)))
        });
    }

    // Sum with Tokio
    {
        let pool = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(cores)
            .enable_all()
            .build()
            .unwrap();
        c.bench_function("tokio", |b| {
            b.iter(|| black_box(sum_tokio(&pool, &data, &mut partial_sums)))
        });
    }

    // Sum with Smol
    {
        let pool = async_executor::Executor::new();
        c.bench_function("smol", |b| {
            b.iter(|| black_box(sum_smol(&pool, &data, &mut partial_sums)))
        });
    }
}

criterion_group!(benches, reduction_bench);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= EPS * a.abs().max(b.abs())
    }

    #[test]
    fn smoke() {
        let data = prepare_input();
        let serial = sum_unrolled(&data);
        let cores = num_cpus::get();
        let mut partial_sums = vec![0.0; cores];

        // Fork Union
        let pool_fu = ForkUnion::try_spawn_in(cores, Global).unwrap();
        let r_fu = sum_fork_union(&pool_fu, &data, &partial_sums);
        assert!(approx_eq(serial, r_fu));

        // Rayon
        let pool_rayon = rayon::ThreadPoolBuilder::new()
            .num_threads(cores)
            .build()
            .unwrap();
        let r_rayon = sum_rayon(&pool_rayon, &data, &partial_sums);
        assert!(approx_eq(serial, r_rayon));

        // Tokio
        let pool_tokio = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(cores)
            .enable_all()
            .build()
            .unwrap();
        let r_tokio = sum_tokio(&pool_tokio, &data, &partial_sums);
        assert!(approx_eq(serial, r_tokio));

        // Smol
        let pool_smol = async_executor::Executor::new();
        let r_smol = sum_smol(&pool_smol, &data, &partial_sums);
        assert!(approx_eq(serial, r_smol));
    }
}
