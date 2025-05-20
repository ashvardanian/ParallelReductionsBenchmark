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
pub fn sum_fork_union(data: &[f32], pool: &fork_union::ForkUnion) -> f64 {
    let cores = pool.thread_count();
    let chunk_size = scalars_per_core(data.len(), cores);

    let mut sums_per_thread = vec![0.0f64; cores];
    let sums_per_thread_ptr = sums_per_thread.as_mut_ptr();

    pool.for_each_thread(|thread_index| unsafe {
        let start = thread_index * chunk_size;
        if start < data.len() {
            let end = usize::min(start + chunk_size, data.len());
            let val = sum_unrolled(&data[start..end]);
            ptr::write(sums_per_thread_ptr.add(thread_index), val);
        }
    });

    sums_per_thread.into_iter().sum::<f64>()
}

/// Parallel reduction through Rayon, using a custom pool with explicit chunking
pub fn sum_rayon(data: &[f32], pool: &rayon::ThreadPool) -> f64 {
    let cores = pool.current_num_threads();
    let chunk_size = scalars_per_core(data.len(), cores);

    pool.broadcast(|context: rayon::BroadcastContext<'_>| {
        let thread_index = context.index();
        let start_offset = thread_index * chunk_size;
        if start_offset >= data.len() {
            0.0
        } else {
            let stop_offset = std::cmp::min(start_offset + chunk_size, data.len());
            sum_unrolled(&data[start_offset..stop_offset])
        }
    })
    .into_iter()
    .sum::<f64>()
}

/// Parallel reduction through Tokio, which notably itself recommends Rayon for CPU-bound tasks
/// https://docs.rs/tokio/latest/tokio/index.html#cpu-bound-tasks-and-blocking-code
pub fn sum_tokio(data: &[f32], pool: &tokio::runtime::Runtime) -> f64 {
    let cores = num_cpus::get();
    let chunk_size = scalars_per_core(data.len(), cores);

    // Raw parts of the slice – immutable, lives as long as `data`.
    let ptr = data.as_ptr();
    let len = data.len();

    pool.block_on(async move {
        let mut handles = Vec::with_capacity(cores);
        for thread_index in 0..cores {
            let start_offset = thread_index * chunk_size;
            let handle = pool.spawn_blocking(move || unsafe {
                if start_offset >= len {
                    0.0
                } else {
                    let stop_offset = std::cmp::min(start_offset + chunk_size, len);
                    let slice = std::slice::from_raw_parts(
                        ptr.add(start_offset),
                        stop_offset - start_offset,
                    );
                    sum_unrolled(slice)
                }
            });
            handles.push(handle);
        }
        let partials = join_all(handles).await;
        partials.into_iter().map(|r| r.unwrap()).sum::<f64>()
    })
}

/// Parallel reduction through "Smol.rs" toolkit.
/// The `async-executor` is recommended, but it doesn't allow setting the number of threads.
pub fn sum_smol(data: &[f32], pool: &async_executor::Executor) -> f64 {
    let cores = num_cpus::get();
    let chunk_size = scalars_per_core(data.len(), cores);

    let ptr = data.as_ptr();
    let len = data.len();

    pool.run(async move {
        let mut tasks = Vec::with_capacity(cores);
        for thread_index in 0..cores {
            let start_offset = thread_index * chunk_size;
            tasks.push(pool.spawn(async move {
                unsafe {
                    if start_offset >= len {
                        0.0
                    } else {
                        let stop_offset = std::cmp::min(start_offset + chunk_size, len);
                        let slice = std::slice::from_raw_parts(
                            ptr.add(start_offset),
                            stop_offset - start_offset,
                        );
                        sum_unrolled(slice)
                    }
                }
            }));
        }
        let partials = join_all(tasks).await;
        partials.into_iter().sum::<f64>()
    })
}

pub fn reduction_bench(c: &mut Criterion) {
    let cores = num_cpus::get();
    let data = prepare_input();

    // Sum with the serial baseline
    c.bench_function("serial", |b| b.iter(|| black_box(sum_unrolled(&data))));

    // Sum with Fork Union
    {
        let pool = fork_union::ForkUnion::try_spawn_in(cores, Global).unwrap();
        c.bench_function("fork_union", |b| {
            b.iter(|| black_box(sum_fork_union(&data, &pool)))
        });
    }

    // Sum with Rayon
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cores)
            .build()
            .unwrap();
        c.bench_function("rayon", |b| b.iter(|| black_box(sum_rayon(&data, &pool))));
    }

    // Sum with Tokio
    {
        let pool = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(cores)
            .enable_all()
            .build()
            .unwrap();
        c.bench_function("tokio", |b| b.iter(|| black_box(sum_tokio(&data, &pool))));
    }

    // Sum with Smol
    {
        let pool = async_executor::Executor::new();
        c.bench_function("smol", |b| b.iter(|| black_box(sum_smol(&data, &pool))));
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

        // Fork Union
        let pool_fu = ForkUnion::try_spawn_in(num_cpus::get(), Global).unwrap();
        let r_fu = sum_fork_union(&data, &pool_fu);
        assert!(approx_eq(serial, r_fu));

        // Rayon
        let pool_rayon = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .build()
            .unwrap();
        let r_rayon = sum_rayon(&data, &pool_rayon);
        assert!(approx_eq(serial, r_rayon));

        // Tokio
        let pool_tokio = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_cpus::get())
            .enable_all()
            .build()
            .unwrap();
        let r_tokio = sum_tokio(&data, &pool_tokio);
        assert!(approx_eq(serial, r_tokio));

        // Smol
        let pool_smol = async_executor::Executor::new();
        let r_smol = sum_smol(&data, &pool_smol);
        assert!(approx_eq(serial, r_smol));
    }
}
