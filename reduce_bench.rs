/// Rust port of `reduce_bench.cpp` comparing Rust thread-pools,
/// like: `rayon`, `tokio`, `smol`, and `fork_union`.
use std::env;
use std::sync::Arc;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

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
            assert!(parsed > 0, "Inappropriate `PARALLEL_REDUCTIONS_LENGTH` value: {val}");
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



/// Synchronous helper invoked by Criterion; wraps a dedicated Tokio runtime.
pub fn sum_tokio(data: &[f32], runtime: &tokio::runtime::Runtime) -> f64 {
    let cores = num_cpus::get();
    let chunk_size = scalars_per_core(data.len(), cores);
    let shared = Arc::from(data);

    runtime.block_on(async move {
        use tokio::task;
        use futures::future::join_all;
        let mut tasks = Vec::with_capacity(cores);
        for tid in 0..cores {
            let start = tid * chunk_size;
            if start >= shared.len() {
                break;
            }
            let stop = std::cmp::min(start + chunk_size, shared.len());
            let local = Arc::clone(&shared);
            tasks.push(task::spawn_blocking(move || sum_unrolled(&local[start..stop])));
        }
        let partials = join_all(tasks).await;
        partials.into_iter().map(|r| r.unwrap()).sum::<f64>()
    })
}

/// Parallel reduction through Rayon, using a custom pool with explicit chunking
pub fn sum_rayon(data: &[f32]) -> f64 {
    let cores = num_cpus::get();
    let chunk_size = scalars_per_core(data.len(), cores);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cores)
        .build()
        .unwrap();

    pool.install(|| {
        (0..cores)
            .into_par_iter()
            .map(|tid| {
                let start = tid * chunk_size;
                if start >= data.len() {
                    0.0
                } else {
                    let stop = std::cmp::min(start + chunk_size, data.len());
                    sum_unrolled(&data[start..stop])
                }
            })
            .sum::<f64>()
    })
}

/// Parallel reduction built on Smol / async‑global‑executor
pub fn sum_smol(data: &[f32]) -> f64 {
    let cores = num_cpus::get();
    let chunk_size = scalars_per_core(data.len(), cores);
    let shared = Arc::from(data);

    smol::block_on(async {
        use futures::future::join_all;
        let mut tasks = Vec::with_capacity(cores);
        for tid in 0..cores {
            let start = tid * chunk_size;
            if start >= shared.len() {
                break;
            }
            let stop = std::cmp::min(start + chunk_size, shared.len());
            let local = Arc::clone(&shared);
            tasks.push(smol::unblock(move || sum_unrolled(&local[start..stop])));
        }
        let partials = join_all(tasks).await;
        partials.into_iter().sum::<f64>()
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke() {
        let data = prepare_input();
        let tokio_rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_cpus::get())
            .enable_all()
            .build()
            .unwrap();
        let serial = sum_unrolled(&data);
        let r_tokio = sum_tokio(&data, &tokio_rt);
        let r_rayon = sum_rayon(&data);
        let r_smol = sum_smol(&data);
        assert_eq!(serial, r_tokio);
        assert_eq!(serial, r_rayon);
        assert_eq!(serial, r_smol);
    }
}


pub fn reduction_bench(c: &mut Criterion) {
    let data = prepare_input();
    let tokio_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()
        .unwrap();

    c.bench_function("serial", |b| b.iter(|| black_box(sum_unrolled(&data))));
    c.bench_function("rayon", |b| b.iter(|| black_box(sum_rayon(&data))));
    c.bench_function("tokio", |b| {
        b.iter(|| black_box(sum_tokio(&data, &tokio_rt)))
    });

    c.bench_function("smol", |b| b.iter(|| black_box(sum_smol(&data))));
}

criterion_group!(benches, reduction_bench);
criterion_main!(benches);
