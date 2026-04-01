//! Benchmarks for 2^24-point Goldilocks NTT.
//!
//! Measures CPU throughput of the golden model (recursive DIF NTT)
//! and the behavioral SDF pipeline simulation.
//!
//! ## Reference comparison points
//!
//! | Platform                        | 2^24 Goldilocks NTT | Source                  |
//! |---------------------------------|---------------------|-------------------------|
//! | This crate (CPU golden model)   | (run this bench)    | `cargo bench`           |
//! | This crate (behavioral SDF sim) | (run this bench)    | `cargo bench`           |
//! | FPGA SDF @ 200 MHz (projected)  | ~84 ms              | 2^24 / 200 MHz          |
//! | FPGA SDF @ 400 MHz (projected)  | ~42 ms              | 2^24 / 400 MHz          |
//! | ICICLE on RTX 4090              | ~3-5 ms             | Ingonyama benchmarks    |
//!
//! The FPGA projections assume 1 element per cycle throughput from
//! the fully pipelined SDF architecture.  Actual numbers depend on
//! synthesis results and target device.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use goldilocks_ntt_hdl::field::element::GoldilocksElement;
use goldilocks_ntt_hdl::golden::reference::dif_ntt;
use goldilocks_ntt_hdl::sim::runner::{simulate_pipeline, SimConfig};

const LOG_N: usize = 24;
const N: usize = 1 << LOG_N;

/// Generate a deterministic input vector of N field elements.
fn generate_input() -> Vec<GoldilocksElement> {
    // Simple PRNG-like sequence: x_{i+1} = x_i * 7 + 3 (mod p)
    let seed = GoldilocksElement::new(12345);
    let multiplier = GoldilocksElement::new(7);
    let offset = GoldilocksElement::new(3);
    std::iter::successors(Some(seed), |prev| Some(*prev * multiplier + offset))
        .take(N)
        .collect()
}

fn bench_golden_model_ntt(c: &mut Criterion) {
    let input = generate_input();

    c.bench_function("golden_dif_ntt_2^24", |b| {
        b.iter(|| black_box(dif_ntt(black_box(&input))));
    });
}

fn bench_behavioral_sdf_sim(c: &mut Criterion) {
    let input = generate_input();

    c.bench_function("behavioral_sdf_sim_2^24", |b| {
        b.iter_batched(
            || input.clone(),
            |data| {
                black_box(
                    SimConfig::new(data, LOG_N)
                        .and_then(|cfg| simulate_pipeline(cfg).run()),
                )
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_group! {
    name = ntt_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_mins(1));
    targets = bench_golden_model_ntt, bench_behavioral_sdf_sim
}
criterion_main!(ntt_benches);
