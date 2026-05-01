//! Benchmarks for the field-generic NTT pipeline.
//!
//! Measures CPU throughput of the `Goldilocks` golden model (recursive
//! DIF NTT), the `hdl-cat` cycle-accurate SDF pipeline simulation
//! (`Goldilocks` 2^4), and the field-generic SDF stage simulation
//! (`BabyBear`, single stage), so the `Goldilocks`-vs-`BabyBear`
//! comparison is on the same harness.
//!
//! ## Reference comparison points
//!
//! | Platform                        | 2^24 Goldilocks NTT | Source                  |
//! |---------------------------------|---------------------|-------------------------|
//! | This crate (CPU golden model)   | (run this bench)    | `cargo bench`           |
//! | FPGA SDF @ 200 MHz (projected)  | ~84 ms              | 2^24 / 200 MHz          |
//! | FPGA SDF @ 400 MHz (projected)  | ~42 ms              | 2^24 / 400 MHz          |
//! | ICICLE on RTX 4090              | ~3-5 ms             | Ingonyama benchmarks    |
//!
//! The FPGA projections assume 1 element per cycle throughput from
//! the fully pipelined SDF architecture.  Actual numbers depend on
//! synthesis results and target device.

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};

use goldilocks_ntt_hdl::field::element::GoldilocksElement;
use goldilocks_ntt_hdl::golden::reference::dif_ntt;
use goldilocks_ntt_hdl::hdl::stage::sdf_stage_generic;
use goldilocks_ntt_hdl::hdl::{BabyBear, PrimeFieldHdl};
use goldilocks_ntt_hdl::sim::runner::{SimConfig, simulate_pipeline};
use hdl_cat_kind::BitSeq;
use hdl_cat_sim::Testbench;

const LOG_N_LARGE: usize = 24;
const N_LARGE: usize = 1 << LOG_N_LARGE;

const LOG_N_SMALL: usize = 4;
const N_SMALL: usize = 1 << LOG_N_SMALL;

/// Generate a deterministic input vector of `n` field elements.
fn generate_input(n: usize) -> Vec<GoldilocksElement> {
    let seed = GoldilocksElement::new(12345);
    let multiplier = GoldilocksElement::new(7);
    let offset = GoldilocksElement::new(3);
    std::iter::successors(Some(seed), |prev| Some(*prev * multiplier + offset))
        .take(n)
        .collect()
}

fn bench_golden_model_ntt(c: &mut Criterion) {
    let input = generate_input(N_LARGE);

    c.bench_function("golden_dif_ntt_2^24", |b| {
        b.iter(|| black_box(dif_ntt(black_box(&input))));
    });
}

/// Benchmark the hdl-cat cycle-accurate simulation of a 4-stage (16-point) pipeline.
///
/// The full 24-stage pipeline is too slow for cycle-accurate simulation
/// (2^24 testbench cycles), so we benchmark the largest practical size.
fn bench_hdl_cat_sim_size_16(c: &mut Criterion) {
    let input = generate_input(N_SMALL);

    c.bench_function("hdl_cat_sdf_sim_2^4", |b| {
        b.iter_batched(
            || input.clone(),
            |data| {
                black_box(
                    SimConfig::new(data, LOG_N_SMALL).and_then(|cfg| simulate_pipeline(cfg).run()),
                )
            },
            BatchSize::SmallInput,
        );
    });
}

/// Benchmark a single field-generic SDF stage simulation over `BabyBear`.
///
/// Drives an 8-cycle workload (one fill + one butterfly frame at depth 2)
/// through `hdl-cat`'s [`Testbench`] using
/// [`sdf_stage_generic::<BabyBear>`].  Provides a cross-field comparison
/// data point against the `Goldilocks` 2^4 pipeline simulation, since
/// the multi-stage `BabyBear` pipeline composition is not yet exposed.
fn bench_babybear_stage_sim(c: &mut Criterion) {
    let data: Vec<u64> = vec![100, 200, 300, 400, 500, 600, 700, 800];
    let step_root: u64 = 3;
    let inputs: Vec<BitSeq> = data
        .iter()
        .map(|d| {
            BabyBear::to_bitseq(*d)
                .concat(BitSeq::from_vec(vec![true]))
                .concat(BabyBear::to_bitseq(step_root))
        })
        .collect();

    c.bench_function("hdl_cat_babybear_stage_depth_2", |b| {
        b.iter_batched(
            || inputs.clone(),
            |ins| {
                black_box(
                    sdf_stage_generic::<BabyBear>(2)
                        .map(|stage| Testbench::new(stage).run(ins).run()),
                )
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group! {
    name = ntt_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(30));
    targets = bench_golden_model_ntt, bench_hdl_cat_sim_size_16, bench_babybear_stage_sim
}
criterion_main!(ntt_benches);
