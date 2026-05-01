# goldilocks-ntt-hdl

A field-generic Number Theoretic Transform pipeline in `hdl-cat`, with
`Goldilocks` (p = 2^64 - 2^32 + 1) and `BabyBear` (p = 2^31 - 2^27 + 1)
implementations behind a single `PrimeFieldHdl` trait.  The 24-stage
2^24-point SDF (Single-path Delay Feedback) pipeline is composed via
`comp-cat-rs` free-category structure.

## Overview

This crate provides:

- **Goldilocks field arithmetic**: `GoldilocksElement` newtype with
  modular add, subtract, multiply, negate, exponentiation, and
  inverse.  Roots of unity up to order 2^32.
- **Field-generic SDF stages via `PrimeFieldHdl`**: the inline modular
  arithmetic (`alloc_constants`, `inline_add`, `inline_sub`,
  `inline_mul_reduce`) is abstracted behind a trait so the same SDF
  stage construction code targets multiple primes.  `Goldilocks` uses
  the constant-depth Solinas form `2^64 ≡ 2^32 − 1 (mod p)`; `BabyBear`
  uses a 12-pass cascaded Solinas reduction with `2^31 ≡ 2^27 − 1
  (mod p)`.
- **Golden model**: Pure recursive DIF NTT with verified round-trip
  inverse, bit-reversal permutation (`Goldilocks`-specific).
- **Categorical pipeline description**: The 24-stage `Goldilocks` SDF
  pipeline is a free category graph
  (`comp_cat_rs::collapse::free_category`).  Each stage is an edge; the
  `interpret()` universal property composes them into a single pipeline
  descriptor.
- **`hdl-cat` modules**: `CircuitArrow` and `Sync` machines for modular
  arithmetic (adder, subtractor, 7-cycle pipelined multiplier), DIF
  butterfly, on-the-fly twiddle accumulator, parameterized delay line,
  SDF stage, and Verilog emission via `emit_sync_graph`.
- **`Io`-wrapped simulation**: Behavioral SDF simulation with all
  side effects deferred inside `comp_cat_rs::effect::io::Io::suspend`.

## Architecture

```text
Layer 1 (Pure)                    Layer 2 (HDL)
--------------------              --------------------
field/                            hdl/arithmetic/
  element.rs  (GoldilocksElement)   adder.rs    (1-cycle)
  roots.rs    (roots of unity)      subtractor.rs (1-cycle)
                                    multiplier.rs (7-cycle)
golden/                           hdl/
  reference.rs (recursive DIF NTT)  field_hdl.rs            (PrimeFieldHdl)
                                    goldilocks_field_hdl.rs (Goldilocks impl)
graph/                              babybear_field_hdl.rs   (BabyBear impl)
  ntt_graph.rs (Graph: 25V, 24E)    butterfly.rs  (8-cycle DIF)
                                    delay.rs      (circular buffer)
interpret/                          stage.rs      (SDF fill/butterfly)
  signal.rs     (StageSignal)       pipeline.rs   (24 composed stages)
  descriptor.rs (SdfStageDescriptor)
  hdl_morphism.rs (GraphMorphism) sim/
                                    runner.rs   (Io-wrapped sim)
```

**Layer 1** is pure: zero `mut`, combinators only, comp-cat-rs effects.
**Layer 2** quarantines `mut` inside `Io::suspend` closures at the
simulation boundary.

The bridge between layers is the `interpret()` universal property of the
free category: it maps the abstract graph into concrete HDL stage
descriptors, which Layer 2 materializes into `hdl-cat` `Sync` machines.

## SDF Pipeline

Each of the 24 stages alternates between two phases:

- **Fill phase** (D_k cycles): incoming data fills the delay line and
  passes through to the output.
- **Butterfly phase** (D_k cycles): incoming data pairs with delayed
  data through the DIF butterfly `(a+b, (a-b)*twiddle)`.

Stage k has delay depth D_k = 2^(23-k).  Twiddle factors are generated
on-the-fly via a running accumulator (one field multiply per cycle),
eliminating 128 MB of twiddle ROM.

Output is in bit-reversed order (standard DIF property).

## Usage

```rust
use goldilocks_ntt_hdl::field::element::GoldilocksElement;
use goldilocks_ntt_hdl::golden::reference::{dif_ntt, inverse_ntt};

// Forward NTT (output in bit-reversed order)
let input: Vec<GoldilocksElement> = (1..=8)
    .map(GoldilocksElement::new)
    .collect();
let forward = dif_ntt(&input).ok();

// Inverse NTT recovers the original
let recovered = forward.and_then(|f| inverse_ntt(&f).ok());
```

```rust
use goldilocks_ntt_hdl::graph::ntt_graph::{NttGraph, full_pipeline_path};
use goldilocks_ntt_hdl::interpret::hdl_morphism::HdlInterpretation;
use goldilocks_ntt_hdl::interpret::descriptor::SdfStageDescriptor;
use comp_cat_rs::collapse::free_category::interpret;

// Compose the full pipeline via the free category universal property
let interp = HdlInterpretation::new().ok();
let path = full_pipeline_path().ok();

let descriptor = interp.zip(path).map(|(i, p)| {
    interpret::<NttGraph, _>(
        &i,
        &p,
        |_| SdfStageDescriptor::identity(),
        SdfStageDescriptor::compose,
    )
});
// descriptor.stage_count() == 24
```

```rust
use goldilocks_ntt_hdl::field::element::GoldilocksElement;
use goldilocks_ntt_hdl::sim::runner::{SimConfig, simulate_pipeline};

// Behavioral simulation with deferred execution
let input: Vec<GoldilocksElement> = (0..16)
    .map(GoldilocksElement::new)
    .collect();
let config = SimConfig::new(input, 4).ok();

// Nothing executes until .run()
let result = config.map(|c| simulate_pipeline(c).run());
```

## comp-cat-rs Integration

| comp-cat-rs concept | NTT mapping |
|---|---|
| `Graph` | `NttGraph`: 25 vertices (stage boundaries), 24 edges (stages) |
| `Path` | Full pipeline path: 24 composed singleton edges |
| `GraphMorphism` | `HdlInterpretation`: vertex to signal type, edge to stage descriptor |
| `interpret()` | Composes 24 descriptors into one pipeline descriptor |
| `Io<Error, _>` | Wraps simulation side effects; `.run()` at boundary only |
| `Braided` / `Symmetric` | Bit-reversal permutation (output reordering) |

## Building

```sh
cargo build
cargo test
RUSTFLAGS="-D warnings" cargo clippy
cargo doc --no-deps --open
```

## Testing

131 tests across three levels:

- **Unit tests** (104): field axioms, root of unity properties, graph
  structure, descriptor composition, interpretation correctness,
  pipeline construction, `Goldilocks` and `BabyBear`
  `PrimeFieldHdl` add / sub / mul-reduce.
- **Integration tests** (19): golden model round-trip, passthrough
  preservation, output length matching, hdl arithmetic property tests,
  and a `BabyBear` cross-field SDF stage smoke test
  (`tests/babybear_stage.rs`).
- **Doctests** (8): `GoldilocksElement`, roots, NTT forward/inverse,
  graph path, categorical interpretation, simulation.

```sh
cargo test          # all unit + integration tests
cargo test --doc    # doctests
```

## Benchmarks

2^24-point Goldilocks NTT benchmarks comparing CPU implementations
against projected FPGA and published GPU numbers.

```sh
cargo bench
```

This runs three benchmarks via `criterion`:

- **`golden_dif_ntt_2^24`**: pure recursive `Goldilocks` DIF NTT (CPU,
  single-threaded).
- **`hdl_cat_sdf_sim_2^4`**: cycle-accurate `Goldilocks` SDF pipeline
  simulation at 16 points (the largest size practical without a full
  testbench cycle budget).
- **`hdl_cat_babybear_stage_depth_2`**: cycle-accurate single
  `BabyBear` SDF stage simulation, providing a same-harness
  cross-field comparison data point until a multi-stage `BabyBear`
  pipeline is exposed.

### Reference comparison

| Platform                       | 2^24 Goldilocks NTT | Notes                         |
|--------------------------------|---------------------|-------------------------------|
| CPU golden model               | (run `cargo bench`) | recursive DIF, single-thread  |
| CPU behavioral SDF sim         | (run `cargo bench`) | 24-stage streaming simulation |
| FPGA SDF @ 200 MHz (projected) | ~84 ms              | 2^24 cycles / 200 MHz         |
| FPGA SDF @ 400 MHz (projected) | ~42 ms              | 2^24 cycles / 400 MHz         |
| RTX 4090 (ICICLE)              | ~3-5 ms             | Ingonyama published numbers   |

FPGA projections assume 1 element per cycle throughput from the fully
pipelined SDF architecture.  Actual numbers depend on synthesis results
and target device (delay line memory, DSP utilization, clock closure).

The RTX 4090 number is the current ceiling; a future `cuda` feature
flag integrating the `icicle` crate would enable direct comparison
on the same machine.

## License

Licensed under either of

- MIT license
- Apache License, Version 2.0

at your option.
