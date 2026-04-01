# goldilocks-ntt-hdl

A 2^24-point Number Theoretic Transform over the Goldilocks field
(p = 2^64 - 2^32 + 1), implemented as a fully pipelined SDF
(Single-path Delay Feedback) architecture in `RustHDL`, with
compositional pipeline assembly driven by `comp-cat-rs`.

## Overview

This crate provides:

- **Goldilocks field arithmetic**: `GoldilocksElement` newtype with
  modular add, subtract, multiply, negate, exponentiation, and
  inverse.  Roots of unity up to order 2^32.
- **Golden model**: Pure recursive DIF NTT with verified round-trip
  inverse, bit-reversal permutation.
- **Categorical pipeline description**: The 24-stage SDF pipeline is
  a free category graph (`comp_cat_rs::collapse::free_category`).
  Each stage is an edge; the `interpret()` universal property composes
  them into a single pipeline descriptor.
- **`RustHDL` modules**: `LogicBlock` implementations for modular
  arithmetic (adder, subtractor, 7-cycle pipelined multiplier), DIF
  butterfly, on-the-fly twiddle accumulator, parameterized delay line,
  SDF stage, and the full 24-stage pipeline.
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
  reference.rs (recursive DIF NTT)  butterfly.rs  (8-cycle DIF)
                                    twiddle.rs    (on-the-fly accum)
graph/                              delay.rs      (circular buffer)
  ntt_graph.rs (Graph: 25V, 24E)    stage.rs      (SDF fill/butterfly)
                                    pipeline.rs   (24 composed stages)
interpret/
  signal.rs     (StageSignal)     sim/
  descriptor.rs (SdfStageDescriptor)  runner.rs   (Io-wrapped sim)
  hdl_morphism.rs (GraphMorphism)
```

**Layer 1** is pure: zero `mut`, combinators only, comp-cat-rs effects.
**Layer 2** quarantines `mut` inside `RustHDL`'s `Logic::update` methods
and `Io::suspend` closures at the simulation boundary.

The bridge between layers is the `interpret()` universal property of the
free category: it maps the abstract graph into concrete HDL stage
descriptors, which Layer 2 materializes into `RustHDL` modules.

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
    .map(|i| GoldilocksElement::new(i))
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
    .map(|i| GoldilocksElement::new(i))
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

61 tests across three levels:

- **Unit tests** (56): field axioms, root of unity properties, graph
  structure, descriptor composition, interpretation correctness,
  pipeline construction.
- **Integration tests** (4): golden model round-trip, passthrough
  preservation, output length matching.
- **Doctests** (1): `GoldilocksElement` arithmetic.

```sh
cargo test          # all unit + integration tests
cargo test --doc    # doctests
```

## License

Licensed under either of

- MIT license
- Apache License, Version 2.0

at your option.
