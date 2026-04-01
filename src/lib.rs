//! # goldilocks-ntt-hdl
//!
//! A 2^24-point Number Theoretic Transform over the Goldilocks field
//! (p = 2^64 - 2^32 + 1), implemented as a fully pipelined SDF
//! (Single-path Delay Feedback) architecture in `RustHDL`.
//!
//! ## Two-Layer Architecture
//!
//! ```text
//! Layer 1 (Pure)                    Layer 2 (HDL)
//! --------------------              --------------------
//! field/   element, roots           hdl/arithmetic/  add, sub, mul
//! golden/  recursive DIF NTT        hdl/  butterfly, twiddle, delay
//! graph/   NttGraph (25V, 24E)      hdl/  stage, pipeline
//! interpret/  GraphMorphism          sim/  Io-wrapped behavioral sim
//! ```
//!
//! **Layer 1** is pure: zero `mut`, combinators only, `comp-cat-rs`
//! effects.  **Layer 2** quarantines `mut` inside `RustHDL`'s
//! `Logic::update` methods and [`Io::suspend`](comp_cat_rs::effect::io::Io::suspend)
//! closures.
//!
//! ## Categorical Composition
//!
//! The 24-stage SDF pipeline is modeled as a free category graph
//! ([`graph::ntt_graph::NttGraph`]) with 25 vertices and 24 edges.
//! The [`comp_cat_rs::collapse::free_category::interpret`] universal
//! property composes the stages into a single pipeline descriptor
//! via [`interpret::hdl_morphism::HdlInterpretation`].
//!
//! ## Quick Start
//!
//! ```
//! use goldilocks_ntt_hdl::field::element::GoldilocksElement;
//! use goldilocks_ntt_hdl::golden::reference::{dif_ntt, inverse_ntt};
//!
//! let input: Vec<GoldilocksElement> = (1..=8)
//!     .map(GoldilocksElement::new)
//!     .collect();
//!
//! // Forward DIF NTT (output in bit-reversed order)
//! let forward = dif_ntt(&input).ok();
//!
//! // Inverse NTT recovers the original
//! let recovered = forward.and_then(|f| inverse_ntt(&f).ok());
//! ```

pub mod error;
pub mod field;
pub mod golden;
pub mod graph;
pub mod hdl;
pub mod interpret;
pub mod sim;
