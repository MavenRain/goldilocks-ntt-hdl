//! hdl-cat hardware description modules for the NTT pipeline.
//!
//! This module contains the hdl-cat hardware layer built from categorical
//! circuit abstractions.  All components are constructed via
//! [`hdl_cat_ir::HdlGraphBuilder`] and composed using categorical operations
//! like [`hdl_cat_sync::compose_sync`] for sequential composition.
//!
//! State management is handled through hdl-cat's `Sync<S, I, O>` machines,
//! which are pure from the outside but maintain internal state across cycles.
//!
//! All domain logic (field arithmetic, graph structure, interpretation)
//! remains in the pure categorical layer outside this module.

pub mod arithmetic;
pub mod babybear_field_hdl;
pub mod butterfly;
pub mod common;
pub mod delay;
pub mod field_hdl;
pub mod goldilocks_field_hdl;
pub mod goldilocks_reduce;
pub mod pipeline;
pub mod stage;

// Re-export key types and functions
pub use arithmetic::{
    GoldilocksAddArrow, GoldilocksMulArrow, GoldilocksSubArrow, goldilocks_add_comb,
    goldilocks_mul_comb, goldilocks_sub_comb,
};
pub use babybear_field_hdl::{BABYBEAR_PRIME_U64, BabyBear, BabyBearConstants};
pub use butterfly::{BUTTERFLY_LATENCY, DifButterflySync, dif_butterfly};
pub use common::{GOLDILOCKS_PRIME_U64, GoldilocksElement, bitseq_to_u64, u64_to_bitseq};
pub use delay::delay_n;
pub use field_hdl::PrimeFieldHdl;
pub use goldilocks_field_hdl::{Goldilocks, GoldilocksConstants};
pub use goldilocks_reduce::{goldilocks_mul_reduce_arrow, goldilocks_reduce_arrow};
pub use pipeline::{
    emit_pipeline_circom, emit_pipeline_verilog, emit_size_4_pipeline_circom,
    emit_size_4_pipeline_verilog, size_4_pipeline,
};
pub use stage::{emit_sdf_stage_circom, sdf_stage, sdf_stage_depth_1, sdf_stage_depth_2};
