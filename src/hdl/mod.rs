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
    goldilocks_add_comb, goldilocks_sub_comb, goldilocks_mul_comb,
    GoldilocksAddArrow, GoldilocksSubArrow, GoldilocksMulArrow,
};
pub use babybear_field_hdl::{BabyBear, BabyBearConstants, BABYBEAR_PRIME_U64};
pub use butterfly::{dif_butterfly, DifButterflySync, BUTTERFLY_LATENCY};
pub use common::{GoldilocksElement, GOLDILOCKS_PRIME_U64, u64_to_bitseq, bitseq_to_u64};
pub use delay::delay_n;
pub use goldilocks_reduce::{goldilocks_reduce_arrow, goldilocks_mul_reduce_arrow};
pub use pipeline::{size_4_pipeline, emit_size_4_pipeline_verilog};
pub use stage::{sdf_stage, sdf_stage_depth_1, sdf_stage_depth_2};
pub use field_hdl::PrimeFieldHdl;
pub use goldilocks_field_hdl::{Goldilocks, GoldilocksConstants};
