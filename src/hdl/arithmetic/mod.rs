//! Goldilocks field arithmetic HDL modules.
//!
//! Each module provides both combinational [`hdl_cat_circuit::CircuitArrow`]
//! and stateful [`hdl_cat_sync::Sync`] variants.  The combinational versions
//! are pure circuit arrows built from IR instructions.  The stateful versions
//! wrap the combinational circuits with delay elements to provide registered
//! outputs and enable pipelined composition.

pub mod adder;
pub mod multiplier;
pub mod subtractor;

pub use adder::{
    goldilocks_add_comb, goldilocks_add_sync,
    GoldilocksAddArrow, GoldilocksAddSync, AdderInput, AdderOutput,
};
pub use multiplier::{
    goldilocks_mul_comb, goldilocks_mul_sync, goldilocks_mul_pipelined,
    GoldilocksMulArrow, GoldilocksMulSync, GoldilocksMulPipelined,
    MulInput, MulOutput, MULTIPLIER_LATENCY,
};
pub use subtractor::{
    goldilocks_sub_comb, goldilocks_sub_sync,
    GoldilocksSubArrow, GoldilocksSubSync, SubInput, SubOutput,
};
