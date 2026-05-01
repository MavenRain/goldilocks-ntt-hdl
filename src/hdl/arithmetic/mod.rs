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
    AdderInput, AdderOutput, GoldilocksAddArrow, GoldilocksAddSync, goldilocks_add_comb,
    goldilocks_add_sync,
};
pub use multiplier::{
    GoldilocksMulArrow, GoldilocksMulPipelined, GoldilocksMulSync, MULTIPLIER_LATENCY, MulInput,
    MulOutput, goldilocks_mul_comb, goldilocks_mul_pipelined, goldilocks_mul_sync,
};
pub use subtractor::{
    GoldilocksSubArrow, GoldilocksSubSync, SubInput, SubOutput, goldilocks_sub_comb,
    goldilocks_sub_sync,
};
