//! `RustHDL` hardware description modules for the NTT pipeline.
//!
//! The `mut` keyword is quarantined to this module: it appears only
//! inside `#[hdl_gen]` blocks (required by `RustHDL`'s `Logic` trait)
//! and inside `Io::suspend` closures at the simulation boundary.
//!
//! All domain logic (field arithmetic, graph structure, interpretation)
//! remains in the pure categorical layer outside this module.

pub mod arithmetic;
pub mod butterfly;
pub mod common;
pub mod delay;
pub mod pipeline;
pub mod stage;
pub mod twiddle;
