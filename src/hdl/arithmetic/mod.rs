//! Goldilocks field arithmetic HDL modules.
//!
//! Each module is a `LogicBlock` with registered outputs and
//! valid-in/valid-out handshaking for pipeline composition.

pub mod adder;
pub mod multiplier;
pub mod subtractor;
