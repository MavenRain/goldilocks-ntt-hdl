//! Goldilocks field arithmetic.
//!
//! The Goldilocks prime p = 2^64 - 2^32 + 1 admits efficient modular
//! arithmetic and has roots of unity of order up to 2^32, making it
//! ideal for NTT-based computations.

pub mod element;
pub mod roots;
