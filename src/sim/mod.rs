//! `Io`-wrapped simulation harness.
//!
//! All mutable `RustHDL` simulation state is quarantined inside
//! [`comp_cat_rs::effect::io::Io::suspend`] closures, maintaining
//! the pure functional interface of the outer layers.

pub mod runner;
