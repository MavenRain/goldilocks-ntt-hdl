//! `Io`-wrapped simulation harness.
//!
//! All simulation state is managed through hdl-cat's `Testbench` abstraction,
//! which provides pure functional simulation of circuit graphs.
//! Effects are quarantined inside [`comp_cat_rs::effect::io::Io::suspend`] closures.

pub mod runner;
