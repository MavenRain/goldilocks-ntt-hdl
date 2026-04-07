//! Interpretation layer: maps the NTT graph into HDL module descriptors.
//!
//! This is the bridge between the categorical graph structure and
//! concrete `hdl-cat` modules, using [`comp_cat_rs::collapse::free_category::GraphMorphism`].

pub mod descriptor;
pub mod hdl_morphism;
pub mod signal;
