//! Field-generic HDL arithmetic trait.
//!
//! [`PrimeFieldHdl`] abstracts the inline modular arithmetic used by
//! the SDF pipeline.  Each implementation supplies constant allocation
//! and modular add / sub / multiply-reduce as [`HdlGraphBuilder`]
//! transformers.  This enables the butterfly, twiddle, stage, and
//! pipeline modules to be parameterized over field choice rather than
//! hardcoded to Goldilocks.
//!
//! The trait mirrors the existing inline helpers in [`super::stage`]:
//! `inline_mod_add`, `inline_mod_sub`, and `inline_mul_reduce` become
//! the Goldilocks implementation of the three arithmetic methods.
//!
//! # Design
//!
//! Each method takes ownership of the [`HdlGraphBuilder`] and returns
//! a new builder with the arithmetic instructions appended, plus the
//! output [`WireId`].  This matches hdl-cat's pure, append-only IR
//! construction model.
//!
//! The [`Constants`] associated type holds field-specific constant
//! wire handles (prime, correction factors, identity elements) that
//! are allocated once per circuit and shared across all inline
//! operations within that circuit.

use hdl_cat_error::Error;
use hdl_cat_ir::{HdlGraphBuilder, WireId, WireTy};
use hdl_cat_kind::BitSeq;

/// HDL-level prime field arithmetic.
///
/// Implementations provide constant allocation and inline modular
/// arithmetic for a specific prime field, enabling field-generic
/// NTT pipeline construction.
pub trait PrimeFieldHdl {
    /// Constant wire handles shared by the inline arithmetic
    /// operations.  Allocated once per circuit via
    /// [`alloc_constants`](PrimeFieldHdl::alloc_constants).
    type Constants;

    /// Bit width of a single field element (e.g. 64 for Goldilocks,
    /// 254 for BN254 scalar field, 255 for BLS12-381 scalar field).
    ///
    /// Returns `u32` to match [`WireTy::Bits`] without requiring
    /// a cast.
    fn element_width() -> u32;

    /// The [`WireTy`] for a single field element.
    #[must_use]
    fn element_wire_ty() -> WireTy {
        WireTy::Bits(Self::element_width())
    }

    /// Allocate constant wires and emit their defining instructions.
    ///
    /// Called once at the start of circuit construction.  The returned
    /// [`Constants`](PrimeFieldHdl::Constants) handle is threaded
    /// through all subsequent arithmetic calls.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if IR construction fails.
    fn alloc_constants(bld: HdlGraphBuilder) -> Result<(HdlGraphBuilder, Self::Constants), Error>;

    /// Inline modular addition: `(a + b) mod p`.
    ///
    /// Both `a` and `b` must be canonical (in `[0, p)`).
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if IR construction fails.
    fn inline_add(
        bld: HdlGraphBuilder,
        a: WireId,
        b: WireId,
        c: &Self::Constants,
    ) -> Result<(HdlGraphBuilder, WireId), Error>;

    /// Inline modular subtraction: `(a - b) mod p`.
    ///
    /// Both `a` and `b` must be canonical (in `[0, p)`).
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if IR construction fails.
    fn inline_sub(
        bld: HdlGraphBuilder,
        a: WireId,
        b: WireId,
        c: &Self::Constants,
    ) -> Result<(HdlGraphBuilder, WireId), Error>;

    /// Inline modular multiply-reduce: `(a * b) mod p`.
    ///
    /// Performs the full multiply (producing a double-width
    /// intermediate) and field-specific reduction in one pass.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if IR construction fails.
    fn inline_mul_reduce(
        bld: HdlGraphBuilder,
        a: WireId,
        b: WireId,
        c: &Self::Constants,
    ) -> Result<(HdlGraphBuilder, WireId), Error>;

    /// The prime modulus as a `u128`.
    fn prime_u128() -> u128;

    /// Convert a `u64` value to a [`BitSeq`] of the correct element
    /// width.
    fn to_bitseq(val: u64) -> BitSeq;

    /// Extract a `u64` from a [`BitSeq`].
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if the bit sequence has incorrect width.
    fn from_bitseq(seq: &BitSeq) -> Result<u64, Error>;

    /// The identity element wire from the allocated constants.
    ///
    /// Used by the twiddle accumulator as the reset value.
    fn one_wire(c: &Self::Constants) -> WireId;

    /// The zero element wire from the allocated constants.
    ///
    /// Used for delay line initialization.
    fn zero_wire(c: &Self::Constants) -> WireId;
}
