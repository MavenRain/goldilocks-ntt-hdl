//! Shared HDL constants and type aliases for the Goldilocks NTT pipeline.
//!
//! hdl-cat circuits consume [`hdl_cat_bits::Bits<64>`] directly for field
//! elements. The simulation boundary (`sim/runner.rs`) performs conversion
//! from [`GoldilocksElement`](crate::field::element::GoldilocksElement) to
//! [`BitSeq`] on input, and the reverse on output.

use hdl_cat_bits::Bits;
use hdl_cat_kind::BitSeq;

/// Width of a Goldilocks field element in bits.
pub const GOLDILOCKS_WIDTH: usize = 64;

/// A Goldilocks field element as an hdl-cat bit vector.
pub type GoldilocksBits = Bits<64>;

/// Type alias for convenience.
pub type GoldilocksElement = Bits<64>;

/// The Goldilocks prime `p = 2^64 − 2^32 + 1`, as a `u128` literal.
///
/// Used by hdl-cat circuits that reduce modulo `p`.  Circuits embed this
/// constant via [`hdl_cat_ir::Op::Const`] instructions.
pub const GOLDILOCKS_PRIME_U128: u128 = 0xFFFF_FFFF_0000_0001_u128;

/// The Goldilocks prime as a `u64`.  Used for software-side comparisons.
pub const GOLDILOCKS_PRIME_U64: u64 = 0xFFFF_FFFF_0000_0001_u64;

/// Extract a `u64` value from a [`BitSeq`] representing a 64-bit Goldilocks element.
///
/// # Errors
///
/// Returns [`hdl_cat_error::Error`] if the [`BitSeq`] has more than 64 bits.
pub fn bitseq_to_u64(seq: &BitSeq) -> Result<u64, hdl_cat_error::Error> {
    if seq.len() > 64 {
        Err(hdl_cat_error::Error::WidthMismatch {
            expected: hdl_cat_error::Width::new(64),
            actual: hdl_cat_error::Width::new(u32::try_from(seq.len()).unwrap_or(u32::MAX)),
        })
    } else {
        Ok((0..seq.len().min(64)).fold(
            0u64,
            |acc, i| {
                if seq.bit(i) { acc | (1u64 << i) } else { acc }
            },
        ))
    }
}

/// Create a [`BitSeq`] from a `u64` Goldilocks element value.
pub fn u64_to_bitseq(val: u64) -> BitSeq {
    let bits: Vec<bool> = (0..64).map(|i| (val >> i) & 1 == 1).collect();
    BitSeq::from_vec(bits)
}

/// Create a 128-bit [`BitSeq`] from a `u128` value.
pub fn u128_to_bitseq(val: u128) -> BitSeq {
    let bits: Vec<bool> = (0..128).map(|i| (val >> i) & 1 == 1).collect();
    BitSeq::from_vec(bits)
}

/// Create a 32-bit [`BitSeq`] of zeros.
pub fn zeros_32_bitseq() -> BitSeq {
    BitSeq::from_vec((0..32).map(|_| false).collect())
}

/// Create a 64-bit [`BitSeq`] of zeros.
pub fn zeros_64_bitseq() -> BitSeq {
    BitSeq::from_vec((0..64).map(|_| false).collect())
}
