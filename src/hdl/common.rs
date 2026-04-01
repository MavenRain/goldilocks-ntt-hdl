//! Shared HDL type aliases and constants.

use rust_hdl::prelude::*;

/// Goldilocks field element width in bits.
pub const GOLDILOCKS_WIDTH: usize = 64;

/// Convert a `Bits<64>` to a `u64`.
///
/// Extracts each bit via `get_bit` and reconstructs the value.
/// No panicking indexing: `get_bit` returns `false` for out-of-range.
#[must_use]
pub fn bits_to_u64(b: Bits<GOLDILOCKS_WIDTH>) -> u64 {
    (0..GOLDILOCKS_WIDTH).fold(0_u64, |acc, i| {
        acc | (u64::from(b.get_bit(i)) << i)
    })
}

/// Convert a `u64` to `Bits<64>`.
#[must_use]
pub fn u64_to_bits(v: u64) -> Bits<GOLDILOCKS_WIDTH> {
    bits(v)
}

/// Convert a `Bits<32>` to a `u64`.
#[must_use]
pub fn bits32_to_u64(b: Bits<32>) -> u64 {
    (0..32_usize).fold(0_u64, |acc, i| {
        acc | (u64::from(b.get_bit(i)) << i)
    })
}
