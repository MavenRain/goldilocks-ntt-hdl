//! Solinas reduction of a 128-bit product to a canonical 64-bit Goldilocks
//! element.
//!
//! The Goldilocks prime `p = 2^64 − 2^32 + 1` admits the identities
//!
//! ```text
//! 2^64  ≡ 2^32 − 1 (mod p)
//! 2^96  ≡ −1       (mod p)
//! ```
//!
//! so that a 128-bit product split into four 32-bit limbs
//! `(hi_hi, hi_lo, lo_hi, lo_lo)` reduces as
//!
//! ```text
//! hi_hi · 2^96 + hi_lo · 2^64 + lo_hi · 2^32 + lo_lo
//!   ≡ −hi_hi + hi_lo · (2^32 − 1) + (lo_hi · 2^32 + lo_lo)   (mod p)
//!   =  lo + (hi_lo << 32) − hi_lo − hi_hi                    (mod p)
//! ```
//!
//! A few conditional adds/subtracts of `p` then bring the value into
//! `[0, p)`.  hdl-cat circuits use 2's-complement wrapping arithmetic,
//! so the intermediate 128-bit values track the mathematical residue
//! modulo `2^128`, from which we recover the residue modulo `p` by
//! comparison and correction.
//!
//! This module exposes [`goldilocks_reduce_arrow`] as a combinational
//! circuit and [`goldilocks_mul_reduce_arrow`] for the full 64×64 multiply
//! and reduce operation.

use hdl_cat_bits::Bits;
use hdl_cat_circuit::{CircuitArrow, Obj};
use hdl_cat_error::Error;
use hdl_cat_ir::{BinOp, HdlGraphBuilder, Op, WireTy};

use crate::hdl::common::{u128_to_bitseq, GOLDILOCKS_PRIME_U128};

/// Type alias for 64-bit Goldilocks field element.
pub type GoldilocksElement = Bits<64>;

/// Circuit arrow for Goldilocks reduction: 128-bit input → 64-bit output.
pub type GoldilocksReduceArrow = CircuitArrow<Obj<Bits<128>>, Obj<GoldilocksElement>>;

/// Circuit arrow for multiply-reduce: (64×64) → 64-bit output.
pub type GoldilocksMulReduceArrow = CircuitArrow<
    hdl_cat_circuit::CircuitTensor<Obj<GoldilocksElement>, Obj<GoldilocksElement>>,
    Obj<GoldilocksElement>,
>;

/// Construct a combinational Goldilocks reduction circuit.
///
/// Takes a 128-bit product and reduces it modulo the Goldilocks prime
/// to produce a canonical 64-bit result in `[0, p)`.
///
/// The algorithm uses the Solinas identity with 128-bit intermediate
/// arithmetic to avoid carry/borrow tracking: it biases the value by
/// adding `p` before any subtractions, guaranteeing all intermediates
/// are non-negative, then normalises with at most two conditional
/// subtractions.
///
/// # Errors
///
/// Returns [`Error`] if IR construction fails.
#[allow(clippy::too_many_lines)]
pub fn goldilocks_reduce_arrow() -> Result<GoldilocksReduceArrow, Error> {
    let p = GOLDILOCKS_PRIME_U128;
    let two_p = 2 * p;
    let corr_val = 0xFFFF_FFFF_u128; // 2^32 - 1

    // ── Wire declarations ────────────────────────────────────────────
    let (bld, product) = HdlGraphBuilder::new().with_wire(WireTy::Bits(128));

    // 32-bit limb wires
    let (bld, a0) = bld.with_wire(WireTy::Bits(32));
    let (bld, a1) = bld.with_wire(WireTy::Bits(32));
    let (bld, a2) = bld.with_wire(WireTy::Bits(32));
    let (bld, a3) = bld.with_wire(WireTy::Bits(32));

    // 128-bit zero-extended limbs
    let (bld, a1_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, a2_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, a2_ext_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, a3_ext_128) = bld.with_wire(WireTy::Bits(128));

    // cross = a1 + a2 (128-bit, exact for 32-bit inputs)
    let (bld, cross_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, cross_lo) = bld.with_wire(WireTy::Bits(32));
    let (bld, cross_carry_32) = bld.with_wire(WireTy::Bits(32));
    let (bld, cross_carry_bit) = bld.with_wire(WireTy::Bit);

    // partial = a0 | (cross_lo << 32) as concatenation
    let (bld, partial_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, partial_128) = bld.with_wire(WireTy::Bits(128));

    // Constants (128-bit)
    let (bld, corr_val_wire) = bld.with_wire(WireTy::Bits(128));
    let (bld, p_wire) = bld.with_wire(WireTy::Bits(128));
    let (bld, two_p_wire) = bld.with_wire(WireTy::Bits(128));
    let (bld, zero_128_wire) = bld.with_wire(WireTy::Bits(128));

    // carry correction (conditional on cross_carry_bit)
    let (bld, carry_corr_128) = bld.with_wire(WireTy::Bits(128));

    // Accumulation
    let (bld, acc1) = bld.with_wire(WireTy::Bits(128)); // partial + carry_corr
    let (bld, acc2) = bld.with_wire(WireTy::Bits(128)); // acc1 + p  (bias)
    let (bld, acc3) = bld.with_wire(WireTy::Bits(128)); // acc2 - a2
    let (bld, acc4) = bld.with_wire(WireTy::Bits(128)); // acc3 - a3

    // Normalisation
    let (bld, lt_2p) = bld.with_wire(WireTy::Bit);
    let (bld, ge_2p) = bld.with_wire(WireTy::Bit);
    let (bld, step1_sub) = bld.with_wire(WireTy::Bits(128));
    let (bld, step1) = bld.with_wire(WireTy::Bits(128));
    let (bld, below_p) = bld.with_wire(WireTy::Bit);
    let (bld, at_least_p) = bld.with_wire(WireTy::Bit);
    let (bld, step2_sub) = bld.with_wire(WireTy::Bits(128));
    let (bld, result_128) = bld.with_wire(WireTy::Bits(128));

    // Final 64-bit output
    let (bld, result) = bld.with_wire(WireTy::Bits(64));

    // ── Constants ────────────────────────────────────────────────────
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(corr_val), ty: WireTy::Bits(128) },
        vec![], corr_val_wire,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(p), ty: WireTy::Bits(128) },
        vec![], p_wire,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(two_p), ty: WireTy::Bits(128) },
        vec![], two_p_wire,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(0), ty: WireTy::Bits(128) },
        vec![], zero_128_wire,
    )?;

    // ── Extract 32-bit limbs ─────────────────────────────────────────
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![product], a0)?;
    let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![product], a1)?;
    let bld = bld.with_instruction(Op::Slice { lo: 64, hi: 96 }, vec![product], a2)?;
    let bld = bld.with_instruction(Op::Slice { lo: 96, hi: 128 }, vec![product], a3)?;

    // ── Zero-extend limbs to 128 bits ────────────────────────────────
    // a1_128, a2_128 for cross computation; a2_ext_128, a3_ext_128 for subtraction.
    // Concat(low=a_32, high=zeros_96) produces 128 bits with the 32-bit
    // value in the low position.  We build the 96-bit zero pad as
    // Concat(zeros_64, zeros_32).  But Concat only takes two inputs, so
    // we use Slice on the 128-bit zero constant instead.
    //
    // Simpler: zero-extend 32→64 via Concat(a, zeros_32), then 64→128
    // via Concat(val_64, zeros_64).

    // a1 → 64 → 128
    let (bld, a1_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, zeros_32_w) = bld.with_wire(WireTy::Bits(32));
    let (bld, zeros_64_w) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(
        Op::Const { bits: crate::hdl::common::zeros_32_bitseq(), ty: WireTy::Bits(32) },
        vec![], zeros_32_w,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: crate::hdl::common::zeros_64_bitseq(), ty: WireTy::Bits(64) },
        vec![], zeros_64_w,
    )?;

    // a1_64 = Concat(low=a1, high=zeros_32) → 64 bits
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![a1, zeros_32_w], a1_64,
    )?;
    // a1_128 = Concat(low=a1_64, high=zeros_64) → 128 bits
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![a1_64, zeros_64_w], a1_128,
    )?;

    // a2 → 64 → 128
    let (bld, a2_64) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![a2, zeros_32_w], a2_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![a2_64, zeros_64_w], a2_128,
    )?;
    // Copy for subtraction (same value, separate wire for clarity)
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![a2_64, zeros_64_w], a2_ext_128,
    )?;

    // a3 → 64 → 128
    let (bld, a3_64) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![a3, zeros_32_w], a3_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![a3_64, zeros_64_w], a3_ext_128,
    )?;

    // ── cross = a1 + a2 (128-bit, exact) ─────────────────────────────
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![a1_128, a2_128], cross_128)?;

    // cross_lo = cross[0:32] (low 32 bits)
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![cross_128], cross_lo)?;

    // cross_carry = cross[32:64] (carry region; only bit 32 can be set)
    let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![cross_128], cross_carry_32)?;
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 1 }, vec![cross_carry_32], cross_carry_bit)?;

    // ── partial = Concat(a0, cross_lo) = a0 + cross_lo * 2^32 ───────
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![a0, cross_lo], partial_64,
    )?;
    // Extend to 128 bits
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![partial_64, zeros_64_w], partial_128,
    )?;

    // ── carry correction ─────────────────────────────────────────────
    // If cross overflowed (carry bit set), add (2^32 - 1) to account
    // for the extra 2^64 ≡ 2^32 - 1 (mod p).
    let bld = bld.with_instruction(
        Op::Mux, vec![cross_carry_bit, zero_128_wire, corr_val_wire], carry_corr_128,
    )?;

    // ── Accumulate with bias ─────────────────────────────────────────
    // acc1 = partial + carry_corr
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![partial_128, carry_corr_128], acc1)?;
    // acc2 = acc1 + p  (bias to ensure all subsequent subtractions stay positive)
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![acc1, p_wire], acc2)?;
    // acc3 = acc2 - a2_ext
    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![acc2, a2_ext_128], acc3)?;
    // acc4 = acc3 - a3_ext
    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![acc3, a3_ext_128], acc4)?;

    // ── Normalisation: at most two conditional subtractions of p ──────
    // acc4 = true_result + p, and true_result ∈ [0, 2p).
    // So acc4 ∈ [p, 3p).  We need:
    //   if acc4 >= 2p: subtract p  →  [p, 2p)
    //   then if >= p: subtract p  →  [0, p)
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![acc4, two_p_wire], lt_2p)?;
    let bld = bld.with_instruction(Op::Not, vec![lt_2p], ge_2p)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![acc4, p_wire], step1_sub)?;
    let bld = bld.with_instruction(Op::Mux, vec![ge_2p, acc4, step1_sub], step1)?;

    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![step1, p_wire], below_p)?;
    let bld = bld.with_instruction(Op::Not, vec![below_p], at_least_p)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![step1, p_wire], step2_sub)?;
    let bld = bld.with_instruction(Op::Mux, vec![at_least_p, step1, step2_sub], result_128)?;

    // ── Extract low 64 bits ──────────────────────────────────────────
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 64 }, vec![result_128], result)?;

    Ok(CircuitArrow::from_raw_parts(
        bld.build(),
        vec![product],
        vec![result],
    ))
}

/// Construct a combinational 64×64 multiply-reduce circuit.
///
/// Multiplies two 64-bit operands to produce a 128-bit intermediate
/// using schoolbook decomposition into four 32×32 partial products,
/// then reduces modulo the Goldilocks prime to 64 bits via
/// the Solinas identity.
///
/// # Errors
///
/// Returns [`Error`] if construction fails.
#[allow(clippy::too_many_lines)]
pub fn goldilocks_mul_reduce_arrow() -> Result<GoldilocksMulReduceArrow, Error> {
    let p = GOLDILOCKS_PRIME_U128;
    let two_p = 2 * p;
    let corr_val = 0xFFFF_FFFF_u128;

    // ── Wire declarations ────────────────────────────────────────────
    let (bld, a) = HdlGraphBuilder::new().with_wire(WireTy::Bits(64));
    let (bld, b) = bld.with_wire(WireTy::Bits(64));

    // Split into 32-bit halves
    let (bld, a_lo) = bld.with_wire(WireTy::Bits(32));
    let (bld, a_hi) = bld.with_wire(WireTy::Bits(32));
    let (bld, b_lo) = bld.with_wire(WireTy::Bits(32));
    let (bld, b_hi) = bld.with_wire(WireTy::Bits(32));

    // Zero-extended to 64 bits for 32×32→64 multiplication
    let (bld, zeros_32_w) = bld.with_wire(WireTy::Bits(32));
    let (bld, zeros_64_w) = bld.with_wire(WireTy::Bits(64));
    let (bld, a_lo_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, a_hi_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, b_lo_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, b_hi_64) = bld.with_wire(WireTy::Bits(64));

    // Four partial products (each 64-bit, exact for 32×32)
    let (bld, p0) = bld.with_wire(WireTy::Bits(64)); // a_lo * b_lo
    let (bld, p1) = bld.with_wire(WireTy::Bits(64)); // a_lo * b_hi
    let (bld, p2) = bld.with_wire(WireTy::Bits(64)); // a_hi * b_lo
    let (bld, p3) = bld.with_wire(WireTy::Bits(64)); // a_hi * b_hi

    // Assemble 128-bit product from partial products
    // product = p0 + (p1 + p2) << 32 + p3 << 64
    let (bld, p0_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, p3_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, cross_64) = bld.with_wire(WireTy::Bits(64));    // p1 + p2
    let (bld, cross_carry) = bld.with_wire(WireTy::Bit);       // overflow from p1+p2
    let (bld, cross_lo_32) = bld.with_wire(WireTy::Bits(32));  // cross[0:32]
    let (bld, cross_hi_32) = bld.with_wire(WireTy::Bits(32));  // cross[32:64]

    // cross_lo << 32 as 64-bit, then extend to 128
    let (bld, cross_lo_shifted_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, cross_lo_shifted_128) = bld.with_wire(WireTy::Bits(128));

    // cross_hi zero-extended → placed at bit 64 in the product
    let (bld, cross_hi_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, cross_hi_128_raw) = bld.with_wire(WireTy::Bits(128));

    // cross carry contributes to bit 96
    let (bld, zero_128_wire) = bld.with_wire(WireTy::Bits(128));
    let (bld, carry_64_const) = bld.with_wire(WireTy::Bits(128));
    let (bld, carry_contribution) = bld.with_wire(WireTy::Bits(128));

    // Assembled 128-bit product
    let (bld, prod_step1) = bld.with_wire(WireTy::Bits(128));
    let (bld, prod_step2) = bld.with_wire(WireTy::Bits(128));
    let (bld, prod_step3) = bld.with_wire(WireTy::Bits(128));
    let (bld, product_128) = bld.with_wire(WireTy::Bits(128));

    // ── Solinas reduction from 128-bit product ───────────────────────
    // (Same algorithm as goldilocks_reduce_arrow but inlined)
    let (bld, r_a0) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_a1) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_a2) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_a3) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_a1_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, r_a2_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, r_a1_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_a2_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_a2_ext_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_a3_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, r_a3_ext_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_cross_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_cross_lo) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_cross_carry_32) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_cross_carry_bit) = bld.with_wire(WireTy::Bit);
    let (bld, r_partial_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, r_partial_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_corr_val_wire) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_p_wire) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_two_p_wire) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_zero_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_carry_corr) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_acc1) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_acc2) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_acc3) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_acc4) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_lt_2p) = bld.with_wire(WireTy::Bit);
    let (bld, r_ge_2p) = bld.with_wire(WireTy::Bit);
    let (bld, r_step1_sub) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_step1) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_below_p) = bld.with_wire(WireTy::Bit);
    let (bld, r_at_least_p) = bld.with_wire(WireTy::Bit);
    let (bld, r_step2_sub) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_result_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, result) = bld.with_wire(WireTy::Bits(64));

    // ── Constants ────────────────────────────────────────────────────
    let bld = bld.with_instruction(
        Op::Const { bits: crate::hdl::common::zeros_32_bitseq(), ty: WireTy::Bits(32) },
        vec![], zeros_32_w,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: crate::hdl::common::zeros_64_bitseq(), ty: WireTy::Bits(64) },
        vec![], zeros_64_w,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(0), ty: WireTy::Bits(128) },
        vec![], zero_128_wire,
    )?;
    // 2^64 as a 128-bit constant (bit 64 set)
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(1_u128 << 64), ty: WireTy::Bits(128) },
        vec![], carry_64_const,
    )?;

    // ── Split operands into 32-bit halves ────────────────────────────
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![a], a_lo)?;
    let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![a], a_hi)?;
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![b], b_lo)?;
    let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![b], b_hi)?;

    // Zero-extend to 64 bits
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![a_lo, zeros_32_w], a_lo_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![a_hi, zeros_32_w], a_hi_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![b_lo, zeros_32_w], b_lo_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![b_hi, zeros_32_w], b_hi_64,
    )?;

    // ── Four partial products (32×32→64, exact) ──────────────────────
    let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_lo_64, b_lo_64], p0)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_lo_64, b_hi_64], p1)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_hi_64, b_lo_64], p2)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_hi_64, b_hi_64], p3)?;

    // ── Assemble 128-bit product ─────────────────────────────────────
    // product = p0 + (p1+p2)<<32 + p3<<64
    //
    // Step 1: cross = p1 + p2 (64-bit wrapping; detect carry)
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![p1, p2], cross_64)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![cross_64, p1], cross_carry)?;

    // Split cross into 32-bit halves
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![cross_64], cross_lo_32)?;
    let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![cross_64], cross_hi_32)?;

    // cross_lo << 32 as 64-bit: Concat(zeros_32, cross_lo_32)
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![zeros_32_w, cross_lo_32],
        cross_lo_shifted_64,
    )?;

    // Extend partial products to 128 bits
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![p0, zeros_64_w], p0_128,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![cross_lo_shifted_64, zeros_64_w],
        cross_lo_shifted_128,
    )?;

    // cross_hi goes to bits 64..96: zero-extend to 64, then place in high half
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![cross_hi_32, zeros_32_w],
        cross_hi_64,
    )?;
    // cross_hi_128 = Concat(zeros_64, cross_hi_64) → cross_hi at bits 64..96
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![zeros_64_w, cross_hi_64],
        cross_hi_128_raw,
    )?;

    // p3 goes to bits 64..128: Concat(zeros_64, p3)
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![zeros_64_w, p3], p3_128,
    )?;

    // carry contribution: if cross overflowed, add 2^64 (at bit position 64+32=96)
    // Actually cross carry means (p1+p2) had a carry into bit 64, which represents
    // 2^64 in the cross value.  Since cross is shifted left by 32, the carry
    // contributes 2^96 to the product.  As a 128-bit value: 1 << 96.
    let (bld, carry_96_const) = bld.with_wire(WireTy::Bits(128));
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(1_u128 << 96), ty: WireTy::Bits(128) },
        vec![], carry_96_const,
    )?;
    let bld = bld.with_instruction(
        Op::Mux, vec![cross_carry, zero_128_wire, carry_96_const], carry_contribution,
    )?;

    // Assemble: product = p0 + cross_lo_shifted + cross_hi_at_64 + p3_at_64 + carry
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![p0_128, cross_lo_shifted_128], prod_step1)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![prod_step1, cross_hi_128_raw], prod_step2)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![prod_step2, p3_128], prod_step3)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![prod_step3, carry_contribution], product_128)?;

    // ── Solinas reduction (inlined from reduce_arrow logic) ──────────
    // Constants
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(corr_val), ty: WireTy::Bits(128) },
        vec![], r_corr_val_wire,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(p), ty: WireTy::Bits(128) },
        vec![], r_p_wire,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(two_p), ty: WireTy::Bits(128) },
        vec![], r_two_p_wire,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(0), ty: WireTy::Bits(128) },
        vec![], r_zero_128,
    )?;

    // Extract 32-bit limbs from product
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![product_128], r_a0)?;
    let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![product_128], r_a1)?;
    let bld = bld.with_instruction(Op::Slice { lo: 64, hi: 96 }, vec![product_128], r_a2)?;
    let bld = bld.with_instruction(Op::Slice { lo: 96, hi: 128 }, vec![product_128], r_a3)?;

    // Zero-extend limbs to 128
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![r_a1, zeros_32_w], r_a1_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![r_a1_64, zeros_64_w], r_a1_128,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![r_a2, zeros_32_w], r_a2_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![r_a2_64, zeros_64_w], r_a2_128,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![r_a2_64, zeros_64_w], r_a2_ext_128,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![r_a3, zeros_32_w], r_a3_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![r_a3_64, zeros_64_w], r_a3_ext_128,
    )?;

    // cross = a1 + a2
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![r_a1_128, r_a2_128], r_cross_128)?;
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![r_cross_128], r_cross_lo)?;
    let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![r_cross_128], r_cross_carry_32)?;
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 1 }, vec![r_cross_carry_32], r_cross_carry_bit)?;

    // partial = Concat(a0, cross_lo)
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![r_a0, r_cross_lo], r_partial_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![r_partial_64, zeros_64_w], r_partial_128,
    )?;

    // Carry correction
    let bld = bld.with_instruction(
        Op::Mux, vec![r_cross_carry_bit, r_zero_128, r_corr_val_wire], r_carry_corr,
    )?;

    // Accumulate with bias
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![r_partial_128, r_carry_corr], r_acc1)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![r_acc1, r_p_wire], r_acc2)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![r_acc2, r_a2_ext_128], r_acc3)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![r_acc3, r_a3_ext_128], r_acc4)?;

    // Normalise
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![r_acc4, r_two_p_wire], r_lt_2p)?;
    let bld = bld.with_instruction(Op::Not, vec![r_lt_2p], r_ge_2p)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![r_acc4, r_p_wire], r_step1_sub)?;
    let bld = bld.with_instruction(Op::Mux, vec![r_ge_2p, r_acc4, r_step1_sub], r_step1)?;

    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![r_step1, r_p_wire], r_below_p)?;
    let bld = bld.with_instruction(Op::Not, vec![r_below_p], r_at_least_p)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![r_step1, r_p_wire], r_step2_sub)?;
    let bld = bld.with_instruction(Op::Mux, vec![r_at_least_p, r_step1, r_step2_sub], r_result_128)?;

    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 64 }, vec![r_result_128], result)?;

    Ok(CircuitArrow::from_raw_parts(
        bld.build(),
        vec![a, b],
        vec![result],
    ))
}

/// Software reference implementation for testing.
#[must_use]
pub fn reference_reduce(product: u128) -> u64 {
    u64::try_from(product % GOLDILOCKS_PRIME_U128).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdl::common::{u64_to_bitseq, GOLDILOCKS_PRIME_U64};
    use hdl_cat_sim::Testbench;

    #[test]
    fn reduce_arrow_builds() -> Result<(), Error> {
        let _arrow = goldilocks_reduce_arrow()?;
        Ok(())
    }

    #[test]
    fn mul_reduce_arrow_builds() -> Result<(), Error> {
        let _arrow = goldilocks_mul_reduce_arrow()?;
        Ok(())
    }

    #[test]
    fn reference_reduce_basic() {
        assert_eq!(reference_reduce(0), 0);
        assert_eq!(reference_reduce(1), 1);
        assert_eq!(reference_reduce(42), 42);
        assert_eq!(reference_reduce(GOLDILOCKS_PRIME_U128), 0);
        assert_eq!(reference_reduce(GOLDILOCKS_PRIME_U128 - 1), GOLDILOCKS_PRIME_U64 - 1);
    }

    #[test]
    fn reference_mul_reduce_basic() {
        assert_eq!(reference_reduce(0), 0);
        assert_eq!(reference_reduce(42), 42);
        assert_eq!(reference_reduce(15), 15);
    }

    #[test]
    fn reduce_arrow_matches_reference() -> Result<(), Error> {
        let arrow = goldilocks_reduce_arrow()?;
        let p = GOLDILOCKS_PRIME_U128;

        let test_cases: Vec<u128> = vec![
            0, 1, 42, p - 1, p, p + 1, 2 * p, 2 * p + 7,
            u128::from(u64::MAX), u128::from(u64::MAX) * u128::from(u64::MAX),
        ];

        for val in test_cases {
            let input = crate::hdl::common::u128_to_bitseq(val);
            let inputs = vec![input];
            let sync = hdl_cat_sync::Sync::lift_comb(arrow.clone());
            let testbench = Testbench::new(sync);
            let result = testbench.run(inputs).run()?;
            let output = crate::hdl::common::bitseq_to_u64(result[0].value())?;
            let expected = reference_reduce(val);
            assert_eq!(output, expected, "reduce({val:#034x}): got {output:#018x}, expected {expected:#018x}");
        }

        Ok(())
    }

    #[test]
    fn mul_reduce_matches_reference() -> Result<(), Error> {
        let arrow = goldilocks_mul_reduce_arrow()?;
        let p64 = GOLDILOCKS_PRIME_U64;

        let test_cases: Vec<(u64, u64)> = vec![
            (0, 0), (1, 1), (1, 42), (3, 5), (7, 11),
            (p64 - 1, 1), (p64 - 1, 2), (p64 - 1, p64 - 1),
            (0xDEAD_BEEF, 0xCAFE_BABE), (u64::MAX, u64::MAX),
        ];

        for (a_val, b_val) in test_cases {
            let input_a = u64_to_bitseq(a_val);
            let input_b = u64_to_bitseq(b_val);
            let inputs = vec![input_a.concat(input_b)];

            let sync = hdl_cat_sync::Sync::lift_comb(arrow.clone());
            let testbench = Testbench::new(sync);
            let result = testbench.run(inputs).run()?;
            let output = crate::hdl::common::bitseq_to_u64(result[0].value())?;
            let expected = reference_reduce(u128::from(a_val) * u128::from(b_val));
            assert_eq!(output, expected,
                "mul_reduce({a_val:#018x}, {b_val:#018x}): got {output:#018x}, expected {expected:#018x}");
        }

        Ok(())
    }
}
