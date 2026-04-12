//! Single SDF (Single-path Delay Feedback) stage.
//!
//! Each stage streams one element per cycle and alternates between two
//! phases of length `DEPTH`:
//!
//! - **Fill phase** (counter in `[0, D)`): the incoming element is
//!   written into the delay line (shift register) and passed through
//!   to the output unchanged.
//! - **Butterfly phase** (counter in `[D, 2D)`): the incoming element
//!   pairs with the element from `DEPTH` cycles ago (the shift
//!   register's tail).  The DIF butterfly computes
//!   `upper = (a + b) mod p` and `lower = ((a - b) * twiddle) mod p`.
//!   The upper output is written back into the delay line; the lower
//!   output is emitted downstream.
//!
//! The twiddle accumulator resets to `1` during the fill phase and
//! advances by `step_root` each butterfly cycle, producing the
//! sequence `1, w, w^2, ...` over the D butterfly cycles.
//!
//! This module inlines single-cycle combinational arithmetic (modular
//! add, sub, and 64x64 multiply-reduce) so the SDF phase logic is
//! fully contained in one [`Sync`] machine per stage.

use hdl_cat_bits::Bits;
use hdl_cat_circuit::{CircuitTensor, Obj};
use hdl_cat_error::Error;
use hdl_cat_ir::{BinOp, HdlGraphBuilder, Op, WireId, WireTy};
use hdl_cat_kind::BitSeq;
use hdl_cat_sync::Sync;

use crate::hdl::common::{
    u64_to_bitseq, u128_to_bitseq, zeros_32_bitseq, zeros_64_bitseq,
    GOLDILOCKS_PRIME_U64, GOLDILOCKS_PRIME_U128,
};

/// Counter width used by the SDF phase logic.  Supports depths up to 2^24.
pub const COUNTER_BITS: usize = 24;

/// Type alias for 64-bit Goldilocks field element.
pub type GoldilocksElement = Bits<64>;

/// Input bundle for an SDF stage: `((data, valid), step_root)`.
pub type SdfStageInput = CircuitTensor<
    CircuitTensor<Obj<GoldilocksElement>, Obj<bool>>,
    Obj<GoldilocksElement>,
>;

/// Output bundle for an SDF stage: `(data, valid)`.
pub type SdfStageOutput = CircuitTensor<Obj<GoldilocksElement>, Obj<bool>>;

/// Phantom state marker for SDF stage sync machines.
///
/// The actual state layout (delay registers, twiddle, counter) is managed
/// internally by [`machine::from_raw`]; this type is a convenient phantom
/// for composition and type aliases.
pub type SdfStageState = Obj<GoldilocksElement>;

/// A parametric SDF stage as a sync machine.
pub type SdfStageSync = Sync<SdfStageState, SdfStageInput, SdfStageOutput>;

// ── Shared arithmetic constants ──────────────────────────────────────

/// Constant wire handles shared by the inline modular arithmetic helpers.
struct ArithConstants {
    p_64: WireId,
    wrap_corr_64: WireId,
    one_64: WireId,
    p_128: WireId,
    two_p_128: WireId,
    corr_128: WireId,
    zero_128: WireId,
    carry_96_const: WireId,
    zeros_32: WireId,
    zeros_64: WireId,
}

/// Allocate and emit all constant wires used by the inline arithmetic.
fn alloc_constants(bld: HdlGraphBuilder) -> Result<(HdlGraphBuilder, ArithConstants), Error> {
    let p = GOLDILOCKS_PRIME_U128;

    let (bld, p_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, wrap_corr_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, one_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, p_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, two_p_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, corr_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, zero_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, carry_96_const) = bld.with_wire(WireTy::Bits(128));
    let (bld, zeros_32) = bld.with_wire(WireTy::Bits(32));
    let (bld, zeros_64) = bld.with_wire(WireTy::Bits(64));

    let bld = bld.with_instruction(
        Op::Const { bits: u64_to_bitseq(GOLDILOCKS_PRIME_U64), ty: WireTy::Bits(64) },
        vec![], p_64,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u64_to_bitseq(0xFFFF_FFFF), ty: WireTy::Bits(64) },
        vec![], wrap_corr_64,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u64_to_bitseq(1), ty: WireTy::Bits(64) },
        vec![], one_64,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(p), ty: WireTy::Bits(128) },
        vec![], p_128,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(2 * p), ty: WireTy::Bits(128) },
        vec![], two_p_128,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(0xFFFF_FFFF), ty: WireTy::Bits(128) },
        vec![], corr_128,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(0), ty: WireTy::Bits(128) },
        vec![], zero_128,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: u128_to_bitseq(1_u128 << 96), ty: WireTy::Bits(128) },
        vec![], carry_96_const,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: zeros_32_bitseq(), ty: WireTy::Bits(32) },
        vec![], zeros_32,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: zeros_64_bitseq(), ty: WireTy::Bits(64) },
        vec![], zeros_64,
    )?;

    Ok((bld, ArithConstants {
        p_64, wrap_corr_64, one_64,
        p_128, two_p_128, corr_128, zero_128,
        carry_96_const, zeros_32, zeros_64,
    }))
}

// ── Inline modular arithmetic helpers ────────────────────────────────

/// Compute `(a + b) mod p` inline, returning the output wire.
fn inline_mod_add(
    bld: HdlGraphBuilder, a: WireId, b: WireId, c: &ArithConstants,
) -> Result<(HdlGraphBuilder, WireId), Error> {
    let (bld, sum64) = bld.with_wire(WireTy::Bits(64));
    let (bld, add_overflow) = bld.with_wire(WireTy::Bit);
    let (bld, sum_plus_wrap) = bld.with_wire(WireTy::Bits(64));
    let (bld, sum_lt_prime) = bld.with_wire(WireTy::Bit);
    let (bld, sum_ge_prime) = bld.with_wire(WireTy::Bit);
    let (bld, sum_minus_prime) = bld.with_wire(WireTy::Bits(64));
    let (bld, temp_adj) = bld.with_wire(WireTy::Bits(64));
    let (bld, adjusted) = bld.with_wire(WireTy::Bits(64));
    let (bld, adj_lt_prime) = bld.with_wire(WireTy::Bit);
    let (bld, adj_ge_prime) = bld.with_wire(WireTy::Bit);
    let (bld, adj_minus_prime) = bld.with_wire(WireTy::Bits(64));
    let (bld, result) = bld.with_wire(WireTy::Bits(64));

    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![a, b], sum64)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![sum64, a], add_overflow)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![sum64, c.wrap_corr_64], sum_plus_wrap)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![sum64, c.p_64], sum_lt_prime)?;
    let bld = bld.with_instruction(Op::Not, vec![sum_lt_prime], sum_ge_prime)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![sum64, c.p_64], sum_minus_prime)?;
    let bld = bld.with_instruction(Op::Mux, vec![sum_ge_prime, sum64, sum_minus_prime], temp_adj)?;
    let bld = bld.with_instruction(Op::Mux, vec![add_overflow, temp_adj, sum_plus_wrap], adjusted)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![adjusted, c.p_64], adj_lt_prime)?;
    let bld = bld.with_instruction(Op::Not, vec![adj_lt_prime], adj_ge_prime)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![adjusted, c.p_64], adj_minus_prime)?;
    let bld = bld.with_instruction(Op::Mux, vec![adj_ge_prime, adjusted, adj_minus_prime], result)?;

    Ok((bld, result))
}

/// Compute `(a - b) mod p` inline, returning the output wire.
fn inline_mod_sub(
    bld: HdlGraphBuilder, a: WireId, b: WireId, c: &ArithConstants,
) -> Result<(HdlGraphBuilder, WireId), Error> {
    let (bld, diff64) = bld.with_wire(WireTy::Bits(64));
    let (bld, underflow) = bld.with_wire(WireTy::Bit);
    let (bld, diff_minus_corr) = bld.with_wire(WireTy::Bits(64));
    let (bld, result) = bld.with_wire(WireTy::Bits(64));

    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![a, b], diff64)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![a, b], underflow)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![diff64, c.wrap_corr_64], diff_minus_corr)?;
    let bld = bld.with_instruction(Op::Mux, vec![underflow, diff64, diff_minus_corr], result)?;

    Ok((bld, result))
}

/// Compute `(a * b) mod p` inline via schoolbook 32x32 partial products
/// and Solinas reduction.  Returns the 64-bit result wire.
#[allow(clippy::too_many_lines)]
fn inline_mul_reduce(
    bld: HdlGraphBuilder, a: WireId, b: WireId, c: &ArithConstants,
) -> Result<(HdlGraphBuilder, WireId), Error> {
    // ── Split into 32-bit halves ─────────────────────────────────────
    let (bld, a_lo) = bld.with_wire(WireTy::Bits(32));
    let (bld, a_hi) = bld.with_wire(WireTy::Bits(32));
    let (bld, b_lo) = bld.with_wire(WireTy::Bits(32));
    let (bld, b_hi) = bld.with_wire(WireTy::Bits(32));
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![a], a_lo)?;
    let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![a], a_hi)?;
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![b], b_lo)?;
    let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![b], b_hi)?;

    // ── Zero-extend to 64 bits ───────────────────────────────────────
    let (bld, a_lo_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, a_hi_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, b_lo_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, b_hi_64) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![a_lo, c.zeros_32], a_lo_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![a_hi, c.zeros_32], a_hi_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![b_lo, c.zeros_32], b_lo_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![b_hi, c.zeros_32], b_hi_64,
    )?;

    // ── Four partial products (32x32 -> 64, exact) ───────────────────
    let (bld, pp0) = bld.with_wire(WireTy::Bits(64));
    let (bld, pp1) = bld.with_wire(WireTy::Bits(64));
    let (bld, pp2) = bld.with_wire(WireTy::Bits(64));
    let (bld, pp3) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_lo_64, b_lo_64], pp0)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_lo_64, b_hi_64], pp1)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_hi_64, b_lo_64], pp2)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_hi_64, b_hi_64], pp3)?;

    // ── Assemble 128-bit product: p0 + (p1+p2)<<32 + p3<<64 ─────────
    let (bld, cross_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, cross_carry) = bld.with_wire(WireTy::Bit);
    let (bld, cross_lo_32) = bld.with_wire(WireTy::Bits(32));
    let (bld, cross_hi_32) = bld.with_wire(WireTy::Bits(32));
    let (bld, cross_lo_shifted_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, cross_lo_shifted_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, cross_hi_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, cross_hi_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, pp0_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, pp3_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, carry_contrib) = bld.with_wire(WireTy::Bits(128));
    let (bld, prod_s1) = bld.with_wire(WireTy::Bits(128));
    let (bld, prod_s2) = bld.with_wire(WireTy::Bits(128));
    let (bld, prod_s3) = bld.with_wire(WireTy::Bits(128));
    let (bld, product_128) = bld.with_wire(WireTy::Bits(128));

    // cross = p1 + p2 (64-bit wrapping; detect carry)
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![pp1, pp2], cross_64)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![cross_64, pp1], cross_carry)?;
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![cross_64], cross_lo_32)?;
    let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![cross_64], cross_hi_32)?;

    // cross_lo << 32 as 128-bit
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 },
        vec![c.zeros_32, cross_lo_32], cross_lo_shifted_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 },
        vec![cross_lo_shifted_64, c.zeros_64], cross_lo_shifted_128,
    )?;

    // cross_hi at bits 64..96
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 },
        vec![cross_hi_32, c.zeros_32], cross_hi_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 },
        vec![c.zeros_64, cross_hi_64], cross_hi_128,
    )?;

    // Extend pp0 and pp3 to 128 bits
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 },
        vec![pp0, c.zeros_64], pp0_128,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 },
        vec![c.zeros_64, pp3], pp3_128,
    )?;

    // Carry contribution: if cross overflowed, add 1<<96
    let bld = bld.with_instruction(
        Op::Mux, vec![cross_carry, c.zero_128, c.carry_96_const], carry_contrib,
    )?;

    // Assemble product
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Add), vec![pp0_128, cross_lo_shifted_128], prod_s1,
    )?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Add), vec![prod_s1, cross_hi_128], prod_s2,
    )?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Add), vec![prod_s2, pp3_128], prod_s3,
    )?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Add), vec![prod_s3, carry_contrib], product_128,
    )?;

    // ── Solinas reduction ────────────────────────────────────────────
    let (bld, r_a0) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_a1) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_a2) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_a3) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_a1_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, r_a2_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, r_a3_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, r_a1_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_a2_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_a2_ext_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_a3_ext_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_cross_128) = bld.with_wire(WireTy::Bits(128));
    let (bld, r_cross_lo) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_cross_carry_32) = bld.with_wire(WireTy::Bits(32));
    let (bld, r_cross_carry_bit) = bld.with_wire(WireTy::Bit);
    let (bld, r_partial_64) = bld.with_wire(WireTy::Bits(64));
    let (bld, r_partial_128) = bld.with_wire(WireTy::Bits(128));
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

    // Extract 32-bit limbs from the 128-bit product
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![product_128], r_a0)?;
    let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![product_128], r_a1)?;
    let bld = bld.with_instruction(Op::Slice { lo: 64, hi: 96 }, vec![product_128], r_a2)?;
    let bld = bld.with_instruction(Op::Slice { lo: 96, hi: 128 }, vec![product_128], r_a3)?;

    // Zero-extend limbs to 128 bits
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![r_a1, c.zeros_32], r_a1_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![r_a1_64, c.zeros_64], r_a1_128,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![r_a2, c.zeros_32], r_a2_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![r_a2_64, c.zeros_64], r_a2_128,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![r_a2_64, c.zeros_64], r_a2_ext_128,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![r_a3, c.zeros_32], r_a3_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![r_a3_64, c.zeros_64], r_a3_ext_128,
    )?;

    // cross = a1 + a2 (128-bit exact)
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Add), vec![r_a1_128, r_a2_128], r_cross_128,
    )?;
    let bld = bld.with_instruction(
        Op::Slice { lo: 0, hi: 32 }, vec![r_cross_128], r_cross_lo,
    )?;
    let bld = bld.with_instruction(
        Op::Slice { lo: 32, hi: 64 }, vec![r_cross_128], r_cross_carry_32,
    )?;
    let bld = bld.with_instruction(
        Op::Slice { lo: 0, hi: 1 }, vec![r_cross_carry_32], r_cross_carry_bit,
    )?;

    // partial = Concat(a0, cross_lo)
    let bld = bld.with_instruction(
        Op::Concat { low_width: 32, high_width: 32 }, vec![r_a0, r_cross_lo], r_partial_64,
    )?;
    let bld = bld.with_instruction(
        Op::Concat { low_width: 64, high_width: 64 }, vec![r_partial_64, c.zeros_64], r_partial_128,
    )?;

    // Carry correction: if cross overflowed, add 2^32 - 1
    let bld = bld.with_instruction(
        Op::Mux, vec![r_cross_carry_bit, c.zero_128, c.corr_128], r_carry_corr,
    )?;

    // Accumulate with bias (add p to keep positive through subtractions)
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Add), vec![r_partial_128, r_carry_corr], r_acc1,
    )?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Add), vec![r_acc1, c.p_128], r_acc2,
    )?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Sub), vec![r_acc2, r_a2_ext_128], r_acc3,
    )?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Sub), vec![r_acc3, r_a3_ext_128], r_acc4,
    )?;

    // Normalise: at most two conditional subtractions of p
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Lt), vec![r_acc4, c.two_p_128], r_lt_2p,
    )?;
    let bld = bld.with_instruction(Op::Not, vec![r_lt_2p], r_ge_2p)?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Sub), vec![r_acc4, c.p_128], r_step1_sub,
    )?;
    let bld = bld.with_instruction(
        Op::Mux, vec![r_ge_2p, r_acc4, r_step1_sub], r_step1,
    )?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Lt), vec![r_step1, c.p_128], r_below_p,
    )?;
    let bld = bld.with_instruction(Op::Not, vec![r_below_p], r_at_least_p)?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Sub), vec![r_step1, c.p_128], r_step2_sub,
    )?;
    let bld = bld.with_instruction(
        Op::Mux, vec![r_at_least_p, r_step1, r_step2_sub], r_result_128,
    )?;

    // Extract low 64 bits
    let bld = bld.with_instruction(
        Op::Slice { lo: 0, hi: 64 }, vec![r_result_128], result,
    )?;

    Ok((bld, result))
}

// ── Counter helper ───────────────────────────────────────────────────

/// Create a [`COUNTER_BITS`]-wide [`BitSeq`] from a `u32`.
fn counter_to_bitseq(val: u32) -> BitSeq {
    BitSeq::from_vec(
        (0..COUNTER_BITS).map(|i| (val >> i) & 1 == 1).collect(),
    )
}

// ── SDF stage constructor ────────────────────────────────────────────

/// Construct an SDF stage with the given delay depth.
///
/// The stage implements the SDF algorithm with fill and butterfly phases,
/// coordinated by an internal counter.  The delay line is a D-deep shift
/// register.  All modular arithmetic (add, sub, 64x64 multiply-reduce)
/// is inlined as combinational logic.
///
/// # Arguments
///
/// * `depth` - The delay depth for this stage (must be >= 1)
///
/// # Errors
///
/// Returns [`Error`] if `depth` is zero or IR construction fails.
#[allow(clippy::too_many_lines)]
pub fn sdf_stage(depth: usize) -> Result<SdfStageSync, Error> {
    if depth == 0 {
        return Err(Error::WidthMismatch {
            expected: hdl_cat_error::Width::new(1),
            actual: hdl_cat_error::Width::new(0),
        });
    }

    let counter_bits = u32::try_from(COUNTER_BITS).unwrap_or(16);

    // ── Source wire allocation ────────────────────────────────────────
    // State wires: delay_0 .. delay_{D-1}, twiddle, counter
    let (bld, delay_wires) = (0..depth).fold(
        (HdlGraphBuilder::new(), Vec::new()),
        |(b, v), _| {
            let (b, w) = b.with_wire(WireTy::Bits(64));
            (b, v.into_iter().chain(core::iter::once(w)).collect())
        },
    );

    let (bld, twiddle) = bld.with_wire(WireTy::Bits(64));
    let (bld, counter) = bld.with_wire(WireTy::Bits(counter_bits));

    // Data input wires
    let (bld, data_in) = bld.with_wire(WireTy::Bits(64));
    let (bld, valid_in) = bld.with_wire(WireTy::Bit);
    let (bld, step_root) = bld.with_wire(WireTy::Bits(64));

    // ── Shared constants ─────────────────────────────────────────────
    let (bld, arith) = alloc_constants(bld)?;

    // ── Counter constants (16-bit) ───────────────────────────────────
    let depth_u32 = u32::try_from(depth).map_err(|_| Error::Overflow {
        width: hdl_cat_error::Width::new(u32::MAX),
    })?;
    let max_counter_u32 = u32::try_from(2 * depth - 1).map_err(|_| Error::Overflow {
        width: hdl_cat_error::Width::new(u32::MAX),
    })?;

    let (bld, depth_const) = bld.with_wire(WireTy::Bits(counter_bits));
    let (bld, max_counter_const) = bld.with_wire(WireTy::Bits(counter_bits));
    let (bld, one_ctr) = bld.with_wire(WireTy::Bits(counter_bits));
    let (bld, zero_ctr) = bld.with_wire(WireTy::Bits(counter_bits));

    let bld = bld.with_instruction(
        Op::Const { bits: counter_to_bitseq(depth_u32), ty: WireTy::Bits(counter_bits) },
        vec![], depth_const,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: counter_to_bitseq(max_counter_u32), ty: WireTy::Bits(counter_bits) },
        vec![], max_counter_const,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: counter_to_bitseq(1), ty: WireTy::Bits(counter_bits) },
        vec![], one_ctr,
    )?;
    let bld = bld.with_instruction(
        Op::Const { bits: counter_to_bitseq(0), ty: WireTy::Bits(counter_bits) },
        vec![], zero_ctr,
    )?;

    // ── Phase logic ──────────────────────────────────────────────────
    // phase = 0 (fill) when counter < depth; phase = 1 (butterfly) otherwise
    let (bld, counter_lt_depth) = bld.with_wire(WireTy::Bit);
    let (bld, phase) = bld.with_wire(WireTy::Bit);

    let bld = bld.with_instruction(
        Op::Bin(BinOp::Lt), vec![counter, depth_const], counter_lt_depth,
    )?;
    let bld = bld.with_instruction(Op::Not, vec![counter_lt_depth], phase)?;

    // ── Read delayed value (tail of shift register) ──────────────────
    let (bld, delayed) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(
        Op::Slice { lo: 0, hi: 64 }, vec![delay_wires[depth - 1]], delayed,
    )?;

    // ── Butterfly arithmetic (always computed; muxed by phase) ────────
    let (bld, upper) = inline_mod_add(bld, delayed, data_in, &arith)?;
    let (bld, diff) = inline_mod_sub(bld, delayed, data_in, &arith)?;
    let (bld, lower) = inline_mul_reduce(bld, diff, twiddle, &arith)?;

    // ── Twiddle advancement ──────────────────────────────────────────
    let (bld, twiddle_stepped) = inline_mul_reduce(bld, twiddle, step_root, &arith)?;

    // ── Phase-dependent muxing ───────────────────────────────────────
    // data_out: fill -> data_in (pass-through), butterfly -> lower
    let (bld, data_out) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(
        Op::Mux, vec![phase, data_in, lower], data_out,
    )?;

    // valid_out: identity copy
    let (bld, valid_out) = bld.with_wire(WireTy::Bit);
    let bld = bld.with_instruction(
        Op::Slice { lo: 0, hi: 1 }, vec![valid_in], valid_out,
    )?;

    // delay_in: fill -> data_in, butterfly -> upper
    let (bld, delay_in) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(
        Op::Mux, vec![phase, data_in, upper], delay_in,
    )?;

    // next_twiddle: fill -> 1, butterfly -> twiddle * step_root
    let (bld, next_twiddle) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(
        Op::Mux, vec![phase, arith.one_64, twiddle_stepped], next_twiddle,
    )?;

    // ── Counter logic ────────────────────────────────────────────────
    let (bld, counter_inc) = bld.with_wire(WireTy::Bits(counter_bits));
    let (bld, counter_at_max) = bld.with_wire(WireTy::Bit);
    let (bld, next_counter) = bld.with_wire(WireTy::Bits(counter_bits));

    let bld = bld.with_instruction(
        Op::Bin(BinOp::Add), vec![counter, one_ctr], counter_inc,
    )?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Eq), vec![counter, max_counter_const], counter_at_max,
    )?;
    // at_max=0 -> counter_inc, at_max=1 -> zero (wrap)
    let bld = bld.with_instruction(
        Op::Mux, vec![counter_at_max, counter_inc, zero_ctr], next_counter,
    )?;

    // ── Delay line shift register ────────────────────────────────────
    // next_delay_0 = delay_in (identity copy via Slice)
    let (bld, next_delay_0) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(
        Op::Slice { lo: 0, hi: 64 }, vec![delay_in], next_delay_0,
    )?;

    // next_delay_i = delay_{i-1} for i in 1..D
    let (bld, next_delay_wires) = (1..depth).try_fold(
        (bld, vec![next_delay_0]),
        |acc, i| {
            let (b, v) = acc;
            let (b, w) = b.with_wire(WireTy::Bits(64));
            let b = b.with_instruction(
                Op::Slice { lo: 0, hi: 64 }, vec![delay_wires[i - 1]], w,
            )?;
            Ok::<_, Error>((b, v.into_iter().chain(core::iter::once(w)).collect()))
        },
    )?;

    // ── Build the graph and assemble the Sync machine ────────────────
    let graph = bld.build();

    // Input wires: state ++ data
    // State: [delay_0 .. delay_{D-1}, twiddle, counter]
    // Data:  [data_in, valid_in, step_root]
    let input_wires: Vec<WireId> = delay_wires.iter().copied()
        .chain(core::iter::once(twiddle))
        .chain(core::iter::once(counter))
        .chain(core::iter::once(data_in))
        .chain(core::iter::once(valid_in))
        .chain(core::iter::once(step_root))
        .collect();

    // Output wires: next_state ++ data_output
    // Next state: [next_delay_0 .. next_delay_{D-1}, next_twiddle, next_counter]
    // Data out:   [data_out, valid_out]
    let output_wires: Vec<WireId> = next_delay_wires.iter().copied()
        .chain(core::iter::once(next_twiddle))
        .chain(core::iter::once(next_counter))
        .chain(core::iter::once(data_out))
        .chain(core::iter::once(valid_out))
        .collect();

    let state_wire_count = depth + 2; // D delays + twiddle + counter

    // Initial state: delays = 0, twiddle = 1, counter = 0
    let initial_state = BitSeq::from_vec(
        (0..64 * depth).map(|_| false)           // delay registers: zero
            .chain(core::iter::once(true))        // twiddle bit 0 = 1
            .chain((1..64).map(|_| false))        // twiddle remaining bits
            .chain((0..COUNTER_BITS).map(|_| false)) // counter: zero
            .collect(),
    );

    Ok(hdl_cat_sync::machine::from_raw(
        graph, input_wires, output_wires, initial_state, state_wire_count,
    ))
}

/// Create a depth-2 SDF stage (for testing).
///
/// # Errors
///
/// Returns [`Error`] if IR construction fails.
pub fn sdf_stage_depth_2() -> Result<SdfStageSync, Error> {
    sdf_stage(2)
}

/// Create a depth-1 SDF stage (for testing).
///
/// # Errors
///
/// Returns [`Error`] if IR construction fails.
pub fn sdf_stage_depth_1() -> Result<SdfStageSync, Error> {
    sdf_stage(1)
}

/// Software reference for a single DIF butterfly computation.
#[must_use]
pub fn reference_dif_butterfly(a: u64, b: u64, twiddle: u64) -> (u64, u64) {
    let p = u128::from(GOLDILOCKS_PRIME_U64);
    let upper = u64::try_from((u128::from(a) + u128::from(b)) % p).unwrap_or(0);
    let diff = (u128::from(a) + p - u128::from(b)) % p;
    let lower = u64::try_from((diff * u128::from(twiddle)) % p).unwrap_or(0);
    (upper, lower)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdl::common::{bitseq_to_u64, u64_to_bitseq};
    use hdl_cat_sim::Testbench;

    /// Pack a stage input cycle into a [`BitSeq`].
    fn make_input(data: u64, valid: bool, step_root: u64) -> BitSeq {
        u64_to_bitseq(data)
            .concat(BitSeq::from_vec(vec![valid]))
            .concat(u64_to_bitseq(step_root))
    }

    /// Unpack a stage output cycle from a [`BitSeq`].
    fn read_output(bits: &BitSeq) -> Result<(u64, bool), Error> {
        let (data_bits, valid_bits) = bits.clone().split_at(64);
        let data = bitseq_to_u64(&data_bits)?;
        let valid = valid_bits.bit(0);
        Ok((data, valid))
    }

    // ── Software reference SDF state machine ─────────────────────────

    struct RefState {
        delay: Vec<u64>,
        twiddle: u64,
        counter: usize,
        depth: usize,
    }

    impl RefState {
        fn new(depth: usize) -> Self {
            Self { delay: vec![0; depth], twiddle: 1, counter: 0, depth }
        }

        fn step(self, data_in: u64, step_root: u64) -> (Self, u64) {
            let p = u128::from(GOLDILOCKS_PRIME_U64);
            let is_butterfly = self.counter >= self.depth;
            let delayed = self.delay[self.depth - 1];

            let (data_out, delay_in, next_tw) = if is_butterfly {
                let upper = u64::try_from(
                    (u128::from(delayed) + u128::from(data_in)) % p,
                ).unwrap_or(0);
                let diff = (u128::from(delayed) + p - u128::from(data_in)) % p;
                let lower = u64::try_from(
                    (diff * u128::from(self.twiddle)) % p,
                ).unwrap_or(0);
                let tw = u64::try_from(
                    (u128::from(self.twiddle) * u128::from(step_root)) % p,
                ).unwrap_or(0);
                (lower, upper, tw)
            } else {
                (data_in, data_in, 1)
            };

            // Shift register: delay_in enters at position 0
            let next_delay: Vec<u64> = core::iter::once(delay_in)
                .chain(self.delay[..self.depth - 1].iter().copied())
                .collect();

            let next_counter = if self.counter == 2 * self.depth - 1 {
                0
            } else {
                self.counter + 1
            };

            (RefState {
                delay: next_delay,
                twiddle: next_tw,
                counter: next_counter,
                depth: self.depth,
            }, data_out)
        }
    }

    // ── Tests ────────────────────────────────────────────────────────

    #[test]
    fn stage_depth_1_builds() -> Result<(), Error> {
        let _s = sdf_stage_depth_1()?;
        Ok(())
    }

    #[test]
    fn stage_depth_2_builds() -> Result<(), Error> {
        let _s = sdf_stage_depth_2()?;
        Ok(())
    }

    #[test]
    fn stage_depth_0_errors() {
        assert!(sdf_stage(0).is_err());
    }

    #[test]
    fn reference_butterfly_basic() {
        let (upper, lower) = reference_dif_butterfly(10, 5, 2);
        assert_eq!(upper, 15);
        assert_eq!(lower, 10); // (10 - 5) * 2 = 10
    }

    #[test]
    fn depth_1_fill_pass_through() -> Result<(), Error> {
        let stage = sdf_stage(1)?;
        let step_root = 7_u64; // arbitrary

        // Depth 1: fill for 1 cycle, butterfly for 1 cycle.
        // Cycle 0 (fill): output should be data_in.
        let inputs = vec![
            make_input(42, true, step_root),
            make_input(99, true, step_root), // butterfly cycle (not checked here)
        ];

        let tb = Testbench::new(stage);
        let results = tb.run(inputs).run()?;

        let (out_0, valid_0) = read_output(results[0].value())?;
        assert_eq!(out_0, 42, "fill pass-through failed");
        assert!(valid_0, "valid should be true");

        Ok(())
    }

    #[test]
    fn depth_1_matches_reference() -> Result<(), Error> {
        let stage = sdf_stage(1)?;
        let step_root = 1_u64; // simplest twiddle

        // Feed 4 cycles (2 periods of 2D=2 each)
        let data: Vec<u64> = vec![10, 20, 30, 40];

        let inputs: Vec<BitSeq> = data.iter()
            .map(|d| make_input(*d, true, step_root))
            .collect();

        let tb = Testbench::new(stage);
        let results = tb.run(inputs).run()?;

        // Compare against software reference
        let (_, ref_outputs) = data.iter().fold(
            (RefState::new(1), Vec::new()),
            |(state, outs), d| {
                let (next_state, out) = state.step(*d, step_root);
                (next_state, outs.into_iter().chain(core::iter::once(out)).collect())
            },
        );

        results.iter().zip(ref_outputs.iter()).enumerate().try_for_each(
            |(i, (sample, expected))| {
                let (actual, _) = read_output(sample.value())?;
                assert_eq!(
                    actual, *expected,
                    "cycle {i}: got {actual:#018x}, expected {expected:#018x}",
                );
                Ok::<(), Error>(())
            },
        )?;

        Ok(())
    }

    #[test]
    fn depth_2_matches_reference() -> Result<(), Error> {
        let stage = sdf_stage(2)?;
        let step_root = 1_u64;

        // Feed 8 cycles (2 periods of 2D=4 each)
        let data: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let inputs: Vec<BitSeq> = data.iter()
            .map(|d| make_input(*d, true, step_root))
            .collect();

        let tb = Testbench::new(stage);
        let results = tb.run(inputs).run()?;

        let (_, ref_outputs) = data.iter().fold(
            (RefState::new(2), Vec::new()),
            |(state, outs), d| {
                let (next_state, out) = state.step(*d, step_root);
                (next_state, outs.into_iter().chain(core::iter::once(out)).collect())
            },
        );

        results.iter().zip(ref_outputs.iter()).enumerate().try_for_each(
            |(i, (sample, expected))| {
                let (actual, _) = read_output(sample.value())?;
                assert_eq!(
                    actual, *expected,
                    "cycle {i}: got {actual:#018x}, expected {expected:#018x}",
                );
                Ok::<(), Error>(())
            },
        )?;

        Ok(())
    }

    #[test]
    fn depth_2_nontrivial_twiddle() -> Result<(), Error> {
        let stage = sdf_stage(2)?;
        let step_root = 7_u64; // non-trivial twiddle step

        let data: Vec<u64> = vec![100, 200, 300, 400];

        let inputs: Vec<BitSeq> = data.iter()
            .map(|d| make_input(*d, true, step_root))
            .collect();

        let tb = Testbench::new(stage);
        let results = tb.run(inputs).run()?;

        let (_, ref_outputs) = data.iter().fold(
            (RefState::new(2), Vec::new()),
            |(state, outs), d| {
                let (next_state, out) = state.step(*d, step_root);
                (next_state, outs.into_iter().chain(core::iter::once(out)).collect())
            },
        );

        results.iter().zip(ref_outputs.iter()).enumerate().try_for_each(
            |(i, (sample, expected))| {
                let (actual, _) = read_output(sample.value())?;
                assert_eq!(
                    actual, *expected,
                    "cycle {i}: got {actual:#018x}, expected {expected:#018x}",
                );
                Ok::<(), Error>(())
            },
        )?;

        Ok(())
    }
}
