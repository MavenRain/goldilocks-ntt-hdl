//! Goldilocks prime field implementation of [`PrimeFieldHdl`].
//!
//! `p = 2^64 - 2^32 + 1` (the Goldilocks prime).
//!
//! The inline arithmetic mirrors the existing helpers in
//! [`super::stage`] (`inline_mod_add`, `inline_mod_sub`,
//! `inline_mul_reduce`).  The Solinas reduction exploits the identity
//! `2^64 ≡ 2^32 - 1 (mod p)` for efficient 128-bit-to-64-bit
//! reduction after multiplication.
//!
//! # Migration path
//!
//! Once the butterfly, twiddle, stage, and pipeline modules are
//! refactored to accept `F: PrimeFieldHdl` as a type parameter, the
//! private inline helpers in `stage.rs` become dead code and this
//! module becomes the single source of truth for Goldilocks arithmetic.

use hdl_cat_error::Error;
use hdl_cat_ir::{BinOp, HdlGraphBuilder, Op, WireId, WireTy};
use hdl_cat_kind::BitSeq;

use crate::hdl::common::{
    bitseq_to_u64, u128_to_bitseq, u64_to_bitseq, zeros_32_bitseq,
    zeros_64_bitseq, GOLDILOCKS_PRIME_U128, GOLDILOCKS_PRIME_U64,
};
use crate::hdl::field_hdl::PrimeFieldHdl;

/// Goldilocks prime field marker type.
///
/// Implements [`PrimeFieldHdl`] with `p = 2^64 - 2^32 + 1`,
/// 64-bit element width, and Solinas reduction.
pub struct Goldilocks;

/// Constant wire handles for Goldilocks inline arithmetic.
///
/// Allocated once per circuit via
/// [`Goldilocks::alloc_constants`] and shared across all
/// add / sub / mul operations within that circuit.
pub struct GoldilocksConstants {
    /// `p` as a 64-bit wire.
    p_64: WireId,
    /// Wrap correction `2^32 - 1` as a 64-bit wire.
    wrap_corr_64: WireId,
    /// Multiplicative identity `1` as a 64-bit wire.
    one_64: WireId,
    /// Additive identity `0` as a 64-bit wire.
    zeros_64: WireId,
    /// `p` as a 128-bit wire.
    p_128: WireId,
    /// `2p` as a 128-bit wire.
    two_p_128: WireId,
    /// Correction `2^32 - 1` as a 128-bit wire.
    corr_128: WireId,
    /// Zero as a 128-bit wire.
    zero_128: WireId,
    /// `2^96` as a 128-bit wire (carry correction).
    carry_96_const: WireId,
    /// 32 zero bits.
    zeros_32: WireId,
}

#[allow(clippy::too_many_lines)]
impl PrimeFieldHdl for Goldilocks {
    type Constants = GoldilocksConstants;

    fn element_width() -> u32 {
        64
    }

    fn alloc_constants(
        bld: HdlGraphBuilder,
    ) -> Result<(HdlGraphBuilder, Self::Constants), Error> {
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

        let p = GOLDILOCKS_PRIME_U128;

        let bld = bld.with_instruction(
            Op::Const {
                bits: u64_to_bitseq(GOLDILOCKS_PRIME_U64),
                ty: WireTy::Bits(64),
            },
            vec![],
            p_64,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: u64_to_bitseq(0xFFFF_FFFF),
                ty: WireTy::Bits(64),
            },
            vec![],
            wrap_corr_64,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: u64_to_bitseq(1),
                ty: WireTy::Bits(64),
            },
            vec![],
            one_64,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: u128_to_bitseq(p),
                ty: WireTy::Bits(128),
            },
            vec![],
            p_128,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: u128_to_bitseq(2 * p),
                ty: WireTy::Bits(128),
            },
            vec![],
            two_p_128,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: u128_to_bitseq(0xFFFF_FFFF),
                ty: WireTy::Bits(128),
            },
            vec![],
            corr_128,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: u128_to_bitseq(0),
                ty: WireTy::Bits(128),
            },
            vec![],
            zero_128,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: u128_to_bitseq(1_u128 << 96),
                ty: WireTy::Bits(128),
            },
            vec![],
            carry_96_const,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: zeros_32_bitseq(),
                ty: WireTy::Bits(32),
            },
            vec![],
            zeros_32,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: zeros_64_bitseq(),
                ty: WireTy::Bits(64),
            },
            vec![],
            zeros_64,
        )?;

        Ok((
            bld,
            GoldilocksConstants {
                p_64,
                wrap_corr_64,
                one_64,
                zeros_64,
                p_128,
                two_p_128,
                corr_128,
                zero_128,
                carry_96_const,
                zeros_32,
            },
        ))
    }

    fn inline_add(
        bld: HdlGraphBuilder,
        a: WireId,
        b: WireId,
        c: &Self::Constants,
    ) -> Result<(HdlGraphBuilder, WireId), Error> {
        // (a + b) mod p with single-correction for Goldilocks.
        // Sum fits in [0, 2p) since both operands are canonical.
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
        let bld =
            bld.with_instruction(Op::Bin(BinOp::Lt), vec![sum64, a], add_overflow)?;
        let bld = bld.with_instruction(
            Op::Bin(BinOp::Add),
            vec![sum64, c.wrap_corr_64],
            sum_plus_wrap,
        )?;
        let bld = bld.with_instruction(
            Op::Bin(BinOp::Lt),
            vec![sum64, c.p_64],
            sum_lt_prime,
        )?;
        let bld =
            bld.with_instruction(Op::Not, vec![sum_lt_prime], sum_ge_prime)?;
        let bld = bld.with_instruction(
            Op::Bin(BinOp::Sub),
            vec![sum64, c.p_64],
            sum_minus_prime,
        )?;
        let bld = bld.with_instruction(
            Op::Mux,
            vec![sum_ge_prime, sum64, sum_minus_prime],
            temp_adj,
        )?;
        let bld = bld.with_instruction(
            Op::Mux,
            vec![add_overflow, temp_adj, sum_plus_wrap],
            adjusted,
        )?;
        let bld = bld.with_instruction(
            Op::Bin(BinOp::Lt),
            vec![adjusted, c.p_64],
            adj_lt_prime,
        )?;
        let bld =
            bld.with_instruction(Op::Not, vec![adj_lt_prime], adj_ge_prime)?;
        let bld = bld.with_instruction(
            Op::Bin(BinOp::Sub),
            vec![adjusted, c.p_64],
            adj_minus_prime,
        )?;
        let bld = bld.with_instruction(
            Op::Mux,
            vec![adj_ge_prime, adjusted, adj_minus_prime],
            result,
        )?;

        Ok((bld, result))
    }

    fn inline_sub(
        bld: HdlGraphBuilder,
        a: WireId,
        b: WireId,
        c: &Self::Constants,
    ) -> Result<(HdlGraphBuilder, WireId), Error> {
        // (a - b) mod p.  If a < b, wrapping subtraction gives
        // a - b + 2^64; correcting by subtracting (2^32 - 1) gives
        // a - b + p (since 2^64 - (2^32 - 1) = p).
        let (bld, diff64) = bld.with_wire(WireTy::Bits(64));
        let (bld, underflow) = bld.with_wire(WireTy::Bit);
        let (bld, diff_minus_corr) = bld.with_wire(WireTy::Bits(64));
        let (bld, result) = bld.with_wire(WireTy::Bits(64));

        let bld =
            bld.with_instruction(Op::Bin(BinOp::Sub), vec![a, b], diff64)?;
        let bld =
            bld.with_instruction(Op::Bin(BinOp::Lt), vec![a, b], underflow)?;
        let bld = bld.with_instruction(
            Op::Bin(BinOp::Sub),
            vec![diff64, c.wrap_corr_64],
            diff_minus_corr,
        )?;
        let bld = bld.with_instruction(
            Op::Mux,
            vec![underflow, diff64, diff_minus_corr],
            result,
        )?;

        Ok((bld, result))
    }

    #[allow(clippy::too_many_lines)]
    fn inline_mul_reduce(
        bld: HdlGraphBuilder,
        a: WireId,
        b: WireId,
        c: &Self::Constants,
    ) -> Result<(HdlGraphBuilder, WireId), Error> {
        // Schoolbook 32x32 partial products + Solinas reduction.
        // Uses the identity 2^64 ≡ 2^32 - 1 (mod p) for efficient
        // 128-bit to 64-bit reduction.

        // ── Split into 32-bit halves ────────────────────────────────
        let (bld, a_lo) = bld.with_wire(WireTy::Bits(32));
        let (bld, a_hi) = bld.with_wire(WireTy::Bits(32));
        let (bld, b_lo) = bld.with_wire(WireTy::Bits(32));
        let (bld, b_hi) = bld.with_wire(WireTy::Bits(32));
        let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![a], a_lo)?;
        let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![a], a_hi)?;
        let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![b], b_lo)?;
        let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![b], b_hi)?;

        // ── Zero-extend to 64 bits ──────────────────────────────────
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

        // ── Four partial products (32x32 -> 64, exact) ──────────────
        let (bld, pp0) = bld.with_wire(WireTy::Bits(64));
        let (bld, pp1) = bld.with_wire(WireTy::Bits(64));
        let (bld, pp2) = bld.with_wire(WireTy::Bits(64));
        let (bld, pp3) = bld.with_wire(WireTy::Bits(64));
        let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_lo_64, b_lo_64], pp0)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_lo_64, b_hi_64], pp1)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_hi_64, b_lo_64], pp2)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_hi_64, b_hi_64], pp3)?;

        // ── Assemble 128-bit product ────────────────────────────────
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

        let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![pp1, pp2], cross_64)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![cross_64, pp1], cross_carry)?;
        let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![cross_64], cross_lo_32)?;
        let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![cross_64], cross_hi_32)?;

        let bld = bld.with_instruction(
            Op::Concat { low_width: 32, high_width: 32 }, vec![c.zeros_32, cross_lo_32], cross_lo_shifted_64,
        )?;
        let bld = bld.with_instruction(
            Op::Concat { low_width: 64, high_width: 64 }, vec![cross_lo_shifted_64, c.zeros_64], cross_lo_shifted_128,
        )?;
        let bld = bld.with_instruction(
            Op::Concat { low_width: 32, high_width: 32 }, vec![cross_hi_32, c.zeros_32], cross_hi_64,
        )?;
        let bld = bld.with_instruction(
            Op::Concat { low_width: 64, high_width: 64 }, vec![c.zeros_64, cross_hi_64], cross_hi_128,
        )?;
        let bld = bld.with_instruction(
            Op::Concat { low_width: 64, high_width: 64 }, vec![pp0, c.zeros_64], pp0_128,
        )?;
        let bld = bld.with_instruction(
            Op::Concat { low_width: 64, high_width: 64 }, vec![c.zeros_64, pp3], pp3_128,
        )?;
        let bld = bld.with_instruction(
            Op::Mux, vec![cross_carry, c.zero_128, c.carry_96_const], carry_contrib,
        )?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![pp0_128, cross_lo_shifted_128], prod_s1)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![prod_s1, cross_hi_128], prod_s2)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![prod_s2, pp3_128], prod_s3)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![prod_s3, carry_contrib], product_128)?;

        // ── Solinas reduction: 128-bit -> 64-bit ────────────────────
        // Extract 32-bit limbs: product = a0 + a1*2^32 + a2*2^64 + a3*2^96
        // Using 2^64 ≡ 2^32 - 1 and 2^96 ≡ -1 (mod p):
        // result ≡ a0 + a1*2^32 + a2*(2^32 - 1) - a3 (mod p)
        //        = a0 + (a1 + a2)*2^32 - a2 - a3 (mod p)
        let (bld, r_a0) = bld.with_wire(WireTy::Bits(32));
        let (bld, r_a1) = bld.with_wire(WireTy::Bits(32));
        let (bld, r_a2) = bld.with_wire(WireTy::Bits(32));
        let (bld, r_a3) = bld.with_wire(WireTy::Bits(32));
        let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![product_128], r_a0)?;
        let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![product_128], r_a1)?;
        let bld = bld.with_instruction(Op::Slice { lo: 64, hi: 96 }, vec![product_128], r_a2)?;
        let bld = bld.with_instruction(Op::Slice { lo: 96, hi: 128 }, vec![product_128], r_a3)?;

        // Zero-extend limbs
        let (bld, r_a1_64) = bld.with_wire(WireTy::Bits(64));
        let (bld, r_a2_64) = bld.with_wire(WireTy::Bits(64));
        let (bld, r_a3_64) = bld.with_wire(WireTy::Bits(64));
        let (bld, r_a1_128) = bld.with_wire(WireTy::Bits(128));
        let (bld, r_a2_128) = bld.with_wire(WireTy::Bits(128));
        let (bld, r_a3_128) = bld.with_wire(WireTy::Bits(128));
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
            Op::Concat { low_width: 32, high_width: 32 }, vec![r_a3, c.zeros_32], r_a3_64,
        )?;
        let bld = bld.with_instruction(
            Op::Concat { low_width: 64, high_width: 64 }, vec![r_a3_64, c.zeros_64], r_a3_128,
        )?;

        // cross = a1 + a2
        let (bld, r_cross_128) = bld.with_wire(WireTy::Bits(128));
        let (bld, r_cross_lo) = bld.with_wire(WireTy::Bits(32));
        let (bld, r_cross_carry_32) = bld.with_wire(WireTy::Bits(32));
        let (bld, r_cross_carry_bit) = bld.with_wire(WireTy::Bit);
        let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![r_a1_128, r_a2_128], r_cross_128)?;
        let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![r_cross_128], r_cross_lo)?;
        let bld = bld.with_instruction(Op::Slice { lo: 32, hi: 64 }, vec![r_cross_128], r_cross_carry_32)?;
        let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 1 }, vec![r_cross_carry_32], r_cross_carry_bit)?;

        // partial = Concat(a0, cross_lo)
        let (bld, r_partial_64) = bld.with_wire(WireTy::Bits(64));
        let (bld, r_partial_128) = bld.with_wire(WireTy::Bits(128));
        let bld = bld.with_instruction(
            Op::Concat { low_width: 32, high_width: 32 }, vec![r_a0, r_cross_lo], r_partial_64,
        )?;
        let bld = bld.with_instruction(
            Op::Concat { low_width: 64, high_width: 64 }, vec![r_partial_64, c.zeros_64], r_partial_128,
        )?;

        // Carry correction and accumulate with bias (add p to stay positive)
        let (bld, r_carry_corr) = bld.with_wire(WireTy::Bits(128));
        let (bld, r_acc1) = bld.with_wire(WireTy::Bits(128));
        let (bld, r_acc2) = bld.with_wire(WireTy::Bits(128));
        let (bld, r_acc3) = bld.with_wire(WireTy::Bits(128));
        let (bld, r_acc4) = bld.with_wire(WireTy::Bits(128));
        let bld = bld.with_instruction(
            Op::Mux, vec![r_cross_carry_bit, c.zero_128, c.corr_128], r_carry_corr,
        )?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![r_partial_128, c.two_p_128], r_acc1)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![r_acc1, r_a2_128], r_acc2)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![r_acc2, r_a3_128], r_acc3)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![r_acc3, r_carry_corr], r_acc4)?;

        // Final reduction: subtract p once or twice to get canonical result
        let (bld, r_lt_2p) = bld.with_wire(WireTy::Bit);
        let (bld, r_ge_2p) = bld.with_wire(WireTy::Bit);
        let (bld, r_step1_sub) = bld.with_wire(WireTy::Bits(128));
        let (bld, r_step1) = bld.with_wire(WireTy::Bits(128));
        let (bld, r_below_p) = bld.with_wire(WireTy::Bit);
        let (bld, r_at_least_p) = bld.with_wire(WireTy::Bit);
        let (bld, r_step2_sub) = bld.with_wire(WireTy::Bits(128));
        let (bld, r_result_128) = bld.with_wire(WireTy::Bits(128));
        let (bld, result) = bld.with_wire(WireTy::Bits(64));

        let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![r_acc4, c.two_p_128], r_lt_2p)?;
        let bld = bld.with_instruction(Op::Not, vec![r_lt_2p], r_ge_2p)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![r_acc4, c.p_128], r_step1_sub)?;
        let bld = bld.with_instruction(Op::Mux, vec![r_ge_2p, r_acc4, r_step1_sub], r_step1)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![r_step1, c.p_128], r_below_p)?;
        let bld = bld.with_instruction(Op::Not, vec![r_below_p], r_at_least_p)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![r_step1, c.p_128], r_step2_sub)?;
        let bld = bld.with_instruction(Op::Mux, vec![r_at_least_p, r_step1, r_step2_sub], r_result_128)?;
        let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 64 }, vec![r_result_128], result)?;

        Ok((bld, result))
    }

    fn prime_u128() -> u128 {
        GOLDILOCKS_PRIME_U128
    }

    fn to_bitseq(val: u64) -> BitSeq {
        u64_to_bitseq(val)
    }

    fn from_bitseq(seq: &BitSeq) -> Result<u64, Error> {
        bitseq_to_u64(seq)
    }

    fn one_wire(c: &Self::Constants) -> WireId {
        c.one_64
    }

    fn zero_wire(c: &Self::Constants) -> WireId {
        c.zeros_64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goldilocks_element_width_is_64() {
        assert_eq!(Goldilocks::element_width(), 64);
    }

    #[test]
    fn goldilocks_prime_matches() {
        assert_eq!(
            Goldilocks::prime_u128(),
            0xFFFF_FFFF_0000_0001_u128
        );
    }

    #[test]
    fn constants_allocate_successfully() -> Result<(), Error> {
        let (bld, _) = HdlGraphBuilder::new().with_wire(WireTy::Bits(64));
        let (_bld, _constants) = Goldilocks::alloc_constants(bld)?;
        Ok(())
    }
}
