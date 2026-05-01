//! `BabyBear` prime field implementation of [`PrimeFieldHdl`].
//!
//! `p = 2^31 - 2^27 + 1 = 2_013_265_921` (the `BabyBear` prime used
//! by Plonky3 and ICICLE).  Elements are stored in 32-bit wires in
//! canonical form (value in `[0, p)`).
//!
//! Modular reduction after multiplication uses the Solinas identity
//! `2^31 ≡ 2^27 − 1 (mod p)`: given a 64-bit product `T = T_lo +
//! T_hi * 2^31`, we have `T ≡ T_lo + T_hi * (2^27 − 1) (mod p)`.  A
//! single fold contracts the width by roughly 4 bits; twelve folds
//! bring any value below `2p`, after which one conditional
//! subtraction of `p` yields canonical form.
//!
//! Unlike Goldilocks' 64-bit Solinas form, `BabyBear` does not
//! admit a constant-depth single-step reduction, so the pass-based
//! cascade is the simplest correct formulation.
//!
//! # Pass analysis
//!
//! At pass `k` with input wire holding content of width `W_k`,
//! `T_hi` occupies `W_k − 31` bits, and the output is bounded by
//! `2^31 + 2^(W_k − 4)`.  Starting from a 62-bit product:
//!
//! | Pass | Input width | Output bound |
//! |---|---|---|
//! | 0 | 62 | `~2^58` |
//! | 1 | 58 | `~2^54` |
//! | 2 | 54 | `~2^50` |
//! | 3 | 50 | `~2^46` |
//! | 4 | 46 | `~2^42` |
//! | 5 | 42 | `~2^38` |
//! | 6 | 38 | `~2^34` |
//! | 7 | 34 | `~2^31 + 2^30` |
//! | 8-11 | ≤ 34 | `< 2^31 + 2^27` |
//!
//! After twelve passes the value fits in 32 bits and is `< 2p`, so
//! one conditional `−p` yields the canonical result.

use hdl_cat_error::Error;
use hdl_cat_ir::{BinOp, HdlGraphBuilder, Op, WireId, WireTy};
use hdl_cat_kind::BitSeq;

use crate::hdl::field_hdl::PrimeFieldHdl;

/// The `BabyBear` prime as a `u64`: `2^31 − 2^27 + 1`.
pub const BABYBEAR_PRIME_U64: u64 = 0x7800_0001;

/// The `BabyBear` prime as a `u128`.
pub const BABYBEAR_PRIME_U128: u128 = 0x7800_0001;

/// Number of Solinas folding passes applied during `mul_reduce`.
const REDUCTION_PASSES: u32 = 12;

/// `BabyBear` prime field marker type.
///
/// Implements [`PrimeFieldHdl`] with `p = 2^31 − 2^27 + 1`, 32-bit
/// element width, and cascaded Solinas reduction.
pub struct BabyBear;

/// Constant wire handles for `BabyBear` inline arithmetic.
///
/// Allocated once per circuit via
/// [`BabyBear::alloc_constants`] and shared across all
/// add / sub / mul operations within that circuit.
pub struct BabyBearConstants {
    /// `p` as a 32-bit wire.
    p_32: WireId,
    /// Multiplicative identity `1` as a 32-bit wire.
    one_32: WireId,
    /// 4-bit zero constant (for widening `T_hi` 33→37).
    zeros_4: WireId,
    /// 27-bit zero constant (for the `<< 27` Concat).
    zeros_27: WireId,
    /// 31-bit zero constant (for widening `T_hi` 33→64).
    zeros_31: WireId,
    /// 32-bit zero constant, also the additive identity wire.
    zeros_32: WireId,
    /// 33-bit zero constant (for widening `T_lo` 31→64).
    zeros_33: WireId,
}

/// Create an `N`-bit [`BitSeq`] of zeros.
fn zeros_bitseq(width: u32) -> BitSeq {
    BitSeq::from_vec((0..width).map(|_| false).collect())
}

/// Create a 32-bit [`BitSeq`] holding the low 32 bits of `val`.
fn u32_bitseq(val: u64) -> BitSeq {
    BitSeq::from_vec((0..32).map(|i| (val >> i) & 1 == 1).collect())
}

impl BabyBear {
    /// Apply one Solinas fold at 64-bit width.
    ///
    /// Interprets `cur` as `T = T_lo + T_hi * 2^31` with `T_lo`
    /// the low 31 bits and `T_hi` the high 33 bits, and produces
    /// `T_lo + T_hi * (2^27 − 1)` as a new 64-bit wire.
    fn solinas_pass(
        bld: HdlGraphBuilder,
        cur: WireId,
        c: &BabyBearConstants,
    ) -> Result<(HdlGraphBuilder, WireId), Error> {
        let (bld, t_lo) = bld.with_wire(WireTy::Bits(31));
        let (bld, t_hi) = bld.with_wire(WireTy::Bits(33));
        let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 31 }, vec![cur], t_lo)?;
        let bld = bld.with_instruction(Op::Slice { lo: 31, hi: 64 }, vec![cur], t_hi)?;

        let (bld, t_lo_64) = bld.with_wire(WireTy::Bits(64));
        let bld = bld.with_instruction(
            Op::Concat {
                low_width: 31,
                high_width: 33,
            },
            vec![t_lo, c.zeros_33],
            t_lo_64,
        )?;

        let (bld, t_hi_64) = bld.with_wire(WireTy::Bits(64));
        let bld = bld.with_instruction(
            Op::Concat {
                low_width: 33,
                high_width: 31,
            },
            vec![t_hi, c.zeros_31],
            t_hi_64,
        )?;

        let (bld, t_hi_37) = bld.with_wire(WireTy::Bits(37));
        let bld = bld.with_instruction(
            Op::Concat {
                low_width: 33,
                high_width: 4,
            },
            vec![t_hi, c.zeros_4],
            t_hi_37,
        )?;
        let (bld, t_hi_shl_27) = bld.with_wire(WireTy::Bits(64));
        let bld = bld.with_instruction(
            Op::Concat {
                low_width: 27,
                high_width: 37,
            },
            vec![c.zeros_27, t_hi_37],
            t_hi_shl_27,
        )?;

        let (bld, sum) = bld.with_wire(WireTy::Bits(64));
        let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![t_lo_64, t_hi_shl_27], sum)?;
        let (bld, next) = bld.with_wire(WireTy::Bits(64));
        let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![sum, t_hi_64], next)?;

        Ok((bld, next))
    }
}

impl PrimeFieldHdl for BabyBear {
    type Constants = BabyBearConstants;

    fn element_width() -> u32 {
        32
    }

    fn alloc_constants(bld: HdlGraphBuilder) -> Result<(HdlGraphBuilder, Self::Constants), Error> {
        let (bld, p_32) = bld.with_wire(WireTy::Bits(32));
        let (bld, one_32) = bld.with_wire(WireTy::Bits(32));
        let (bld, zeros_4) = bld.with_wire(WireTy::Bits(4));
        let (bld, zeros_27) = bld.with_wire(WireTy::Bits(27));
        let (bld, zeros_31) = bld.with_wire(WireTy::Bits(31));
        let (bld, zeros_32) = bld.with_wire(WireTy::Bits(32));
        let (bld, zeros_33) = bld.with_wire(WireTy::Bits(33));

        let bld = bld.with_instruction(
            Op::Const {
                bits: u32_bitseq(BABYBEAR_PRIME_U64),
                ty: WireTy::Bits(32),
            },
            vec![],
            p_32,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: u32_bitseq(1),
                ty: WireTy::Bits(32),
            },
            vec![],
            one_32,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: zeros_bitseq(4),
                ty: WireTy::Bits(4),
            },
            vec![],
            zeros_4,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: zeros_bitseq(27),
                ty: WireTy::Bits(27),
            },
            vec![],
            zeros_27,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: zeros_bitseq(31),
                ty: WireTy::Bits(31),
            },
            vec![],
            zeros_31,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: zeros_bitseq(32),
                ty: WireTy::Bits(32),
            },
            vec![],
            zeros_32,
        )?;
        let bld = bld.with_instruction(
            Op::Const {
                bits: zeros_bitseq(33),
                ty: WireTy::Bits(33),
            },
            vec![],
            zeros_33,
        )?;

        Ok((
            bld,
            BabyBearConstants {
                p_32,
                one_32,
                zeros_4,
                zeros_27,
                zeros_31,
                zeros_32,
                zeros_33,
            },
        ))
    }

    fn inline_add(
        bld: HdlGraphBuilder,
        a: WireId,
        b: WireId,
        c: &Self::Constants,
    ) -> Result<(HdlGraphBuilder, WireId), Error> {
        // sum = a + b  (fits in 32 bits since a, b < p < 2^31, so sum < 2p < 2^32)
        // if sum >= p: sum -= p
        let (bld, sum) = bld.with_wire(WireTy::Bits(32));
        let (bld, sum_lt_p) = bld.with_wire(WireTy::Bit);
        let (bld, sum_ge_p) = bld.with_wire(WireTy::Bit);
        let (bld, sum_minus_p) = bld.with_wire(WireTy::Bits(32));
        let (bld, result) = bld.with_wire(WireTy::Bits(32));

        let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![a, b], sum)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![sum, c.p_32], sum_lt_p)?;
        let bld = bld.with_instruction(Op::Not, vec![sum_lt_p], sum_ge_p)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![sum, c.p_32], sum_minus_p)?;
        let bld = bld.with_instruction(Op::Mux, vec![sum_ge_p, sum, sum_minus_p], result)?;

        Ok((bld, result))
    }

    fn inline_sub(
        bld: HdlGraphBuilder,
        a: WireId,
        b: WireId,
        c: &Self::Constants,
    ) -> Result<(HdlGraphBuilder, WireId), Error> {
        // diff = a - b (32-bit wrap).  If a < b, diff = a - b + 2^32.
        // Want a - b + p instead: result = diff + p (mod 2^32).
        let (bld, diff) = bld.with_wire(WireTy::Bits(32));
        let (bld, underflow) = bld.with_wire(WireTy::Bit);
        let (bld, diff_plus_p) = bld.with_wire(WireTy::Bits(32));
        let (bld, result) = bld.with_wire(WireTy::Bits(32));

        let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![a, b], diff)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![a, b], underflow)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![diff, c.p_32], diff_plus_p)?;
        let bld = bld.with_instruction(Op::Mux, vec![underflow, diff, diff_plus_p], result)?;

        Ok((bld, result))
    }

    fn inline_mul_reduce(
        bld: HdlGraphBuilder,
        a: WireId,
        b: WireId,
        c: &Self::Constants,
    ) -> Result<(HdlGraphBuilder, WireId), Error> {
        // Zero-extend operands to 64-bit and multiply.  Since both
        // inputs are canonical (< p < 2^31), the product fits in
        // 62 bits, so 64-bit wrapping multiplication is exact.
        let (bld, a_64) = bld.with_wire(WireTy::Bits(64));
        let (bld, b_64) = bld.with_wire(WireTy::Bits(64));
        let bld = bld.with_instruction(
            Op::Concat {
                low_width: 32,
                high_width: 32,
            },
            vec![a, c.zeros_32],
            a_64,
        )?;
        let bld = bld.with_instruction(
            Op::Concat {
                low_width: 32,
                high_width: 32,
            },
            vec![b, c.zeros_32],
            b_64,
        )?;
        let (bld, prod_64) = bld.with_wire(WireTy::Bits(64));
        let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![a_64, b_64], prod_64)?;

        // Cascaded Solinas reduction: twelve passes bring any
        // 62-bit product below 2p.
        let (bld, reduced_64) = (0..REDUCTION_PASSES)
            .try_fold((bld, prod_64), |(bld, cur), _| {
                Self::solinas_pass(bld, cur, c)
            })?;

        // Truncate to 32 bits and apply a single conditional subtraction of p.
        let (bld, reduced_32) = bld.with_wire(WireTy::Bits(32));
        let bld =
            bld.with_instruction(Op::Slice { lo: 0, hi: 32 }, vec![reduced_64], reduced_32)?;

        let (bld, r_lt_p) = bld.with_wire(WireTy::Bit);
        let (bld, r_ge_p) = bld.with_wire(WireTy::Bit);
        let (bld, r_minus_p) = bld.with_wire(WireTy::Bits(32));
        let (bld, result) = bld.with_wire(WireTy::Bits(32));

        let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![reduced_32, c.p_32], r_lt_p)?;
        let bld = bld.with_instruction(Op::Not, vec![r_lt_p], r_ge_p)?;
        let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![reduced_32, c.p_32], r_minus_p)?;
        let bld = bld.with_instruction(Op::Mux, vec![r_ge_p, reduced_32, r_minus_p], result)?;

        Ok((bld, result))
    }

    fn prime_u128() -> u128 {
        BABYBEAR_PRIME_U128
    }

    fn to_bitseq(val: u64) -> BitSeq {
        u32_bitseq(val)
    }

    fn from_bitseq(seq: &BitSeq) -> Result<u64, Error> {
        (seq.len() <= 32)
            .then_some(())
            .ok_or_else(|| hdl_cat_error::Error::WidthMismatch {
                expected: hdl_cat_error::Width::new(32),
                actual: hdl_cat_error::Width::new(u32::try_from(seq.len()).unwrap_or(u32::MAX)),
            })?;
        Ok((0..seq.len().min(32)).fold(
            0u64,
            |acc, i| {
                if seq.bit(i) { acc | (1u64 << i) } else { acc }
            },
        ))
    }

    fn one_wire(c: &Self::Constants) -> WireId {
        c.one_32
    }

    fn zero_wire(c: &Self::Constants) -> WireId {
        c.zeros_32
    }
}

#[cfg(test)]
mod tests {
    use super::{BABYBEAR_PRIME_U64, BabyBear};
    use crate::hdl::field_hdl::PrimeFieldHdl;
    use hdl_cat_error::Error;
    use hdl_cat_ir::{HdlGraphBuilder, WireTy};
    use hdl_cat_sim::interp::interpret;

    fn run_binop<F>(a: u64, b: u64, build: F) -> Result<u64, Error>
    where
        F: FnOnce(
            HdlGraphBuilder,
            hdl_cat_ir::WireId,
            hdl_cat_ir::WireId,
            &super::BabyBearConstants,
        ) -> Result<(HdlGraphBuilder, hdl_cat_ir::WireId), Error>,
    {
        let (bld, a_wire) = HdlGraphBuilder::new().with_wire(WireTy::Bits(32));
        let (bld, b_wire) = bld.with_wire(WireTy::Bits(32));
        let (bld, constants) = BabyBear::alloc_constants(bld)?;
        let (bld, result_wire) = build(bld, a_wire, b_wire, &constants)?;
        let graph = bld.build();

        let env = interpret(
            &graph,
            &[a_wire, b_wire],
            &[BabyBear::to_bitseq(a), BabyBear::to_bitseq(b)],
        )?;
        let result_bits = env
            .get(result_wire.index())
            .and_then(Clone::clone)
            .ok_or_else(|| Error::UndefinedSignal {
                name: hdl_cat_error::SignalName::new("result"),
            })?;
        BabyBear::from_bitseq(&result_bits)
    }

    #[test]
    fn prime_matches() {
        assert_eq!(BabyBear::prime_u128(), u128::from(BABYBEAR_PRIME_U64));
        assert_eq!(BABYBEAR_PRIME_U64, (1_u64 << 31) - (1_u64 << 27) + 1);
    }

    #[test]
    fn element_width_is_32() {
        assert_eq!(BabyBear::element_width(), 32);
    }

    #[test]
    fn add_canonicalizes() -> Result<(), Error> {
        let p = BABYBEAR_PRIME_U64;
        // No wrap needed: small + small.
        assert_eq!(run_binop(1, 2, BabyBear::inline_add)?, 3);
        // Wraps once: (p - 1) + 1 = p → 0.
        assert_eq!(run_binop(p - 1, 1, BabyBear::inline_add)?, 0);
        // Large pair that exceeds p: (p - 1) + (p - 1) = 2p - 2 → p - 2.
        assert_eq!(run_binop(p - 1, p - 1, BabyBear::inline_add)?, p - 2);
        Ok(())
    }

    #[test]
    fn sub_canonicalizes() -> Result<(), Error> {
        let p = BABYBEAR_PRIME_U64;
        // No underflow: 5 - 2 = 3.
        assert_eq!(run_binop(5, 2, BabyBear::inline_sub)?, 3);
        // Underflow: 0 - 1 = p - 1.
        assert_eq!(run_binop(0, 1, BabyBear::inline_sub)?, p - 1);
        // Boundary: 1 - (p - 1) = 2 - p + p = 2.  Hmm actually 1 - (p - 1) mod p = (1 - p + 1) mod p = (2 - p) mod p = 2.
        assert_eq!(run_binop(1, p - 1, BabyBear::inline_sub)?, 2);
        Ok(())
    }

    #[test]
    fn mul_reduce_matches_reference() -> Result<(), Error> {
        let p = BABYBEAR_PRIME_U64;
        let cases = [
            (0_u64, 0_u64),
            (1, 1),
            (2, 3),
            (p - 1, p - 1),
            (p - 1, 2),
            (12345, 67890),
            (p / 2, p / 2),
            (p - 1, 1),
            (0x4000_0000, 0x4000_0000),
            (0x1234_5678, 0x0abc_def0 & 0x7fff_ffff),
        ];
        cases.iter().try_for_each(|&(a, b)| -> Result<(), Error> {
            let expected =
                u64::try_from((u128::from(a) * u128::from(b)) % u128::from(p)).unwrap_or(0);
            let actual = run_binop(a, b, BabyBear::inline_mul_reduce)?;
            assert_eq!(actual, expected, "mul_reduce({a}, {b})");
            Ok(())
        })
    }
}
