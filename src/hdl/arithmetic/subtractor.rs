//! Goldilocks modular subtractor HDL module.
//!
//! Computes `(a − b) mod p` where `p = 2^64 − 2^32 + 1`.  Both inputs are
//! assumed canonical (in `[0, p)`); when `b > a`, the b64 wrapping subtraction
//! underflows by `2^64`, and the identity `2^64 ≡ 2^32 − 1 (mod p)` lets us
//! correct with a single further subtract of `(2^32 − 1)`.

use hdl_cat_bits::Bits;
use hdl_cat_circuit::{CircuitArrow, CircuitTensor, Obj};
use hdl_cat_error::Error;
use hdl_cat_ir::{BinOp, HdlGraphBuilder, Op, WireTy};
use hdl_cat_sync::Sync;

/// Type alias for 64-bit Goldilocks field element.
pub type GoldilocksElement = Bits<64>;

/// Input bundle: two operands (as a tensor product).
pub type SubInput = CircuitTensor<Obj<GoldilocksElement>, Obj<GoldilocksElement>>;

/// Output bundle: the modular difference.
pub type SubOutput = Obj<GoldilocksElement>;

/// A combinational Goldilocks modular subtractor circuit arrow.
pub type GoldilocksSubArrow = CircuitArrow<SubInput, SubOutput>;

/// A stateful (1-cycle latency) Goldilocks modular subtractor.
pub type GoldilocksSubSync = Sync<Obj<GoldilocksElement>, SubInput, SubOutput>;

/// Construct a combinational Goldilocks modular subtractor.
///
/// Takes two 64-bit operands as inputs (minuend, subtrahend) and produces the
/// modular difference. The implementation handles underflow correction using
/// the identity `2^64 ≡ 2^32 − 1 (mod p)`.
///
/// # Errors
///
/// Returns [`Error`] if IR construction fails.
pub fn goldilocks_sub_comb() -> Result<GoldilocksSubArrow, Error> {
    // Declare all wires first to avoid mutability
    let (bld, a) = HdlGraphBuilder::new().with_wire(WireTy::Bits(64)); // minuend
    let (bld, b) = bld.with_wire(WireTy::Bits(64)); // subtrahend
    let (bld, diff64) = bld.with_wire(WireTy::Bits(64)); // a - b (wrapping)
    let (bld, underflow) = bld.with_wire(WireTy::Bit); // b > a
    let (bld, wrap_corr) = bld.with_wire(WireTy::Bits(64)); // 0xFFFF_FFFF
    let (bld, result) = bld.with_wire(WireTy::Bits(64)); // Final result

    // Add the additional wire declaration
    let (bld, diff_minus_corr) = bld.with_wire(WireTy::Bits(64));

    // Chain all instructions functionally
    let bld = bld.with_instruction(
        Op::Const {
            bits: crate::hdl::common::u64_to_bitseq(0xFFFF_FFFF),
            ty: WireTy::Bits(64),
        },
        vec![],
        wrap_corr,
    )?;

    let bld = bld.with_instruction(Op::Bin(BinOp::Sub), vec![a, b], diff64)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![a, b], underflow)?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Sub),
        vec![diff64, wrap_corr],
        diff_minus_corr,
    )?;
    let bld = bld.with_instruction(Op::Mux, vec![underflow, diff64, diff_minus_corr], result)?;

    Ok(CircuitArrow::from_raw_parts(
        bld.build(),
        vec![a, b],
        vec![result],
    ))
}

/// Construct a stateful (1-cycle latency) Goldilocks modular subtractor.
///
/// This wraps the combinational subtractor in a `Sync` machine that provides
/// 1-cycle latency by storing the result in state.
///
/// # Errors
///
/// Returns [`Error`] if construction fails.
pub fn goldilocks_sub_sync() -> Result<GoldilocksSubSync, Error> {
    let comb = goldilocks_sub_comb()?;
    let lifted = Sync::lift_comb(comb);
    let (graph, inputs, outputs, init, sc) = lifted.into_parts();
    Ok(hdl_cat_sync::machine::from_raw(
        graph, inputs, outputs, init, sc,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdl::common::GOLDILOCKS_PRIME_U64;
    use hdl_cat_sim::Testbench;

    /// Reference implementation for testing.
    fn reference_sub(a: u64, b: u64) -> u64 {
        let p = u128::from(GOLDILOCKS_PRIME_U64);
        u64::try_from((u128::from(a) + p - u128::from(b)) % p).unwrap_or(0)
    }

    /// Test subtractor correctness via sync wrapper.
    #[test]
    fn sub_basic() -> Result<(), Error> {
        let sub = goldilocks_sub_sync()?;

        let test_cases: Vec<(u64, u64, u64)> =
            vec![(0, 0, 0), (5, 3, 2), (100, 40, 60), (10, 0, 10)];

        let inputs: Vec<_> = test_cases
            .iter()
            .map(|&(a, b, _)| {
                crate::hdl::common::u64_to_bitseq(a).concat(crate::hdl::common::u64_to_bitseq(b))
            })
            .collect();

        let testbench = Testbench::new(sub);
        let results = testbench.run(inputs).run()?;

        test_cases
            .iter()
            .zip(results.iter())
            .try_for_each(|(&(_, _, expected), sample)| {
                let output_val = crate::hdl::common::bitseq_to_u64(sample.value())?;
                assert_eq!(output_val, expected);
                Ok::<(), hdl_cat_error::Error>(())
            })?;

        Ok(())
    }

    #[test]
    fn sub_underflow_correction() -> Result<(), Error> {
        let sub = goldilocks_sub_sync()?;
        let p = GOLDILOCKS_PRIME_U64;

        let test_cases: Vec<(u64, u64)> = vec![
            (0, 1),             // underflow case
            (1, 2),             // underflow case
            (0, p - 1),         // underflow case
            (p / 2, p / 2 + 1), // underflow case
        ];

        let inputs: Vec<_> = test_cases
            .iter()
            .map(|&(a, b)| {
                crate::hdl::common::u64_to_bitseq(a).concat(crate::hdl::common::u64_to_bitseq(b))
            })
            .collect();

        let testbench = Testbench::new(sub);
        let results = testbench.run(inputs).run()?;

        test_cases
            .iter()
            .zip(results.iter())
            .try_for_each(|(&(a, b), sample)| {
                let output_val = crate::hdl::common::bitseq_to_u64(sample.value())?;
                let expected = reference_sub(a, b);
                assert_eq!(output_val, expected);
                Ok::<(), hdl_cat_error::Error>(())
            })?;

        Ok(())
    }

    #[test]
    fn sync_sub_builds() -> Result<(), Error> {
        let _sub = goldilocks_sub_sync()?;
        Ok(())
    }
}
