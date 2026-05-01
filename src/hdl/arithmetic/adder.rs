//! Goldilocks modular adder HDL module.
//!
//! Computes `(a + b) mod p` where `p = 2^64 − 2^32 + 1`.  Both inputs are
//! assumed canonical (in `[0, p)`); the sum then fits in `[0, 2p) ⊂ [0, 2^65)`,
//! so a single correction by `p` (or by `p − 2^64 = 2^32 − 1` when the b64 sum
//! wraps) suffices.

use hdl_cat_bits::Bits;
use hdl_cat_circuit::{CircuitArrow, CircuitTensor, Obj};
use hdl_cat_error::Error;
use hdl_cat_ir::{BinOp, HdlGraphBuilder, Op, WireTy};
use hdl_cat_sync::Sync;

use crate::hdl::common::GOLDILOCKS_PRIME_U64;

/// Type alias for 64-bit Goldilocks field element.
pub type GoldilocksElement = Bits<64>;

/// Input bundle: two operands (as a tensor product).
pub type AdderInput = CircuitTensor<Obj<GoldilocksElement>, Obj<GoldilocksElement>>;

/// Output bundle: the modular sum.
pub type AdderOutput = Obj<GoldilocksElement>;

/// A combinational Goldilocks modular adder circuit arrow.
pub type GoldilocksAddArrow = CircuitArrow<AdderInput, AdderOutput>;

/// A stateful (1-cycle latency) Goldilocks modular adder.
pub type GoldilocksAddSync = Sync<Obj<GoldilocksElement>, AdderInput, AdderOutput>;

/// Construct a combinational Goldilocks modular adder.
///
/// Takes two 64-bit operands as inputs and produces the modular sum.
/// The implementation handles overflow correction for the Goldilocks prime
/// `p = 2^64 - 2^32 + 1`.
///
/// # Errors
///
/// Returns [`Error`] if IR construction fails.
pub fn goldilocks_add_comb() -> Result<GoldilocksAddArrow, Error> {
    // Declare all wires first
    let (bld, a) = HdlGraphBuilder::new().with_wire(WireTy::Bits(64));
    let (bld, b) = bld.with_wire(WireTy::Bits(64));
    let (bld, sum64) = bld.with_wire(WireTy::Bits(64));
    let (bld, overflow) = bld.with_wire(WireTy::Bit);
    let (bld, prime_wire) = bld.with_wire(WireTy::Bits(64));
    let (bld, wrap_corr) = bld.with_wire(WireTy::Bits(64));
    let (bld, adjusted) = bld.with_wire(WireTy::Bits(64));
    let (bld, ge_prime) = bld.with_wire(WireTy::Bit);
    let (bld, result) = bld.with_wire(WireTy::Bits(64));
    let (bld, sum_plus_wrap) = bld.with_wire(WireTy::Bits(64));
    let (bld, sum_ge_prime) = bld.with_wire(WireTy::Bit);
    let (bld, sum_minus_prime) = bld.with_wire(WireTy::Bits(64));
    let (bld, temp_adj) = bld.with_wire(WireTy::Bits(64));
    let (bld, sum_lt_prime) = bld.with_wire(WireTy::Bit);
    let (bld, adj_lt_prime) = bld.with_wire(WireTy::Bit);
    let (bld, adj_minus_prime) = bld.with_wire(WireTy::Bits(64));

    // Load constants
    let prime_bits = crate::hdl::common::u64_to_bitseq(GOLDILOCKS_PRIME_U64);
    let wrap_corr_bits = crate::hdl::common::u64_to_bitseq(0xFFFF_FFFF);

    // Chain all instructions
    let bld = bld.with_instruction(
        Op::Const {
            bits: prime_bits,
            ty: WireTy::Bits(64),
        },
        vec![],
        prime_wire,
    )?;

    let bld = bld.with_instruction(
        Op::Const {
            bits: wrap_corr_bits,
            ty: WireTy::Bits(64),
        },
        vec![],
        wrap_corr,
    )?;

    // Compute sum64 = a + b (wrapping)
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![a, b], sum64)?;

    // Check overflow: sum64 < a
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![sum64, a], overflow)?;

    // First adjustment computations
    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![sum64, wrap_corr], sum_plus_wrap)?;
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![sum64, prime_wire], sum_lt_prime)?;
    let bld = bld.with_instruction(Op::Not, vec![sum_lt_prime], sum_ge_prime)?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Sub),
        vec![sum64, prime_wire],
        sum_minus_prime,
    )?;
    let bld = bld.with_instruction(
        Op::Mux,
        vec![sum_ge_prime, sum64, sum_minus_prime],
        temp_adj,
    )?;
    let bld = bld.with_instruction(Op::Mux, vec![overflow, temp_adj, sum_plus_wrap], adjusted)?;

    // Second check: adjusted >= prime as !(adjusted < prime)
    let bld = bld.with_instruction(Op::Bin(BinOp::Lt), vec![adjusted, prime_wire], adj_lt_prime)?;
    let bld = bld.with_instruction(Op::Not, vec![adj_lt_prime], ge_prime)?;

    // Final result: ge_prime ? adjusted - prime : adjusted
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Sub),
        vec![adjusted, prime_wire],
        adj_minus_prime,
    )?;
    let bld = bld.with_instruction(Op::Mux, vec![ge_prime, adjusted, adj_minus_prime], result)?;

    Ok(CircuitArrow::from_raw_parts(
        bld.build(),
        vec![a, b],
        vec![result],
    ))
}

/// Construct a stateful (1-cycle latency) Goldilocks modular adder.
///
/// This wraps the combinational adder in a `Sync` machine that provides
/// 1-cycle latency by storing the result in state.
///
/// # Errors
///
/// Returns [`Error`] if construction fails.
pub fn goldilocks_add_sync() -> Result<GoldilocksAddSync, Error> {
    let comb = goldilocks_add_comb()?;
    let lifted = Sync::lift_comb(comb);
    let (graph, inputs, outputs, init, sc) = lifted.into_parts();
    Ok(hdl_cat_sync::machine::from_raw(
        graph, inputs, outputs, init, sc,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdl_cat_sim::Testbench;

    /// Test adder correctness via sync wrapper.
    #[test]
    fn adder_basic() -> Result<(), Error> {
        let adder = goldilocks_add_sync()?;

        let test_cases: Vec<(u64, u64, u64)> =
            vec![(0, 0, 0), (1, 2, 3), (10, 20, 30), (100, 200, 300)];

        let inputs: Vec<_> = test_cases
            .iter()
            .map(|&(a, b, _)| {
                crate::hdl::common::u64_to_bitseq(a).concat(crate::hdl::common::u64_to_bitseq(b))
            })
            .collect();

        let testbench = Testbench::new(adder);
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
    fn adder_modular_reduction() -> Result<(), Error> {
        let adder = goldilocks_add_sync()?;
        let p = GOLDILOCKS_PRIME_U64;

        let test_cases: Vec<(u64, u64, u64)> = vec![
            (p - 1, 1, 0),         // sum = p, should reduce to 0
            (p - 1, 2, 1),         // sum = p + 1, should reduce to 1
            (p - 1, p - 1, p - 2), // sum = 2p - 2, should reduce to p - 2
        ];

        let inputs: Vec<_> = test_cases
            .iter()
            .map(|&(a, b, _)| {
                crate::hdl::common::u64_to_bitseq(a).concat(crate::hdl::common::u64_to_bitseq(b))
            })
            .collect();

        let testbench = Testbench::new(adder);
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
}
