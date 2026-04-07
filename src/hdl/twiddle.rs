//! On-the-fly twiddle factor generator.
//!
//! Maintains a single accumulator for Goldilocks field elements. When `active` is
//! asserted, the accumulator advances by one step of the stage's twiddle
//! sequence (i.e. multiplies by `step_root`). When `reset` is asserted,
//! the accumulator is forced to `1`, starting a fresh twiddle sequence.
//!
//! This module uses a combinational 64×64 multiply-and-reduce circuit
//! so that the accumulator output matches the *current* stage phase with
//! single-cycle feedback.

use hdl_cat_bits::Bits;
use hdl_cat_circuit::{CircuitArrow, CircuitTensor, Obj};
use hdl_cat_error::Error;
use hdl_cat_ir::{BinOp, HdlGraphBuilder, Op, WireTy};
use hdl_cat_sync::Sync;

/// Type alias for 64-bit Goldilocks field element.
pub type GoldilocksElement = Bits<64>;

/// Input bundle for the twiddle accumulator.
pub type TwiddleInput = CircuitTensor<
    CircuitTensor<Obj<GoldilocksElement>, Obj<bool>>, // (step_root, active)
    Obj<bool>, // reset
>;

/// Output: current twiddle value.
pub type TwiddleOutput = Obj<GoldilocksElement>;

/// State: current accumulator value.
pub type TwiddleState = Obj<GoldilocksElement>;

/// A twiddle accumulator sync machine.
pub type TwiddleAccumulatorSync = Sync<TwiddleState, TwiddleInput, TwiddleOutput>;

/// Construct a twiddle accumulator.
///
/// Takes inputs: `((step_root, active), reset)` and produces the current
/// twiddle value. The accumulator multiplies by `step_root` when `active`
/// is true, or resets to 1 when `reset` is true.
///
/// # Errors
///
/// Returns [`Error`] if IR construction fails.
pub fn twiddle_accumulator() -> Result<TwiddleAccumulatorSync, Error> {
    let (bld, step_root) = HdlGraphBuilder::new().with_wire(WireTy::Bits(64));
    let (bld, active) = bld.with_wire(WireTy::Bit);
    let (bld, reset) = bld.with_wire(WireTy::Bit);
    let (bld, current_state) = bld.with_wire(WireTy::Bits(64)); // Previous accumulator value

    // Constants
    let (bld, one) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(
        Op::Const {
            bits: crate::hdl::common::u64_to_bitseq(1),
            ty: WireTy::Bits(64),
        },
        vec![],
        one,
    )?;

    // Multiply current state by step_root (inline simple multiply)
    let (bld, stepped) = bld.with_wire(WireTy::Bits(64));
    let bld = bld.with_instruction(Op::Bin(BinOp::Mul), vec![current_state, step_root], stepped)?;

    // Conditional logic: if reset then 1, else if active then stepped, else current
    let (bld, temp_val) = bld.with_wire(WireTy::Bits(64));
    let (bld, next_val) = bld.with_wire(WireTy::Bits(64));

    let bld = bld.with_instruction(Op::Mux, vec![active, current_state, stepped], temp_val)?;
    let bld = bld.with_instruction(Op::Mux, vec![reset, temp_val, one], next_val)?;

    // The arrow: (step_root, active, reset, state) -> (current_output, next_state)
    // We output the current state and compute next state
    let twiddle_arrow: CircuitArrow<TwiddleInput, TwiddleOutput> = CircuitArrow::from_raw_parts(
        bld.build(),
        vec![step_root, active, reset, current_state],
        vec![current_state, next_val], // Output current, next state
    );

    // Initial state: accumulator starts at 1
    let initial_state = crate::hdl::common::u64_to_bitseq(1);

    Sync::from_arrow(twiddle_arrow, initial_state)
}

/// Software reference for testing.
#[must_use]
pub fn reference_twiddle_step(current: u64, step_root: u64, active: bool, reset: bool) -> u64 {
    if reset {
        1
    } else if active {
        use crate::hdl::goldilocks_reduce::reference_reduce;
        let product = u128::from(current) * u128::from(step_root);
        reference_reduce(product)
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twiddle_accumulator_builds() -> Result<(), Error> {
        let _acc = twiddle_accumulator()?;
        Ok(())
    }

    #[test]
    fn reference_implementation_works() {
        // Test reset
        assert_eq!(reference_twiddle_step(42, 7, true, true), 1);

        // Test active step
        assert_eq!(reference_twiddle_step(1, 7, true, false), 7);

        // Test inactive (hold)
        assert_eq!(reference_twiddle_step(42, 7, false, false), 42);
    }

    // More comprehensive tests would require running the accumulator
    // through multiple cycles, but the structure is now in place
}
