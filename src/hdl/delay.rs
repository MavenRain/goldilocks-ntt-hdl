//! Delay-line primitives for SDF stages.
//!
//! Each SDF stage needs an `N`-cycle delay whose input is phase-multiplexed:
//! during the fill phase the delay's input is the streaming data, and during
//! the butterfly phase its input is the butterfly's upper output.  The delay
//! always shifts every cycle; the phase-dependent mux lives in
//! [`crate::hdl::stage`].
//!
//! In hdl-cat, delays are implemented by chaining `Sync` machines with
//! identity circuits and appropriate state initialization.

use hdl_cat_circuit::{CircuitArrow, Obj};
use hdl_cat_error::Error;
use hdl_cat_ir::{HdlGraphBuilder, Op, WireTy};
use hdl_cat_kind::BitSeq;
use hdl_cat_sync::Sync;

/// A generic delay line for any hardware-typable type.
pub type DelayLine<T> = Sync<Obj<T>, Obj<T>, Obj<T>>;

/// Create a 1-cycle delay for a given type.
///
/// # Errors
///
/// Returns [`Error`] if IR construction fails.
pub fn delay_1<T>() -> Result<DelayLine<T>, Error>
where
    T: Clone + Default + hdl_cat_kind::Hw + hdl_cat_circuit::object::Scalar,
{
    // For simplicity, assume 64-bit elements (Goldilocks)
    let bit_width = 64u32;

    // Create a register with appropriate width
    let (bld, input) = HdlGraphBuilder::new().with_wire(WireTy::Bits(bit_width));
    let (bld, output) = bld.with_wire(WireTy::Bits(bit_width));

    // Create initial state as all zeros
    let initial_bits: Vec<bool> = (0..64).map(|_| false).collect();
    let initial_state = BitSeq::from_vec(initial_bits.clone());

    let bld = bld.with_instruction(
        Op::Reg {
            init: initial_state.clone(),
            ty: WireTy::Bits(bit_width),
        },
        vec![input],
        output,
    )?;

    let arrow: CircuitArrow<Obj<T>, Obj<T>> = CircuitArrow::from_raw_parts(
        bld.build(),
        vec![input],
        vec![output],
    );

    Sync::from_arrow(arrow, initial_state)
}

/// Create an N-cycle delay using a single Array-typed state wire.
///
/// The delay line is represented as one `WireTy::Array` wire with
/// `depth = n` and `element_width = 64`.  On each cycle,
/// [`Op::ArrayShiftIn`] pushes the data input into position 0 and
/// [`Op::ArrayTail`] reads the oldest element from position `n-1`.
///
/// # Errors
///
/// Returns [`Error`] if IR construction fails.
pub fn delay_n<T>(n: usize) -> Result<DelayLine<T>, Error>
where
    T: Clone + Default + hdl_cat_kind::Hw + hdl_cat_circuit::object::Scalar,
{
    if n == 0 {
        // 0-cycle delay is just a pass-through (identity).
        let (bld, input) = HdlGraphBuilder::new().with_wire(WireTy::Bits(64));
        let initial_bits: Vec<bool> = (0..64).map(|_| false).collect();
        let initial_state = BitSeq::from_vec(initial_bits);

        let arrow: CircuitArrow<Obj<T>, Obj<T>> = CircuitArrow::from_raw_parts(
            bld.build(),
            vec![input],
            vec![input],
        );

        Sync::from_arrow(arrow, initial_state)
    } else {
        // Wire layout:
        //   w0 = state array (Array { element_width: 64, depth: n })
        //   w1 = data input (Bits(64))
        //   w2 = next-state array (ArrayShiftIn output)
        //   w3 = data output (ArrayTail output, Bits(64))
        let arr_ty = WireTy::Array { element_width: 64, depth: n };
        let (bld, arr_state) = HdlGraphBuilder::new().with_wire(arr_ty.clone());
        let (bld, data_in) = bld.with_wire(WireTy::Bits(64));
        let (bld, next_arr) = bld.with_wire(arr_ty);
        let (bld, data_out) = bld.with_wire(WireTy::Bits(64));

        // next_arr = shift_in(arr_state, data_in)
        let bld = bld.with_instruction(
            Op::ArrayShiftIn { element_width: 64, depth: n },
            vec![arr_state, data_in],
            next_arr,
        )?;
        // data_out = arr_state[n - 1]
        let bld = bld.with_instruction(
            Op::ArrayTail { element_width: 64, depth: n },
            vec![arr_state],
            data_out,
        )?;

        let graph = bld.build();

        // State: [arr_state], Data: [data_in]
        // Next-state: [next_arr], Data out: [data_out]
        let input_wires = vec![arr_state, data_in];
        let output_wires = vec![next_arr, data_out];

        // Compact initial state: one element's worth of zeros (64 bits).
        let initial_state = BitSeq::from_vec(
            (0..64).map(|_| false).collect(),
        );

        Ok(hdl_cat_sync::machine::from_raw(
            graph,
            input_wires,
            output_wires,
            initial_state,
            1, // 1 state wire (the array)
        ))
    }
}

/// Type alias for a 7-cycle delay of 64-bit Goldilocks elements.
pub type GoldilocksDelay7 = DelayLine<hdl_cat_bits::Bits<64>>;

/// Create a 7-cycle delay for Goldilocks elements.
///
/// # Errors
///
/// Returns [`Error`] if IR construction fails.
pub fn goldilocks_delay_7() -> Result<GoldilocksDelay7, Error> {
    delay_n::<hdl_cat_bits::Bits<64>>(7)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdl::common::{bitseq_to_u64, u64_to_bitseq};
    use hdl_cat_sim::Testbench;

    #[test]
    fn delay_7_outputs_after_7_cycles() -> Result<(), Error> {
        let delay = delay_n::<hdl_cat_bits::Bits<64>>(7)?;

        // Feed value 42 on cycle 0, then zeros for 7 more cycles.
        // Expect 42 to appear at cycle 7.
        let inputs: Vec<_> = core::iter::once(u64_to_bitseq(42))
            .chain((0..9).map(|_| u64_to_bitseq(0)))
            .collect();

        let testbench = Testbench::new(delay);
        let results = testbench.run(inputs).run()?;

        // First 7 outputs should be 0 (initial state)
        (0..7).try_for_each(|i| {
            let val = bitseq_to_u64(results[i].value())?;
            assert_eq!(val, 0, "cycle {i}: expected 0, got {val}");
            Ok::<(), hdl_cat_error::Error>(())
        })?;

        // Cycle 7 should output 42
        let val_7 = bitseq_to_u64(results[7].value())?;
        assert_eq!(val_7, 42, "cycle 7: expected 42, got {val_7}");

        Ok(())
    }
}
