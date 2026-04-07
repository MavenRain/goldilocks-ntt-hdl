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
use hdl_cat_ir::{HdlGraphBuilder, Op, WireId, WireTy};
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

/// Create an N-cycle delay by building an N-element shift register.
///
/// The shift register is implemented as pure wire routing: no IR
/// instructions are needed because the Sync machine's state-thread
/// mechanism performs the shifting.  On each cycle the data input
/// enters position 0, every element moves up one slot, and the
/// element at position N-1 is emitted as the output.
///
/// # Errors
///
/// Returns [`Error`] if IR construction fails.
pub fn delay_n<T>(n: usize) -> Result<DelayLine<T>, Error>
where
    T: Clone + Default + hdl_cat_kind::Hw + hdl_cat_circuit::object::Scalar,
{
    if n == 0 {
        // 0-cycle delay is just a pass-through (identity)
        let (bld, input) = HdlGraphBuilder::new().with_wire(WireTy::Bits(64));
        let initial_bits: Vec<bool> = (0..64).map(|_| false).collect();
        let initial_state = BitSeq::from_vec(initial_bits);

        let arrow: CircuitArrow<Obj<T>, Obj<T>> = CircuitArrow::from_raw_parts(
            bld.build(),
            vec![input],
            vec![input], // pass-through: output = input
        );

        Sync::from_arrow(arrow, initial_state)
    } else {
            // N-cycle shift register with identity instructions.
            //
            // Wire layout (all 64-bit):
            //   wires 0..n-1     = state registers (s_0 .. s_{n-1})
            //   wire  n          = data input (d_in)
            //   wires n+1..2n    = next-state copies (computed by instructions)
            //   wire  2n+1       = output copy
            //
            // Identity instructions (Slice 0..64) implement the shift:
            //   wire n+1 = copy(d_in)      → next_state[0]
            //   wire n+2 = copy(s_0)       → next_state[1]
            //   ...
            //   wire 2n  = copy(s_{n-2})   → next_state[n-1]
            //   wire 2n+1 = copy(s_{n-1})  → data_out
            let bit_width = 64u32;

            // Allocate n+1 source wires + n+1 destination wires
            let bld = (0..(2 * n + 2)).fold(HdlGraphBuilder::new(), |b, _| {
                b.with_wire(WireTy::Bits(bit_width)).0
            });

            // Identity copy: next_state[0] = data_in
            let bld = bld.with_instruction(
                Op::Slice { lo: 0, hi: 64 },
                vec![WireId::new(n)],
                WireId::new(n + 1),
            )?;

            // Identity copies: next_state[i] = s_{i-1}
            let bld = (1..n).try_fold(bld, |b, i| {
                b.with_instruction(
                    Op::Slice { lo: 0, hi: 64 },
                    vec![WireId::new(i - 1)],
                    WireId::new(n + 1 + i),
                )
            })?;

            // Identity copy: output = s_{n-1}
            let bld = bld.with_instruction(
                Op::Slice { lo: 0, hi: 64 },
                vec![WireId::new(n - 1)],
                WireId::new(2 * n + 1),
            )?;

            let graph = bld.build();

            let input_wires: Vec<WireId> = (0..=n).map(WireId::new).collect();

            // Output wires point to instruction-computed destinations
            let output_wires: Vec<WireId> = (0..=n).map(|i| WireId::new(n + 1 + i)).collect();

            let initial_state = BitSeq::from_vec(
                (0..64 * n).map(|_| false).collect(),
            );

            Ok(hdl_cat_sync::machine::from_raw(
                graph,
                input_wires,
                output_wires,
                initial_state,
                n,
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
