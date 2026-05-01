//! Single SDF (Single-path Delay Feedback) stage.
//!
//! Each stage streams one element per cycle and alternates between two
//! phases of length `DEPTH`:
//!
//! - **Fill phase** (counter in `[0, D)`): the incoming element is
//!   written into the delay line and the delay line's oldest element
//!   (tail) is emitted downstream.  On the first frame the delay
//!   contains zeros; on subsequent frames it contains the upper-branch
//!   results from the previous frame's butterfly phase.
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
//! The modular arithmetic (add, sub, multiply-reduce) is supplied by a
//! [`PrimeFieldHdl`] implementation, so the SDF phase logic is
//! field-generic.  The Goldilocks instantiation is provided by
//! [`sdf_stage`], which delegates to [`sdf_stage_generic`] with the
//! [`Goldilocks`] marker type.

use comp_cat_rs::effect::io::Io;
use hdl_cat_bits::Bits;
use hdl_cat_circuit::{CircuitTensor, Obj};
use hdl_cat_error::Error;
use hdl_cat_ir::{BinOp, HdlGraphBuilder, Op, WireId, WireTy};
use hdl_cat_kind::BitSeq;
use hdl_cat_sync::Sync;

use crate::hdl::common::GOLDILOCKS_PRIME_U64;
use crate::hdl::field_hdl::PrimeFieldHdl;
use crate::hdl::goldilocks_field_hdl::Goldilocks;

/// Counter width used by the SDF phase logic.  Supports depths up to 2^24.
pub const COUNTER_BITS: usize = 24;

/// Type alias for 64-bit Goldilocks field element.
pub type GoldilocksElement = Bits<64>;

/// Input bundle for an SDF stage: `((data, valid), step_root)`.
pub type SdfStageInput =
    CircuitTensor<CircuitTensor<Obj<GoldilocksElement>, Obj<bool>>, Obj<GoldilocksElement>>;

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

// ── Counter helper ───────────────────────────────────────────────────

/// Create a [`COUNTER_BITS`]-wide [`BitSeq`] from a `u32`.
fn counter_to_bitseq(val: u32) -> BitSeq {
    BitSeq::from_vec((0..COUNTER_BITS).map(|i| (val >> i) & 1 == 1).collect())
}

// ── SDF stage constructor ────────────────────────────────────────────

/// Construct a field-generic SDF stage with the given delay depth.
///
/// The stage implements the SDF algorithm with fill and butterfly phases,
/// coordinated by an internal counter.  The delay line is a D-deep shift
/// register.  Modular arithmetic (add, sub, multiply-reduce) is supplied
/// by the [`PrimeFieldHdl`] implementation, making this constructor
/// field-generic.
///
/// # Type parameters
///
/// * `F` - The prime field HDL implementation.
///
/// # Arguments
///
/// * `depth` - The delay depth for this stage (must be >= 1).
///
/// # Errors
///
/// Returns [`Error`] if `depth` is zero or IR construction fails.
#[allow(clippy::too_many_lines)]
pub fn sdf_stage_generic<F: PrimeFieldHdl>(depth: usize) -> Result<SdfStageSync, Error> {
    if depth == 0 {
        return Err(Error::WidthMismatch {
            expected: hdl_cat_error::Width::new(1),
            actual: hdl_cat_error::Width::new(0),
        });
    }

    let counter_bits = u32::try_from(COUNTER_BITS).unwrap_or(16);
    let elem_width = F::element_width();
    let elem_ty = F::element_wire_ty();

    // ── Source wire allocation ────────────────────────────────────────
    // State wires: delay_arr (Array), twiddle, counter
    let delay_ty = WireTy::Array {
        element_width: elem_width,
        depth,
    };
    let (bld, delay_arr) = HdlGraphBuilder::new().with_wire(delay_ty.clone());

    let (bld, twiddle) = bld.with_wire(elem_ty.clone());
    let (bld, counter) = bld.with_wire(WireTy::Bits(counter_bits));

    // Data input wires
    let (bld, data_in) = bld.with_wire(elem_ty.clone());
    let (bld, valid_in) = bld.with_wire(WireTy::Bit);
    let (bld, step_root) = bld.with_wire(elem_ty.clone());

    // ── Shared constants ─────────────────────────────────────────────
    let (bld, arith) = F::alloc_constants(bld)?;

    // ── Counter constants ────────────────────────────────────────────
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
        Op::Const {
            bits: counter_to_bitseq(depth_u32),
            ty: WireTy::Bits(counter_bits),
        },
        vec![],
        depth_const,
    )?;
    let bld = bld.with_instruction(
        Op::Const {
            bits: counter_to_bitseq(max_counter_u32),
            ty: WireTy::Bits(counter_bits),
        },
        vec![],
        max_counter_const,
    )?;
    let bld = bld.with_instruction(
        Op::Const {
            bits: counter_to_bitseq(1),
            ty: WireTy::Bits(counter_bits),
        },
        vec![],
        one_ctr,
    )?;
    let bld = bld.with_instruction(
        Op::Const {
            bits: counter_to_bitseq(0),
            ty: WireTy::Bits(counter_bits),
        },
        vec![],
        zero_ctr,
    )?;

    // ── Phase logic ──────────────────────────────────────────────────
    // phase = 0 (fill) when counter < depth; phase = 1 (butterfly) otherwise
    let (bld, counter_lt_depth) = bld.with_wire(WireTy::Bit);
    let (bld, phase) = bld.with_wire(WireTy::Bit);

    let bld = bld.with_instruction(
        Op::Bin(BinOp::Lt),
        vec![counter, depth_const],
        counter_lt_depth,
    )?;
    let bld = bld.with_instruction(Op::Not, vec![counter_lt_depth], phase)?;

    // ── Read delayed value (tail of array) ────────────────────────────
    let (bld, delayed) = bld.with_wire(elem_ty.clone());
    let bld = bld.with_instruction(
        Op::ArrayTail {
            element_width: elem_width,
            depth,
        },
        vec![delay_arr],
        delayed,
    )?;

    // ── Butterfly arithmetic (always computed; muxed by phase) ────────
    let (bld, upper) = F::inline_add(bld, delayed, data_in, &arith)?;
    let (bld, diff) = F::inline_sub(bld, delayed, data_in, &arith)?;
    let (bld, lower) = F::inline_mul_reduce(bld, diff, twiddle, &arith)?;

    // ── Twiddle advancement ──────────────────────────────────────────
    let (bld, twiddle_stepped) = F::inline_mul_reduce(bld, twiddle, step_root, &arith)?;

    // ── Phase-dependent muxing ───────────────────────────────────────
    // data_out: fill -> delayed (R2SDF: drain previous upper from delay),
    //           butterfly -> lower
    let (bld, data_out) = bld.with_wire(elem_ty.clone());
    let bld = bld.with_instruction(Op::Mux, vec![phase, delayed, lower], data_out)?;

    // valid_out: identity copy
    let (bld, valid_out) = bld.with_wire(WireTy::Bit);
    let bld = bld.with_instruction(Op::Slice { lo: 0, hi: 1 }, vec![valid_in], valid_out)?;

    // delay_in: fill -> data_in, butterfly -> upper
    let (bld, delay_in) = bld.with_wire(elem_ty.clone());
    let bld = bld.with_instruction(Op::Mux, vec![phase, data_in, upper], delay_in)?;

    // next_twiddle: fill -> 1, butterfly -> twiddle * step_root
    let (bld, next_twiddle) = bld.with_wire(elem_ty);
    let bld = bld.with_instruction(
        Op::Mux,
        vec![phase, F::one_wire(&arith), twiddle_stepped],
        next_twiddle,
    )?;

    // ── Counter logic ────────────────────────────────────────────────
    let (bld, counter_inc) = bld.with_wire(WireTy::Bits(counter_bits));
    let (bld, counter_at_max) = bld.with_wire(WireTy::Bit);
    let (bld, next_counter) = bld.with_wire(WireTy::Bits(counter_bits));

    let bld = bld.with_instruction(Op::Bin(BinOp::Add), vec![counter, one_ctr], counter_inc)?;
    let bld = bld.with_instruction(
        Op::Bin(BinOp::Eq),
        vec![counter, max_counter_const],
        counter_at_max,
    )?;
    // at_max=0 -> counter_inc, at_max=1 -> zero (wrap)
    let bld = bld.with_instruction(
        Op::Mux,
        vec![counter_at_max, counter_inc, zero_ctr],
        next_counter,
    )?;

    // ── Delay line: shift in via ArrayShiftIn ──────────────────────────
    let (bld, next_delay_arr) = bld.with_wire(delay_ty);
    let bld = bld.with_instruction(
        Op::ArrayShiftIn {
            element_width: elem_width,
            depth,
        },
        vec![delay_arr, delay_in],
        next_delay_arr,
    )?;

    // ── Build the graph and assemble the Sync machine ────────────────
    let graph = bld.build();

    // Input wires: state ++ data
    // State: [delay_arr, twiddle, counter]
    // Data:  [data_in, valid_in, step_root]
    let input_wires: Vec<WireId> = [delay_arr, twiddle, counter, data_in, valid_in, step_root]
        .into_iter()
        .collect();

    // Output wires: next_state ++ data_output
    // Next state: [next_delay_arr, next_twiddle, next_counter]
    // Data out:   [data_out, valid_out]
    let output_wires: Vec<WireId> = [
        next_delay_arr,
        next_twiddle,
        next_counter,
        data_out,
        valid_out,
    ]
    .into_iter()
    .collect();

    let state_wire_count = 3; // array + twiddle + counter

    // Initial state (compact): delay element = 0, twiddle = 1, counter = 0
    let width = usize::try_from(elem_width).unwrap_or(64);
    let initial_state = BitSeq::from_vec(
        (0..width)
            .map(|_| false) // delay element reset (compact)
            .chain(core::iter::once(true)) // twiddle bit 0 = 1
            .chain((1..width).map(|_| false)) // twiddle remaining bits
            .chain((0..COUNTER_BITS).map(|_| false)) // counter: zero
            .collect(),
    );

    Ok(hdl_cat_sync::machine::from_raw(
        graph,
        input_wires,
        output_wires,
        initial_state,
        state_wire_count,
    ))
}

/// Construct a Goldilocks-flavoured SDF stage with the given delay depth.
///
/// This is a convenience wrapper around [`sdf_stage_generic`] that
/// instantiates the field-generic stage with the [`Goldilocks`] marker
/// type, giving callers the default 64-bit Goldilocks arithmetic.
///
/// # Arguments
///
/// * `depth` - The delay depth for this stage (must be >= 1).
///
/// # Errors
///
/// Returns [`Error`] if `depth` is zero or IR construction fails.
pub fn sdf_stage(depth: usize) -> Result<SdfStageSync, Error> {
    sdf_stage_generic::<Goldilocks>(depth)
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

/// Emit an SDF stage as a time-unrolled Circom template.
///
/// Constructs an [`sdf_stage`] of the given delay depth and lowers it
/// through [`hdl_cat_circom::emit_unrolled_template`], producing a flat
/// combinational template that exposes a per-cycle input bus, a per-cycle
/// output bus, and the initial state baked into cycle-0 state wires.
///
/// The Mealy step (delay-line shift, twiddle accumulation, counter
/// rollover) is replicated `num_cycles` times in the rendered template,
/// with cycle-`k+1` state wires driven from cycle-`k`'s next-state wires.
///
/// # Arguments
///
/// * `depth` - Delay depth for the stage (must be >= 1).
/// * `num_cycles` - Number of cycles to unroll (must be >= 1).
/// * `name` - Circom template name.
///
/// # Errors
///
/// Returns [`Error`] if stage construction fails.  The returned [`Io`]
/// surfaces emitter errors at run time (currently only `Reg` ops, which
/// [`sdf_stage`] does not emit).
pub fn emit_sdf_stage_circom(
    depth: usize,
    num_cycles: usize,
    name: &str,
) -> Result<Io<Error, String>, Error> {
    let stage = sdf_stage(depth)?;
    let (graph, input_wires, output_wires, initial_state, state_wire_count) = stage.into_parts();

    let template = hdl_cat_circom::emit_unrolled_template(
        &graph,
        name,
        &input_wires,
        &output_wires,
        state_wire_count,
        &initial_state,
        num_cycles,
    );

    Ok(template.flat_map(|t| t.render()))
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
            Self {
                delay: vec![0; depth],
                twiddle: 1,
                counter: 0,
                depth,
            }
        }

        fn step(self, data_in: u64, step_root: u64) -> (Self, u64) {
            let p = u128::from(GOLDILOCKS_PRIME_U64);
            let is_butterfly = self.counter >= self.depth;
            let delayed = self.delay[self.depth - 1];

            let (data_out, delay_in, next_tw) = if is_butterfly {
                let upper =
                    u64::try_from((u128::from(delayed) + u128::from(data_in)) % p).unwrap_or(0);
                let diff = (u128::from(delayed) + p - u128::from(data_in)) % p;
                let lower = u64::try_from((diff * u128::from(self.twiddle)) % p).unwrap_or(0);
                let tw = u64::try_from((u128::from(self.twiddle) * u128::from(step_root)) % p)
                    .unwrap_or(0);
                (lower, upper, tw)
            } else {
                // R2SDF fill: output the delay tail, store input
                (delayed, data_in, 1)
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

            (
                RefState {
                    delay: next_delay,
                    twiddle: next_tw,
                    counter: next_counter,
                    depth: self.depth,
                },
                data_out,
            )
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
    fn depth_1_fill_emits_delay_tail() -> Result<(), Error> {
        let stage = sdf_stage(1)?;
        let step_root = 7_u64; // arbitrary

        // Depth 1: fill for 1 cycle, butterfly for 1 cycle.
        // Cycle 0 (fill): output should be delay tail (0 on first frame).
        let inputs = vec![
            make_input(42, true, step_root),
            make_input(99, true, step_root), // butterfly cycle
        ];

        let tb = Testbench::new(stage);
        let results = tb.run(inputs).run()?;

        let (out_0, valid_0) = read_output(results[0].value())?;
        assert_eq!(out_0, 0, "fill should emit delay tail (0 on first frame)");
        assert!(valid_0, "valid should be true");

        Ok(())
    }

    #[test]
    fn depth_1_matches_reference() -> Result<(), Error> {
        let stage = sdf_stage(1)?;
        let step_root = 1_u64; // simplest twiddle

        // Feed 4 cycles (2 periods of 2D=2 each)
        let data: Vec<u64> = vec![10, 20, 30, 40];

        let inputs: Vec<BitSeq> = data
            .iter()
            .map(|d| make_input(*d, true, step_root))
            .collect();

        let tb = Testbench::new(stage);
        let results = tb.run(inputs).run()?;

        // Compare against software reference
        let (_, ref_outputs) =
            data.iter()
                .fold((RefState::new(1), Vec::new()), |(state, outs), d| {
                    let (next_state, out) = state.step(*d, step_root);
                    (
                        next_state,
                        outs.into_iter().chain(core::iter::once(out)).collect(),
                    )
                });

        results
            .iter()
            .zip(ref_outputs.iter())
            .enumerate()
            .try_for_each(|(i, (sample, expected))| {
                let (actual, _) = read_output(sample.value())?;
                assert_eq!(
                    actual, *expected,
                    "cycle {i}: got {actual:#018x}, expected {expected:#018x}",
                );
                Ok::<(), Error>(())
            })?;

        Ok(())
    }

    #[test]
    fn depth_2_matches_reference() -> Result<(), Error> {
        let stage = sdf_stage(2)?;
        let step_root = 1_u64;

        // Feed 8 cycles (2 periods of 2D=4 each)
        let data: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let inputs: Vec<BitSeq> = data
            .iter()
            .map(|d| make_input(*d, true, step_root))
            .collect();

        let tb = Testbench::new(stage);
        let results = tb.run(inputs).run()?;

        let (_, ref_outputs) =
            data.iter()
                .fold((RefState::new(2), Vec::new()), |(state, outs), d| {
                    let (next_state, out) = state.step(*d, step_root);
                    (
                        next_state,
                        outs.into_iter().chain(core::iter::once(out)).collect(),
                    )
                });

        results
            .iter()
            .zip(ref_outputs.iter())
            .enumerate()
            .try_for_each(|(i, (sample, expected))| {
                let (actual, _) = read_output(sample.value())?;
                assert_eq!(
                    actual, *expected,
                    "cycle {i}: got {actual:#018x}, expected {expected:#018x}",
                );
                Ok::<(), Error>(())
            })?;

        Ok(())
    }

    #[test]
    fn depth_2_nontrivial_twiddle() -> Result<(), Error> {
        let stage = sdf_stage(2)?;
        let step_root = 7_u64; // non-trivial twiddle step

        let data: Vec<u64> = vec![100, 200, 300, 400];

        let inputs: Vec<BitSeq> = data
            .iter()
            .map(|d| make_input(*d, true, step_root))
            .collect();

        let tb = Testbench::new(stage);
        let results = tb.run(inputs).run()?;

        let (_, ref_outputs) =
            data.iter()
                .fold((RefState::new(2), Vec::new()), |(state, outs), d| {
                    let (next_state, out) = state.step(*d, step_root);
                    (
                        next_state,
                        outs.into_iter().chain(core::iter::once(out)).collect(),
                    )
                });

        results
            .iter()
            .zip(ref_outputs.iter())
            .enumerate()
            .try_for_each(|(i, (sample, expected))| {
                let (actual, _) = read_output(sample.value())?;
                assert_eq!(
                    actual, *expected,
                    "cycle {i}: got {actual:#018x}, expected {expected:#018x}",
                );
                Ok::<(), Error>(())
            })?;

        Ok(())
    }
}
