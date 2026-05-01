//! SDF pipeline compositions of [`crate::hdl::stage::sdf_stage`]s.
//!
//! Each pipeline variant is a concrete composition of a small number of
//! SDF stages whose depths multiply to the NTT size.  For a radix-2 DIF
//! pipeline of size `N = 2^k`, stage `j` (for `j = 0..k`) has delay depth
//! `D_j = 2^(k - 1 - j)`.
//!
//! The step root for each stage is supplied via the pipeline's input bundle
//! rather than baked into the hardware struct, so the same pipeline can be
//! reused for both forward and inverse NTTs with an appropriate choice of
//! primitive roots.
//!
//! This module exposes a [`size_4_pipeline`] constructor (2 stages, depths
//! 2 and 1) composed by merging the underlying IR graphs with appropriate
//! wire remapping.

use comp_cat_rs::effect::io::Io;
use hdl_cat_bits::Bits;
use hdl_cat_circuit::{CircuitTensor, Obj};
use hdl_cat_error::Error;
use hdl_cat_ir::{HdlGraph, HdlGraphBuilder, WireId};
use hdl_cat_kind::BitSeq;
use hdl_cat_sync::Sync;

use crate::hdl::stage::sdf_stage;

/// Type alias for 64-bit Goldilocks field element.
pub type GoldilocksElement = Bits<64>;

/// Input bundle for size-4 pipeline: `(((data, valid), step_root_0), step_root_1)`.
pub type Size4PipelineInput = CircuitTensor<
    CircuitTensor<CircuitTensor<Obj<GoldilocksElement>, Obj<bool>>, Obj<GoldilocksElement>>,
    Obj<GoldilocksElement>,
>;

/// Output bundle for size-4 pipeline: `(data, valid)`.
pub type Size4PipelineOutput = CircuitTensor<Obj<GoldilocksElement>, Obj<bool>>;

/// Phantom state marker for the pipeline.
pub type Size4PipelineState = Obj<GoldilocksElement>;

/// A size-4 SDF pipeline as a sync machine.
pub type Size4PipelineSync = Sync<Size4PipelineState, Size4PipelineInput, Size4PipelineOutput>;

/// Merge two [`HdlGraph`]s, remapping the second graph's wire references.
///
/// Replicates the internal `merge_graphs` from `hdl-cat-sync` so that
/// we can perform partial composition (where only some of `g`'s data
/// inputs are substituted).
fn merge_graphs(
    f_graph: &HdlGraph,
    g_graph: &HdlGraph,
    remap_g: impl Fn(WireId) -> WireId + Clone,
) -> HdlGraph {
    // Allocate wires from both graphs
    let bld = f_graph
        .wires()
        .iter()
        .cloned()
        .chain(g_graph.wires().iter().cloned())
        .fold(HdlGraphBuilder::new(), |b, ty| b.with_wire(ty).0);

    // Copy f's instructions as-is
    let bld = f_graph.instructions().iter().try_fold(bld, |b, instr| {
        b.with_instruction(instr.op().clone(), instr.inputs().to_vec(), instr.output())
    });

    // Copy g's instructions with wire remapping
    let bld = bld.and_then(|b| {
        g_graph.instructions().iter().try_fold(b, |b, instr| {
            let new_inputs: Vec<WireId> = instr
                .inputs()
                .iter()
                .copied()
                .map(remap_g.clone())
                .collect();
            let new_output = remap_g(instr.output());
            b.with_instruction(instr.op().clone(), new_inputs, new_output)
        })
    });

    bld.map_or_else(|_| HdlGraphBuilder::new().build(), HdlGraphBuilder::build)
}

/// Construct a size-4 SDF pipeline.
///
/// Composes two SDF stages with depths 2 and 1 to create a 4-point NTT
/// pipeline.  Stage 0 (depth 2) feeds stage 1 (depth 1); each stage
/// receives its own step root from the pipeline's input bundle.
///
/// The composition merges the two stages' IR graphs at the wire level:
/// stage 1's `data_in` and `valid_in` are substituted with stage 0's
/// `data_out` and `valid_out`, while `step_root_1` remains a
/// pipeline-level input.
///
/// # Errors
///
/// Returns [`Error`] if stage construction fails.
#[allow(clippy::similar_names)]
pub fn size_4_pipeline() -> Result<Size4PipelineSync, Error> {
    let stage_0 = sdf_stage(2)?;
    let stage_1 = sdf_stage(1)?;

    let (f_graph, f_inputs, f_outputs, f_init, f_sc) = stage_0.into_parts();
    let (g_graph, g_inputs, g_outputs, g_init, g_sc) = stage_1.into_parts();

    let f_wire_count = f_graph.wires().len();

    // Split into state and data portions
    let (state_f_in, data_f_in) = f_inputs.split_at(f_sc);
    let (state_f_out, data_f_out) = f_outputs.split_at(f_sc);
    let (state_g_in, data_g_in) = g_inputs.split_at(g_sc);
    let (state_g_out, data_g_out) = g_outputs.split_at(g_sc);

    // data_f_out = [data_out, valid_out]
    // data_g_in  = [data_in, valid_in, step_root]
    //
    // Substitute: data_g_in[0] -> data_f_out[0]  (data flow)
    //             data_g_in[1] -> data_f_out[1]  (valid flow)
    // Keep:       data_g_in[2] as pipeline input  (step_root_1)
    let shift = |w: WireId| WireId::new(w.index() + f_wire_count);

    let substitution: Vec<(WireId, WireId)> = data_g_in[..2]
        .iter()
        .zip(data_f_out.iter())
        .map(|(g_w, f_w)| (shift(*g_w), *f_w))
        .collect();

    let remap_g = move |w: WireId| -> WireId {
        let shifted = WireId::new(w.index() + f_wire_count);
        substitution
            .iter()
            .find_map(|(from, to)| (*from == shifted).then_some(*to))
            .unwrap_or(shifted)
    };

    let merged = merge_graphs(&f_graph, &g_graph, remap_g);

    // Pipeline inputs: state_0 ++ state_1 ++ data_in,valid,root_0 ++ root_1
    let pipeline_inputs: Vec<WireId> = state_f_in
        .iter()
        .copied()
        .chain(state_g_in.iter().copied().map(shift))
        .chain(data_f_in.iter().copied())
        .chain(data_g_in[2..].iter().copied().map(shift))
        .collect();

    // Pipeline outputs: next_state_0 ++ next_state_1 ++ stage_1 data_out
    let pipeline_outputs: Vec<WireId> = state_f_out
        .iter()
        .copied()
        .chain(state_g_out.iter().copied().map(shift))
        .chain(data_g_out.iter().copied().map(shift))
        .collect();

    let combined_state = f_init.concat(g_init);
    let combined_sc = f_sc + g_sc;

    Ok(hdl_cat_sync::machine::from_raw(
        merged,
        pipeline_inputs,
        pipeline_outputs,
        combined_state,
        combined_sc,
    ))
}

/// Accumulated state carried through the stage-folding loop.
///
/// At each step, this captures the merged graph so far plus the wire
/// routing needed to connect the next stage.
struct PipelineAccum {
    /// The merged IR graph so far.
    graph: HdlGraph,
    /// State input wires (for `state_wire_count` calculation).
    state_inputs: Vec<WireId>,
    /// State output wires (next-state feeds).
    state_outputs: Vec<WireId>,
    /// Data input wires from the first stage (pipeline-level inputs).
    data_inputs: Vec<WireId>,
    /// Step root wires collected so far (one per stage).
    step_root_inputs: Vec<WireId>,
    /// Data output wires from the last stage (pipeline-level outputs).
    data_outputs: Vec<WireId>,
    /// Concatenated initial state.
    initial_state: BitSeq,
    /// Total state wire count.
    state_wire_count: usize,
}

/// Split a data-wire slice into `(data_in, valid_in)` and `step_root`.
///
/// SDF stages produce data wire layouts `[data_in, valid_in, step_root]`.
/// Returns `None` if the slice has fewer than 3 elements.
fn split_data_wires(data: &[WireId]) -> Option<(&[WireId], WireId)> {
    let (dv, roots) = data.split_at(2.min(data.len()));
    (dv.len() == 2)
        .then(|| roots.first().copied())
        .flatten()
        .map(|r| (dv, r))
}

/// Compose N SDF stages into a single pipeline [`Sync`] machine.
///
/// Stage `j` (for `j = 0..num_stages`) has delay depth
/// `depths[j]`.  The stages are chained: stage j's `data_out` /
/// `valid_out` feed stage j+1's `data_in` / `valid_in`.  Each stage
/// receives its own `step_root` from the pipeline-level input bundle.
///
/// # Arguments
///
/// * `depths` - Slice of delay depths, one per stage.
///
/// # Errors
///
/// Returns [`Error`] if any stage construction or graph merge fails.
#[allow(clippy::similar_names, clippy::too_many_lines)]
pub fn compose_pipeline(
    depths: &[usize],
) -> Result<Sync<Obj<GoldilocksElement>, PipelineInput, PipelineOutput>, Error> {
    let (&first_depth, rest_depths) = depths.split_first().ok_or_else(|| Error::WidthMismatch {
        expected: hdl_cat_error::Width::new(1),
        actual: hdl_cat_error::Width::new(0),
    })?;

    let stage_0 = sdf_stage(first_depth)?;
    let (f_graph, f_inputs, f_outputs, f_init, f_sc) = stage_0.into_parts();
    let (state_f_in, data_f_in) = f_inputs.split_at(f_sc);
    let (state_f_out, data_f_out) = f_outputs.split_at(f_sc);

    // data_f_in = [data_in, valid_in, step_root_0]
    // data_f_out = [data_out, valid_out]
    let (data_in_valid, step_root_0) =
        split_data_wires(data_f_in).ok_or_else(|| Error::WidthMismatch {
            expected: hdl_cat_error::Width::new(3),
            actual: hdl_cat_error::Width::new(u32::try_from(data_f_in.len()).unwrap_or(0)),
        })?;

    let accum = PipelineAccum {
        graph: f_graph,
        state_inputs: state_f_in.to_vec(),
        state_outputs: state_f_out.to_vec(),
        data_inputs: data_in_valid.to_vec(),
        step_root_inputs: vec![step_root_0],
        data_outputs: data_f_out.to_vec(),
        initial_state: f_init,
        state_wire_count: f_sc,
    };

    let result = rest_depths.iter().try_fold(accum, |acc, &depth| {
        let stage = sdf_stage(depth)?;
        let (g_graph, g_inputs, g_outputs, g_init, g_sc) = stage.into_parts();

        let f_wire_count = acc.graph.wires().len();
        let shift = |w: WireId| WireId::new(w.index() + f_wire_count);

        let (state_g_in, data_g_in) = g_inputs.split_at(g_sc);
        let (state_g_out, data_g_out) = g_outputs.split_at(g_sc);

        // data_g_in = [data_in, valid_in, step_root]
        let (g_dv, g_step_root) =
            split_data_wires(data_g_in).ok_or_else(|| Error::WidthMismatch {
                expected: hdl_cat_error::Width::new(3),
                actual: hdl_cat_error::Width::new(u32::try_from(data_g_in.len()).unwrap_or(0)),
            })?;

        // Substitute: g data_in -> acc data_out, g valid_in -> acc valid_out
        let substitution: Vec<(WireId, WireId)> = g_dv
            .iter()
            .zip(acc.data_outputs.iter())
            .map(|(g_w, f_w)| (shift(*g_w), *f_w))
            .collect();

        let remap_g = move |w: WireId| -> WireId {
            let shifted = WireId::new(w.index() + f_wire_count);
            substitution
                .iter()
                .find_map(|(from, to)| (*from == shifted).then_some(*to))
                .unwrap_or(shifted)
        };

        let merged = merge_graphs(&acc.graph, &g_graph, remap_g);

        Ok::<PipelineAccum, Error>(PipelineAccum {
            graph: merged,
            state_inputs: acc
                .state_inputs
                .into_iter()
                .chain(state_g_in.iter().copied().map(shift))
                .collect(),
            state_outputs: acc
                .state_outputs
                .into_iter()
                .chain(state_g_out.iter().copied().map(shift))
                .collect(),
            data_inputs: acc.data_inputs,
            step_root_inputs: acc
                .step_root_inputs
                .into_iter()
                .chain(core::iter::once(shift(g_step_root)))
                .collect(),
            data_outputs: data_g_out.iter().copied().map(shift).collect(),
            initial_state: acc.initial_state.concat(g_init),
            state_wire_count: acc.state_wire_count + g_sc,
        })
    })?;

    // Pipeline inputs: state ++ data_in,valid ++ step_root_0 ++ ... ++ step_root_{N-1}
    let pipeline_inputs: Vec<WireId> = result
        .state_inputs
        .into_iter()
        .chain(result.data_inputs)
        .chain(result.step_root_inputs)
        .collect();

    // Pipeline outputs: next_state ++ last stage data_out
    let pipeline_outputs: Vec<WireId> = result
        .state_outputs
        .into_iter()
        .chain(result.data_outputs)
        .collect();

    Ok(hdl_cat_sync::machine::from_raw(
        result.graph,
        pipeline_inputs,
        pipeline_outputs,
        result.initial_state,
        result.state_wire_count,
    ))
}

/// Input bundle for an N-stage pipeline.
///
/// The exact circuit-tensor shape depends on the number of stages,
/// but the flat wire layout is:
/// `(data_in:64, valid_in:1, step_root_0:64, ..., step_root_{N-1}:64)`.
pub type PipelineInput =
    CircuitTensor<CircuitTensor<Obj<GoldilocksElement>, Obj<bool>>, Obj<GoldilocksElement>>;

/// Output bundle for an N-stage pipeline: `(data:64, valid:1)`.
pub type PipelineOutput = CircuitTensor<Obj<GoldilocksElement>, Obj<bool>>;

/// Emit the size-4 pipeline to Verilog.
///
/// Produces a stateful Verilog module with `clk` and `rst` ports,
/// using [`hdl_cat_verilog::emit_sync_graph`] to handle the pipeline's
/// internal state registers.
///
/// # Errors
///
/// Returns [`Error`] if pipeline construction fails.
pub fn emit_size_4_pipeline_verilog() -> Result<Io<hdl_cat_error::Error, String>, Error> {
    let pipeline = size_4_pipeline()?;
    let (graph, input_wires, output_wires, initial_state, state_wire_count) = pipeline.into_parts();

    let module = hdl_cat_verilog::emit_sync_graph(
        &graph,
        "size_4_ntt_pipeline",
        state_wire_count,
        &input_wires,
        &output_wires,
        &initial_state,
    );

    Ok(module.flat_map(|m| m.render()))
}

/// Emit an N-stage pipeline to Verilog with BRAM-inferable delay arrays.
///
/// Composes the pipeline via [`compose_pipeline`], then emits using
/// [`hdl_cat_verilog::emit_sync_graph`].  Array-typed state wires
/// (from [`crate::hdl::stage::sdf_stage`]) are auto-detected by
/// the emitter and rendered as `reg` arrays + shift/circular-buffer
/// blocks.
///
/// # Arguments
///
/// * `depths` - Delay depth for each stage.
/// * `name` - Verilog module name.
///
/// # Errors
///
/// Returns [`Error`] if pipeline construction or Verilog emission fails.
pub fn emit_pipeline_verilog(
    depths: &[usize],
    name: &str,
) -> Result<Io<hdl_cat_error::Error, String>, Error> {
    let pipeline = compose_pipeline(depths)?;
    let (graph, input_wires, output_wires, initial_state, state_wire_count) = pipeline.into_parts();

    let module = hdl_cat_verilog::emit_sync_graph(
        &graph,
        name,
        state_wire_count,
        &input_wires,
        &output_wires,
        &initial_state,
    );

    Ok(module.flat_map(|m| m.render()))
}

/// Emit the size-4 pipeline as a time-unrolled Circom template.
///
/// Lowers [`size_4_pipeline`] through
/// [`hdl_cat_circom::emit_unrolled_template`], producing a flat
/// combinational template with per-cycle input/output buses and
/// cycle-0 state initialised from the pipeline's `initial_state`.
///
/// The unrolled template grows as `O(num_cycles * graph_size)`, so
/// this is intended for ZK structural verification rather than
/// production proving at large `num_cycles`.
///
/// # Arguments
///
/// * `num_cycles` - Number of cycles to unroll (must be >= 1).
///
/// # Errors
///
/// Returns [`Error`] if pipeline construction fails.  The returned
/// [`Io`] surfaces emitter errors at run time.
pub fn emit_size_4_pipeline_circom(
    num_cycles: usize,
) -> Result<Io<hdl_cat_error::Error, String>, Error> {
    let pipeline = size_4_pipeline()?;
    let (graph, input_wires, output_wires, initial_state, state_wire_count) = pipeline.into_parts();

    let template = hdl_cat_circom::emit_unrolled_template(
        &graph,
        "size_4_ntt_pipeline",
        &input_wires,
        &output_wires,
        state_wire_count,
        &initial_state,
        num_cycles,
    );

    Ok(template.flat_map(|t| t.render()))
}

/// Emit an N-stage pipeline as a time-unrolled Circom template.
///
/// Composes the pipeline via [`compose_pipeline`], then lowers it
/// through [`hdl_cat_circom::emit_unrolled_template`].  Array-typed
/// state wires (delay lines) are flattened to bit-level signals at
/// every cycle, with cycle-`k+1` state driven from cycle-`k`'s
/// next-state outputs.
///
/// # Arguments
///
/// * `depths` - Delay depth for each stage.
/// * `num_cycles` - Number of cycles to unroll (must be >= 1).
/// * `name` - Circom template name.
///
/// # Errors
///
/// Returns [`Error`] if pipeline construction fails.  The returned
/// [`Io`] surfaces emitter errors at run time.
pub fn emit_pipeline_circom(
    depths: &[usize],
    num_cycles: usize,
    name: &str,
) -> Result<Io<hdl_cat_error::Error, String>, Error> {
    let pipeline = compose_pipeline(depths)?;
    let (graph, input_wires, output_wires, initial_state, state_wire_count) = pipeline.into_parts();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdl::common::{GOLDILOCKS_PRIME_U64, bitseq_to_u64, u64_to_bitseq};
    use hdl_cat_kind::BitSeq;
    use hdl_cat_sim::Testbench;

    /// Pack a pipeline input cycle into a [`BitSeq`].
    fn make_input(data: u64, valid: bool, step_root_0: u64, step_root_1: u64) -> BitSeq {
        u64_to_bitseq(data)
            .concat(BitSeq::from_vec(vec![valid]))
            .concat(u64_to_bitseq(step_root_0))
            .concat(u64_to_bitseq(step_root_1))
    }

    /// Unpack a pipeline output cycle from a [`BitSeq`].
    fn read_output(bits: &BitSeq) -> Result<(u64, bool), Error> {
        let (data_bits, valid_bits) = bits.clone().split_at(64);
        let data = bitseq_to_u64(&data_bits)?;
        let valid = valid_bits.bit(0);
        Ok((data, valid))
    }

    // ── Software reference: two-stage SDF pipeline ───────────────────

    struct RefStage {
        delay: Vec<u64>,
        twiddle: u64,
        counter: usize,
        depth: usize,
    }

    impl RefStage {
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
                // R2SDF fill: output delay tail, store input
                (delayed, data_in, 1)
            };

            let next_delay: Vec<u64> = core::iter::once(delay_in)
                .chain(self.delay[..self.depth - 1].iter().copied())
                .collect();

            let next_counter = if self.counter == 2 * self.depth - 1 {
                0
            } else {
                self.counter + 1
            };

            (
                RefStage {
                    delay: next_delay,
                    twiddle: next_tw,
                    counter: next_counter,
                    depth: self.depth,
                },
                data_out,
            )
        }
    }

    fn ref_pipeline_step(
        s0: RefStage,
        s1: RefStage,
        data_in: u64,
        step_root_0: u64,
        step_root_1: u64,
    ) -> (RefStage, RefStage, u64) {
        let (s0, mid) = s0.step(data_in, step_root_0);
        let (s1, out) = s1.step(mid, step_root_1);
        (s0, s1, out)
    }

    // ── Tests ────────────────────────────────────────────────────────

    #[test]
    fn size_4_pipeline_builds() -> Result<(), Error> {
        let _pipeline = size_4_pipeline()?;
        Ok(())
    }

    #[test]
    fn pipeline_matches_reference() -> Result<(), Error> {
        let pipeline = size_4_pipeline()?;
        let step_root_0 = 1_u64;
        let step_root_1 = 1_u64;

        let data: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let inputs: Vec<BitSeq> = data
            .iter()
            .map(|d| make_input(*d, true, step_root_0, step_root_1))
            .collect();

        let tb = Testbench::new(pipeline);
        let results = tb.run(inputs).run()?;

        // Run the software reference
        let (_, ref_outputs) = data.iter().fold(
            ((RefStage::new(2), RefStage::new(1)), Vec::new()),
            |((s0, s1), outs), d| {
                let (s0, s1, out) = ref_pipeline_step(s0, s1, *d, step_root_0, step_root_1);
                (
                    (s0, s1),
                    outs.into_iter().chain(core::iter::once(out)).collect(),
                )
            },
        );

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
    fn pipeline_nontrivial_roots() -> Result<(), Error> {
        use crate::field::roots::primitive_root_of_unity;

        let pipeline = size_4_pipeline()?;

        let step_root_0 = primitive_root_of_unity(2)
            .map_err(|_| Error::WidthMismatch {
                expected: hdl_cat_error::Width::new(64),
                actual: hdl_cat_error::Width::new(0),
            })?
            .value();
        let step_root_1 = primitive_root_of_unity(1)
            .map_err(|_| Error::WidthMismatch {
                expected: hdl_cat_error::Width::new(64),
                actual: hdl_cat_error::Width::new(0),
            })?
            .value();

        let data: Vec<u64> = vec![1, 2, 3, 4];

        let inputs: Vec<BitSeq> = data
            .iter()
            .map(|d| make_input(*d, true, step_root_0, step_root_1))
            .collect();

        let tb = Testbench::new(pipeline);
        let results = tb.run(inputs).run()?;

        let (_, ref_outputs) = data.iter().fold(
            ((RefStage::new(2), RefStage::new(1)), Vec::new()),
            |((s0, s1), outs), d| {
                let (s0, s1, out) = ref_pipeline_step(s0, s1, *d, step_root_0, step_root_1);
                (
                    (s0, s1),
                    outs.into_iter().chain(core::iter::once(out)).collect(),
                )
            },
        );

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
    fn compose_pipeline_builds_size_4() -> Result<(), Error> {
        let _pipeline = compose_pipeline(&[2, 1])?;
        Ok(())
    }

    #[test]
    fn compose_pipeline_matches_reference() -> Result<(), Error> {
        let pipeline = compose_pipeline(&[2, 1])?;
        let step_root_0 = 1_u64;
        let step_root_1 = 1_u64;

        let data: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let inputs: Vec<BitSeq> = data
            .iter()
            .map(|d| make_input(*d, true, step_root_0, step_root_1))
            .collect();

        let tb = Testbench::new(pipeline);
        let results = tb.run(inputs).run()?;

        let (_, ref_outputs) = data.iter().fold(
            ((RefStage::new(2), RefStage::new(1)), Vec::new()),
            |((s0, s1), outs), d| {
                let (s0, s1, out) = ref_pipeline_step(s0, s1, *d, step_root_0, step_root_1);
                (
                    (s0, s1),
                    outs.into_iter().chain(core::iter::once(out)).collect(),
                )
            },
        );

        results
            .iter()
            .zip(ref_outputs.iter())
            .enumerate()
            .try_for_each(|(i, (sample, expected))| {
                let (actual, _) = read_output(sample.value())?;
                assert_eq!(
                    actual, *expected,
                    "compose_pipeline cycle {i}: got {actual:#018x}, expected {expected:#018x}",
                );
                Ok::<(), Error>(())
            })?;

        Ok(())
    }

    #[test]
    fn compose_pipeline_empty_depths_errors() {
        assert!(compose_pipeline(&[]).is_err());
    }

    #[test]
    fn compose_pipeline_single_stage() -> Result<(), Error> {
        let _pipeline = compose_pipeline(&[4])?;
        Ok(())
    }

    #[test]
    fn verilog_emission_produces_text() -> Result<(), crate::error::Error> {
        let verilog_io = emit_size_4_pipeline_verilog()?;
        let text = verilog_io
            .run()
            .map_err(|e| crate::error::Error::VerilogGen(e.to_string()))?;
        assert!(text.contains("module"));
        assert!(text.contains("size_4_ntt_pipeline"));
        Ok(())
    }

    /// Count occurrences of a pattern in a string.
    fn count_occurrences(text: &str, pattern: &str) -> usize {
        text.match_indices(pattern).count()
    }

    #[test]
    fn bram_emission_size_4_produces_array_decls() -> Result<(), crate::error::Error> {
        let verilog_io = emit_pipeline_verilog(&[2, 1], "bram_ntt_4")?;
        let text = verilog_io
            .run()
            .map_err(|e| crate::error::Error::VerilogGen(e.to_string()))?;
        assert!(text.contains("module"), "missing module keyword");
        assert!(text.contains("bram_ntt_4"), "missing module name");
        // Two stages produce two auto-detected array declarations.
        let arr_decl_count = count_occurrences(&text, "reg [63:0]");
        assert!(
            arr_decl_count >= 2,
            "expected >= 2 array decls, got {arr_decl_count}"
        );
        Ok(())
    }

    #[test]
    fn bram_emission_3_stage_produces_array_decls() -> Result<(), crate::error::Error> {
        let verilog_io = emit_pipeline_verilog(&[4, 2, 1], "bram_ntt_8")?;
        let text = verilog_io
            .run()
            .map_err(|e| crate::error::Error::VerilogGen(e.to_string()))?;
        assert!(text.contains("module"), "missing module keyword");
        assert!(text.contains("bram_ntt_8"), "missing module name");
        let arr_decl_count = count_occurrences(&text, "reg [63:0]");
        assert!(
            arr_decl_count >= 3,
            "expected >= 3 array decls, got {arr_decl_count}"
        );
        Ok(())
    }

    #[test]
    fn bram_emission_4_stage() -> Result<(), crate::error::Error> {
        let verilog_io = emit_pipeline_verilog(&[8, 4, 2, 1], "bram_ntt_16")?;
        let text = verilog_io
            .run()
            .map_err(|e| crate::error::Error::VerilogGen(e.to_string()))?;
        assert!(text.contains("module"), "missing module keyword");
        assert!(text.contains("bram_ntt_16"), "missing module name");
        let arr_decl_count = count_occurrences(&text, "reg [63:0]");
        assert!(
            arr_decl_count >= 4,
            "expected >= 4 array decls, got {arr_decl_count}"
        );
        Ok(())
    }

    #[test]
    fn bram_emission_single_stage() -> Result<(), crate::error::Error> {
        let verilog_io = emit_pipeline_verilog(&[4], "bram_ntt_single")?;
        let text = verilog_io
            .run()
            .map_err(|e| crate::error::Error::VerilogGen(e.to_string()))?;
        assert!(text.contains("module"), "missing module keyword");
        assert!(text.contains("bram_ntt_single"), "missing module name");
        // Single stage: one array decl for the delay line.
        assert!(text.contains("[0:3]"), "missing array depth [0:3]");
        Ok(())
    }

    #[test]
    fn bram_emission_circ_buf_for_large_depth() -> Result<(), crate::error::Error> {
        // depth 64 > CIRC_BUF_THRESHOLD (32), so stage 0 uses CircularBuffer
        let verilog_io = emit_pipeline_verilog(&[64, 1], "circ_buf_ntt")?;
        let text = verilog_io
            .run()
            .map_err(|e| crate::error::Error::VerilogGen(e.to_string()))?;
        assert!(text.contains("module"), "missing module keyword");
        assert!(text.contains("circ_buf_ntt"), "missing module name");
        // Stage 0 (depth 64) gets circular buffer: pointer reg + dynamic index.
        assert!(text.contains("_ptr"), "missing circular buffer pointer");
        // Should NOT have shift-register O(D) lines for stage 0's array.
        assert!(
            !text.contains("[63] <= "),
            "should not have O(D) shift lines for large depth",
        );
        // Stage 1 (depth 1) stays as shift register (below threshold).
        assert!(text.contains("[0:0]"), "missing depth-1 array decl");
        Ok(())
    }

    #[test]
    fn bram_emission_small_depth_stays_shift_register() -> Result<(), crate::error::Error> {
        // depth 16 <= CIRC_BUF_THRESHOLD, should use shift register
        let verilog_io = emit_pipeline_verilog(&[16, 1], "shift_reg_ntt")?;
        let text = verilog_io
            .run()
            .map_err(|e| crate::error::Error::VerilogGen(e.to_string()))?;
        assert!(text.contains("module"), "missing module keyword");
        // Stage 0 (depth 16) uses shift register, no circular buffer pointer.
        assert!(text.contains("[0:15]"), "missing array depth [0:15]");
        // Should have shift-register lines.
        assert!(text.contains("[1] <="), "missing shift line");
        Ok(())
    }
}
