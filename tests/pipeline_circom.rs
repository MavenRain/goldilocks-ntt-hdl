//! Time-unrolled Circom emission for SDF pipelines.
//!
//! Lowers [`goldilocks_ntt_hdl::hdl::pipeline`] composers through
//! [`hdl_cat_circom::emit_unrolled_template`] and verifies the rendered
//! template's structure: template name, port counts per cycle, state
//! initialization, and state plumbing across cycles for a 2-stage
//! (depths `[2, 1]`) pipeline.

use goldilocks_ntt_hdl::error::Error;
use goldilocks_ntt_hdl::hdl::pipeline::{emit_pipeline_circom, emit_size_4_pipeline_circom};

/// Render the size-4 pipeline at a small cycle count and return the text.
fn render_size_4(num_cycles: usize) -> Result<String, Error> {
    emit_size_4_pipeline_circom(num_cycles)?
        .run()
        .map_err(|e| Error::HdlCat(e.to_string()))
}

/// Render the generic pipeline composer at depths and cycles.
fn render_pipeline(depths: &[usize], num_cycles: usize, name: &str) -> Result<String, Error> {
    emit_pipeline_circom(depths, num_cycles, name)?
        .run()
        .map_err(|e| Error::HdlCat(e.to_string()))
}

#[test]
fn size_4_emits_pragma_and_template_header() -> Result<(), Error> {
    let text = render_size_4(2)?;

    assert!(text.contains("pragma circom"), "missing circom pragma");
    assert!(
        text.contains("template size_4_ntt_pipeline"),
        "missing template header in:\n{}",
        text.lines().take(20).collect::<Vec<_>>().join("\n"),
    );
    Ok(())
}

#[test]
fn size_4_emits_per_cycle_data_input_ports() -> Result<(), Error> {
    let text = render_size_4(2)?;

    // size-4 pipeline data inputs per cycle: data_in (64), valid_in (1),
    // step_root_0 (64), step_root_1 (64) = 4 data inputs.
    let input_lines = text.match_indices("signal input").count();
    let expected = 2 * 4;
    assert!(
        input_lines >= expected,
        "expected at least {expected} signal input lines (2 cycles * 4 data inputs), got {input_lines}",
    );
    Ok(())
}

#[test]
fn size_4_emits_per_cycle_data_output_ports() -> Result<(), Error> {
    let text = render_size_4(2)?;

    // size-4 pipeline data outputs per cycle: data_out (64), valid_out (1) = 2.
    let output_lines = text.match_indices("signal output").count();
    let expected = 2 * 2;
    assert!(
        output_lines >= expected,
        "expected at least {expected} signal output lines (2 cycles * 2 data outputs), got {output_lines}",
    );
    Ok(())
}

#[test]
fn size_4_plumbs_state_across_cycles() -> Result<(), Error> {
    let text = render_size_4(2)?;

    // 2 stages * 3 state wires/stage = 6 state wires per cycle.  At cycle 1,
    // each state wire bit is driven from a cycle-0 next-state wire.
    let cycle_1_lhs = text.match_indices("_c1[0] <==").count();
    assert!(
        cycle_1_lhs > 0,
        "no cycle-1 state-bit assigns found; pipeline state plumbing missing",
    );

    let crosses_cycle = text
        .lines()
        .filter(|l| l.contains("_c1[0] <==") && l.contains("_c0["))
        .count();
    assert!(
        crosses_cycle > 0,
        "no cycle-1 assignment driven from a cycle-0 wire found",
    );
    Ok(())
}

#[test]
fn size_4_writes_target_for_inspection() -> Result<(), Error> {
    let text = render_size_4(2)?;
    std::fs::write("target/size_4_ntt_pipeline_c2.circom", &text).map_err(Error::Io)?;
    Ok(())
}

#[test]
fn pipeline_emits_named_template() -> Result<(), Error> {
    let text = render_pipeline(&[2, 1], 2, "ntt_size_4_unrolled")?;

    assert!(
        text.contains("template ntt_size_4_unrolled"),
        "missing template header for custom name",
    );
    Ok(())
}

#[test]
fn pipeline_writes_target_for_inspection() -> Result<(), Error> {
    let text = render_pipeline(&[2, 1], 2, "ntt_size_4_unrolled")?;
    std::fs::write("target/ntt_size_4_unrolled.circom", &text).map_err(Error::Io)?;
    Ok(())
}

#[test]
fn pipeline_rejects_zero_cycles() -> Result<(), Error> {
    let io = emit_pipeline_circom(&[2, 1], 0, "bad")?;
    assert!(io.run().is_err(), "zero cycles should error at run time");
    Ok(())
}

#[test]
fn pipeline_rejects_empty_depths() {
    let result = emit_pipeline_circom(&[], 2, "bad");
    assert!(result.is_err(), "empty depths should error synchronously");
}
