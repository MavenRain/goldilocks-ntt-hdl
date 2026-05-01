//! Time-unrolled Circom emission for SDF stages.
//!
//! Lowers [`goldilocks_ntt_hdl::hdl::stage::sdf_stage`] through
//! [`hdl_cat_circom::emit_unrolled_template`] and verifies the rendered
//! template's structure: template name, port counts per cycle, cycle-0
//! state initialization, and state plumbing across cycles.

use goldilocks_ntt_hdl::error::Error;
use goldilocks_ntt_hdl::hdl::stage::emit_sdf_stage_circom;

/// Render the depth-2, 4-cycle template and return the Circom text.
fn render_depth_2_4_cycles(name: &str) -> Result<String, Error> {
    emit_sdf_stage_circom(2, 4, name)?
        .run()
        .map_err(|e| Error::HdlCat(e.to_string()))
}

#[test]
fn emits_pragma_and_template_header() -> Result<(), Error> {
    let text = render_depth_2_4_cycles("sdf_stage_d2_c4")?;

    assert!(
        text.contains("pragma circom"),
        "missing circom pragma in:\n{text}",
    );
    assert!(
        text.contains("template sdf_stage_d2_c4"),
        "missing template header in:\n{text}",
    );
    Ok(())
}

#[test]
fn emits_per_cycle_data_input_ports() -> Result<(), Error> {
    let text = render_depth_2_4_cycles("sdf_stage_d2_c4")?;

    let input_lines = text.match_indices("signal input").count();
    let expected = 4 * 3;
    assert!(
        input_lines >= expected,
        "expected at least {expected} signal input lines (4 cycles * 3 data inputs), got {input_lines}",
    );

    (0..4_usize).try_for_each(|cycle| {
        let marker = format!("_c{cycle}[");
        let count = text.match_indices(&marker).count();
        if count >= 3 {
            Ok(())
        } else {
            Err(Error::HdlCat(format!(
                "cycle {cycle} has only {count} signal references with marker {marker}",
            )))
        }
    })
}

#[test]
fn emits_per_cycle_data_output_ports() -> Result<(), Error> {
    let text = render_depth_2_4_cycles("sdf_stage_d2_c4")?;

    let output_lines = text.match_indices("signal output").count();
    let expected = 4 * 2;
    assert!(
        output_lines >= expected,
        "expected at least {expected} signal output lines (4 cycles * 2 data outputs), got {output_lines}",
    );
    Ok(())
}

#[test]
fn emits_cycle_renamed_state_signals() -> Result<(), Error> {
    let text = render_depth_2_4_cycles("sdf_stage_d2_c4")?;

    // State wires are local signals (not ports) at every cycle.
    // delay_arr=w0 (state), twiddle=w1 (state), counter=w2 (state).
    (0..4_usize).try_for_each(|cycle| {
        [0_usize, 1, 2].iter().try_for_each(|state_idx| {
            let marker = format!("signal w{state_idx}_c{cycle}[");
            if text.contains(&marker) {
                Ok(())
            } else {
                Err(Error::HdlCat(format!(
                    "missing state signal declaration {marker} in template",
                )))
            }
        })
    })
}

#[test]
fn initializes_cycle_zero_state_from_initial_state() -> Result<(), Error> {
    let text = render_depth_2_4_cycles("sdf_stage_d2_c4")?;

    // Cycle-0 state wires get bit-by-bit constant assigns from the
    // expanded initial state.  Initial state for sdf_stage:
    //   delay_arr (128 bits): all zero
    //   twiddle (64 bits): bit 0 = 1, rest zero
    //   counter (24 bits): all zero
    //
    // The twiddle's bit-0 = 1 is the most distinctive marker.
    assert!(
        text.contains("w1_c0[0] <== 1;"),
        "missing twiddle cycle-0 bit-0 = 1 init line",
    );

    // Counter starts at zero.
    assert!(
        text.contains("w2_c0[0] <== 0;"),
        "missing counter cycle-0 bit-0 = 0 init line",
    );

    Ok(())
}

#[test]
fn plumbs_state_from_previous_next_state() -> Result<(), Error> {
    let text = render_depth_2_4_cycles("sdf_stage_d2_c4")?;

    // For cycles k > 0, state wires (w0, w1, w2) at cycle k are driven
    // by the corresponding next-state wires from cycle k-1.  The exact
    // next-state wire indices depend on graph allocation order; the
    // structural property we can verify is that each cycle k > 0 has
    // assignments to its state signals that reference some _c{k-1} wire.
    (1..4_usize).try_for_each(|cycle| {
        [0_usize, 1, 2].iter().try_for_each(|state_idx| {
            let prefix = format!("w{state_idx}_c{cycle}[0] <== ");
            let prev_marker = format!("_c{}[0]", cycle - 1);
            let has_plumbing = text
                .lines()
                .any(|line| line.contains(&prefix) && line.contains(&prev_marker));
            if has_plumbing {
                Ok(())
            } else {
                Err(Error::HdlCat(format!(
                    "cycle {cycle} state w{state_idx} not driven by a cycle-{} signal",
                    cycle - 1,
                )))
            }
        })
    })
}

#[test]
fn writes_to_target_for_inspection() -> Result<(), Error> {
    let text = render_depth_2_4_cycles("sdf_stage_d2_c4")?;

    std::fs::write("target/sdf_stage_d2_c4.circom", &text).map_err(Error::Io)?;

    Ok(())
}

#[test]
fn rejects_zero_cycles() -> Result<(), Error> {
    // Construction succeeds; the unrolled emitter surfaces the zero-cycle
    // error inside the Io at run time.
    let io = emit_sdf_stage_circom(2, 0, "bad")?;
    assert!(
        io.run().is_err(),
        "zero cycles should produce a runtime error"
    );
    Ok(())
}

#[test]
fn rejects_zero_depth() {
    let result = emit_sdf_stage_circom(0, 4, "bad");
    assert!(
        result.is_err(),
        "zero depth should produce a synchronous error"
    );
}
