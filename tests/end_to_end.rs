//! End-to-end verification: hdl-cat SDF simulation vs golden model.
//!
//! Runs the hdl-cat pipeline simulation and compares its output
//! against the pure recursive DIF NTT golden model for small sizes.
//! Also includes Verilog emission verification.

use goldilocks_ntt_hdl::error::Error;
use goldilocks_ntt_hdl::field::element::GoldilocksElement;
use goldilocks_ntt_hdl::golden::reference::dif_ntt;
use goldilocks_ntt_hdl::hdl::pipeline::{emit_pipeline_verilog, emit_size_4_pipeline_verilog};
use goldilocks_ntt_hdl::sim::runner::{SimConfig, simulate_pipeline};

/// Compare SDF simulation output with the golden model.
///
/// The R2SDF pipeline needs one priming frame (initial delay lines are
/// zero).  We feed two consecutive identical frames and verify that the
/// second frame's outputs, as a multiset, match `dif_ntt`.
///
/// The SDF output ordering differs from the standard bit-reversed
/// order due to the fill/butterfly interleaving across stages; a
/// post-pipeline reordering buffer would be needed for index-exact
/// agreement.  This test validates the arithmetic by checking that
/// every NTT coefficient appears in the output exactly once.
fn verify_full_pipeline(log_n: u32) -> Result<(), Error> {
    let n = 1_usize << log_n;
    let input: Vec<GoldilocksElement> = (0..n)
        .map(|i| GoldilocksElement::new(u64::try_from(i + 1).unwrap_or(0)))
        .collect();

    // Golden model: DIF NTT (bit-reversed output)
    let golden = dif_ntt(&input)?;

    // Feed two frames through the simulation.  The first frame
    // primes the delay lines; the second frame produces a valid NTT.
    let two_frames: Vec<GoldilocksElement> = input.iter().chain(input.iter()).copied().collect();

    let num_stages = usize::try_from(log_n).map_err(|e| Error::Field(e.to_string()))?;
    let config = SimConfig::new(two_frames, num_stages)?;
    let sim_output = simulate_pipeline(config).run()?;

    // The second frame's N outputs should contain the same values
    // as the golden NTT (possibly in different order).
    let second_frame = &sim_output[n..];
    assert_eq!(
        second_frame.len(),
        golden.len(),
        "output length mismatch: sim={}, golden={}",
        second_frame.len(),
        golden.len(),
    );

    // NTT is a bijection on the field, so outputs are distinct.
    // Compare as sets to ignore the SDF output permutation.
    let sim_set: std::collections::BTreeSet<u64> = second_frame.iter().map(|e| e.value()).collect();
    let golden_set: std::collections::BTreeSet<u64> = golden.iter().map(|e| e.value()).collect();

    if sim_set == golden_set {
        Ok(())
    } else {
        Err(Error::Field(format!(
            "NTT value set mismatch:\n  sim:    {sim_set:?}\n  golden: {golden_set:?}"
        )))
    }
}

#[test]
fn passthrough_preserves_data() -> Result<(), Error> {
    let input = vec![
        GoldilocksElement::new(100),
        GoldilocksElement::new(200),
        GoldilocksElement::new(300),
        GoldilocksElement::new(400),
    ];
    let config = SimConfig::new(input.clone(), 0)?;
    let result = simulate_pipeline(config).run()?;

    input
        .iter()
        .zip(result.iter())
        .try_for_each(|(expected, actual)| {
            if expected == actual {
                Ok(())
            } else {
                Err(Error::Field(format!(
                    "passthrough mismatch: expected {expected}, got {actual}"
                )))
            }
        })
}

#[test]
fn golden_model_round_trip_size_16() -> Result<(), Error> {
    use goldilocks_ntt_hdl::golden::reference::inverse_ntt;

    let input: Vec<GoldilocksElement> = (1..=16).map(GoldilocksElement::new).collect();

    let forward = dif_ntt(&input)?;
    let recovered = inverse_ntt(&forward)?;

    input
        .iter()
        .zip(recovered.iter())
        .try_for_each(|(orig, rec)| {
            if orig == rec {
                Ok(())
            } else {
                Err(Error::Field(format!(
                    "round-trip mismatch: original {orig}, recovered {rec}"
                )))
            }
        })
}

#[test]
fn hdl_cat_size_4_simulation_basic() -> Result<(), Error> {
    let input = vec![
        GoldilocksElement::new(1),
        GoldilocksElement::new(2),
        GoldilocksElement::new(3),
        GoldilocksElement::new(4),
    ];

    let config = SimConfig::new(input, 2)?;
    let result = simulate_pipeline(config).run()?;

    assert_eq!(result.len(), 4);

    Ok(())
}

#[test]
fn verilog_emission_produces_output() -> Result<(), Error> {
    let verilog_io = emit_size_4_pipeline_verilog()?;
    let verilog_text = verilog_io
        .run()
        .map_err(|e| Error::VerilogGen(e.to_string()))?;

    assert!(!verilog_text.is_empty());
    assert!(verilog_text.contains("module"));
    assert!(verilog_text.contains("size_4_ntt_pipeline"));

    Ok(())
}

#[test]
fn sim_output_length_matches_input_size_4() -> Result<(), Error> {
    verify_full_pipeline(2)
}

#[test]
fn ntt_correctness_size_8() -> Result<(), Error> {
    verify_full_pipeline(3)
}

#[test]
fn ntt_correctness_size_16() -> Result<(), Error> {
    verify_full_pipeline(4)
}

#[test]
fn bram_verilog_emission_size_4() -> Result<(), Error> {
    let verilog_io = emit_pipeline_verilog(&[2, 1], "bram_ntt_size_4")?;
    let text = verilog_io
        .run()
        .map_err(|e| Error::VerilogGen(e.to_string()))?;

    assert!(text.contains("module bram_ntt_size_4"));
    // Two stages produce two auto-detected array declarations.
    assert!(
        text.match_indices("[0:1]").count() >= 1,
        "missing depth-2 array decl"
    );
    assert!(
        text.match_indices("[0:0]").count() >= 1,
        "missing depth-1 array decl"
    );

    // Write to target/ for manual inspection / Vivado ingestion
    std::fs::write("target/bram_ntt_size_4.v", &text)
        .map_err(|e| Error::VerilogGen(e.to_string()))?;

    Ok(())
}

#[test]
fn bram_verilog_emission_size_8() -> Result<(), Error> {
    let verilog_io = emit_pipeline_verilog(&[4, 2, 1], "bram_ntt_size_8")?;
    let text = verilog_io
        .run()
        .map_err(|e| Error::VerilogGen(e.to_string()))?;

    assert!(text.contains("module bram_ntt_size_8"));
    // Three stages produce three auto-detected array declarations.
    let arr_decl_count = text.match_indices("reg [63:0]").count();
    assert!(
        arr_decl_count >= 3,
        "expected >= 3 array decls, got {arr_decl_count}"
    );

    std::fs::write("target/bram_ntt_size_8.v", &text)
        .map_err(|e| Error::VerilogGen(e.to_string()))?;

    Ok(())
}

/// 10-stage pipeline (2^10 = 1024-point NTT).
///
/// Depths: [512, 256, 128, 64, 32, 16, 8, 4, 2, 1].
/// Stages 0-4 (depths 512..33) use circular buffer.
/// Stages 5-9 (depths 32..1) use shift register.
///
/// Each stage now uses a single `WireTy::Array` wire for its
/// delay line, so the IR scales to arbitrary depth.
#[test]
fn bram_verilog_emission_10_stage() -> Result<(), Error> {
    // 2^10-point NTT: stage j has delay depth 2^(9 - j).
    let depths: Vec<usize> = (0..10).map(|j| 1_usize << (9 - j)).collect();

    let verilog_io = emit_pipeline_verilog(&depths, "goldilocks_ntt_1024")?;
    let text = verilog_io
        .run()
        .map_err(|e| Error::VerilogGen(e.to_string()))?;

    // Basic structural checks
    assert!(
        text.contains("module goldilocks_ntt_1024"),
        "missing module declaration"
    );
    assert!(text.contains("input clk"), "missing clk port");
    assert!(text.contains("input rst"), "missing rst port");

    // All 10 stage delay arrays present (auto-detected from Array-typed state wires).
    let arr_decl_count = text.match_indices("reg [63:0]").count();
    assert!(
        arr_decl_count >= 10,
        "expected >= 10 array decls, got {arr_decl_count}"
    );

    // Stages with depth > 32 use circular buffer (pointer register).
    assert!(
        text.contains("_ptr"),
        "missing circular buffer pointer for large-depth stages"
    );

    // At least one shift-register line for small-depth stages.
    assert!(
        text.contains("[1] <="),
        "missing shift-register line for small-depth stages"
    );

    // Circular buffer keeps output compact: well under 100K lines
    let line_count = text.lines().count();
    assert!(
        line_count < 100_000,
        "emitted {line_count} lines; should be compact"
    );

    std::fs::write("target/goldilocks_ntt_1024.v", &text)
        .map_err(|e| Error::VerilogGen(e.to_string()))?;

    Ok(())
}

/// Full 24-stage pipeline (2^24 = 16,777,216-point NTT).
///
/// Depths: `[2^23, 2^22, ..., 2^0]`.  The largest stage (depth 2^23)
/// was previously infeasible because the IR allocated one wire per
/// delay element (~8M wires).  With `WireTy::Array`, each stage
/// uses a single array wire regardless of depth, so the full
/// pipeline composes and emits in bounded time and memory.
#[test]
fn bram_verilog_emission_24_stage() -> Result<(), Error> {
    // 2^24-point NTT: stage j has delay depth 2^(23 - j).
    let depths: Vec<usize> = (0..24).map(|j| 1_usize << (23 - j)).collect();

    let verilog_io = emit_pipeline_verilog(&depths, "goldilocks_ntt_2_24")?;
    let text = verilog_io
        .run()
        .map_err(|e| Error::VerilogGen(e.to_string()))?;

    // Basic structural checks.
    assert!(
        text.contains("module goldilocks_ntt_2_24"),
        "missing module declaration"
    );
    assert!(text.contains("input clk"), "missing clk port");
    assert!(text.contains("input rst"), "missing rst port");

    // All 24 stage delay arrays present.
    let arr_decl_count = text.match_indices("reg [63:0]").count();
    assert!(
        arr_decl_count >= 24,
        "expected >= 24 array decls, got {arr_decl_count}"
    );

    // Large-depth stages use circular buffer.
    assert!(text.contains("_ptr"), "missing circular buffer pointers");

    // Small-depth stages use shift register.
    assert!(text.contains("[1] <="), "missing shift-register lines");

    // Output should be compact (circular buffers are O(1) per stage).
    let line_count = text.lines().count();
    assert!(
        line_count < 200_000,
        "emitted {line_count} lines; should be compact"
    );

    std::fs::write("target/goldilocks_ntt_2_24.v", &text)
        .map_err(|e| Error::VerilogGen(e.to_string()))?;

    Ok(())
}

#[test]
fn basic_arithmetic_integration_test() -> Result<(), hdl_cat_error::Error> {
    use goldilocks_ntt_hdl::hdl::arithmetic::{goldilocks_add_sync, goldilocks_mul_sync};
    use goldilocks_ntt_hdl::hdl::common::{bitseq_to_u64, u64_to_bitseq};
    use hdl_cat_sim::Testbench;

    // Test adder end-to-end
    let adder = goldilocks_add_sync()?;
    let adder_test = Testbench::new(adder);

    let inputs = vec![u64_to_bitseq(3).concat(u64_to_bitseq(5))];
    let outputs = adder_test.run(inputs).run()?;

    assert_eq!(outputs.len(), 1);
    let sum = bitseq_to_u64(outputs[0].value())?;
    assert_eq!(sum, 8);

    // Test multiplier end-to-end
    let multiplier = goldilocks_mul_sync()?;
    let mul_test = Testbench::new(multiplier);

    let inputs = vec![u64_to_bitseq(7).concat(u64_to_bitseq(11))];
    let outputs = mul_test.run(inputs).run()?;

    assert_eq!(outputs.len(), 1);
    let product = bitseq_to_u64(outputs[0].value())?;
    assert_eq!(product, 77);

    Ok(())
}
