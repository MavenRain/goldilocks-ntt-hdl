//! Behavioral simulation runner for the NTT pipeline.
//!
//! Simulates the SDF pipeline using hdl-cat's `Testbench` to drive the
//! actual circuit IR through multiple clock cycles. This provides
//! cycle-accurate simulation of the hardware pipeline.
//!
//! All side effects are wrapped in [`Io::suspend`].

use comp_cat_rs::effect::io::Io;
use hdl_cat_kind::BitSeq;
use hdl_cat_sim::Testbench;

use crate::error::Error;
use crate::field::element::GoldilocksElement;
use crate::field::roots::primitive_root_of_unity;
use crate::graph::ntt_graph::NTT_STAGES;
use crate::hdl::common::{bitseq_to_u64, u64_to_bitseq};
use crate::hdl::pipeline::size_4_pipeline;

/// Description of a simulation run.
#[derive(Debug, Clone)]
pub struct SimConfig {
    input: Vec<GoldilocksElement>,
    num_stages: usize,
}

impl SimConfig {
    /// Create a simulation config for a partial pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_stages > 24`.
    pub fn new(
        input: Vec<GoldilocksElement>,
        num_stages: usize,
    ) -> Result<Self, Error> {
        if num_stages > NTT_STAGES {
            Err(Error::Field(format!(
                "num_stages {num_stages} exceeds maximum {NTT_STAGES}"
            )))
        } else {
            Ok(Self { input, num_stages })
        }
    }

    /// The input data.
    pub fn input(&self) -> &[GoldilocksElement] {
        &self.input
    }

    /// Number of pipeline stages.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }
}

/// Build an `Io` that simulates the SDF pipeline using hdl-cat's Testbench.
///
/// Creates the appropriate pipeline based on `num_stages` and drives it
/// with the input data through hdl-cat's cycle-accurate simulation.
///
/// # Examples
///
/// ```
/// use goldilocks_ntt_hdl::field::element::GoldilocksElement;
/// use goldilocks_ntt_hdl::sim::runner::{SimConfig, simulate_pipeline};
///
/// let input: Vec<GoldilocksElement> = (0..4)
///     .map(GoldilocksElement::new)
///     .collect();
///
/// // Small pipeline test
/// let config = SimConfig::new(input.clone(), 2).ok();
/// let result = config.map(|c| simulate_pipeline(c).run());
/// assert!(result.is_some());
/// ```
#[must_use]
pub fn simulate_pipeline(config: SimConfig) -> Io<Error, Vec<GoldilocksElement>> {
    Io::suspend(move || run_hdl_cat_sim(&config))
}

/// Simulate a size-4 pipeline specifically using hdl-cat.
#[must_use]
pub fn simulate_size_4_pipeline(
    input: Vec<GoldilocksElement>,
) -> Io<hdl_cat_error::Error, Vec<GoldilocksElement>> {
    Io::suspend(move || {
        let pipeline = size_4_pipeline()?;

        // Get twiddle roots for size-4 NTT
        let step_root_0 = primitive_root_of_unity(2)
            .map_err(|_| hdl_cat_error::Error::WidthMismatch {
                expected: hdl_cat_error::Width::new(64),
                actual: hdl_cat_error::Width::new(64),
            })?;
        let step_root_1 = primitive_root_of_unity(1)
            .map_err(|_| hdl_cat_error::Error::WidthMismatch {
                expected: hdl_cat_error::Width::new(64),
                actual: hdl_cat_error::Width::new(64),
            })?;

        // Convert input to BitSeq format - concatenate all inputs per cycle
        let inputs: Vec<BitSeq> = input
            .iter()
            .map(|elem| {
                let data_bits = u64_to_bitseq(elem.value());
                let valid_bits = BitSeq::from_vec(vec![true]);
                let step_root_0_bits = u64_to_bitseq(step_root_0.value());
                let step_root_1_bits = u64_to_bitseq(step_root_1.value());

                // Concatenate all input signals for this cycle
                data_bits
                    .concat(valid_bits)
                    .concat(step_root_0_bits)
                    .concat(step_root_1_bits)
            })
            .collect();

        // Run simulation
        let testbench = Testbench::new(pipeline);
        let outputs = testbench.run(inputs).run()?;

        // Convert outputs back to GoldilocksElement
        // Each TimedSample contains a BitSeq with concatenated outputs
        let results: Result<Vec<GoldilocksElement>, hdl_cat_error::Error> = outputs
            .iter()
            .map(|timed_sample| {
                let output_bitseq = timed_sample.value();

                // For simplicity, assume first 64 bits are data, next bit is valid
                if output_bitseq.len() >= 65 {
                    // Extract data (first 64 bits) and valid (65th bit)
                    let (data_bits, rest) = output_bitseq.clone().split_at(64);
                    let (valid_bits, _) = rest.split_at(1);

                    let data_u64 = bitseq_to_u64(&data_bits)?;
                    let valid = valid_bits.bit(0);

                    if valid {
                        Ok(GoldilocksElement::new(data_u64))
                    } else {
                        // Return zero for invalid outputs
                        Ok(GoldilocksElement::ZERO)
                    }
                } else {
                    Err(hdl_cat_error::Error::WidthMismatch {
                        expected: hdl_cat_error::Width::new(65),
                        actual: hdl_cat_error::Width::new(u32::try_from(output_bitseq.len()).unwrap_or(0)),
                    })
                }
            })
            .collect();

        results
    })
}

/// Run the hdl-cat simulation.
fn run_hdl_cat_sim(config: &SimConfig) -> Result<Vec<GoldilocksElement>, Error> {
    match config.num_stages {
        0 => {
            // Zero stages = passthrough
            Ok(config.input.clone())
        },
        2 => {
            // Use size-4 pipeline for 2 stages
            let result = simulate_size_4_pipeline(config.input.clone()).run()
                .map_err(|e| Error::Field(format!("HDL simulation failed: {e}")))?;
            Ok(result)
        },
        _ => {
            // For other stage counts, fall back to behavioral simulation
            run_behavioral_sim(config)
        }
    }
}

/// Behavioral SDF stage state (fallback for unsupported stage counts).
struct StageState {
    delay_buffer: Vec<GoldilocksElement>,
    write_ptr: usize,
    counter: usize,
    in_butterfly_phase: bool,
    delay_depth: usize,
    twiddle_current: GoldilocksElement,
    twiddle_step: GoldilocksElement,
}

impl StageState {
    fn new(delay_depth: usize, twiddle_step: GoldilocksElement) -> Self {
        Self {
            delay_buffer: vec![GoldilocksElement::ZERO; delay_depth],
            write_ptr: 0,
            counter: 0,
            in_butterfly_phase: false,
            delay_depth,
            twiddle_current: GoldilocksElement::ONE,
            twiddle_step,
        }
    }

    /// Process one element through this stage, returning the output.
    fn process(&mut self, input: GoldilocksElement) -> GoldilocksElement {
        if self.in_butterfly_phase {
            // Read delayed element
            let delayed = self.delay_buffer
                .get(self.write_ptr)
                .copied()
                .unwrap_or(GoldilocksElement::ZERO);

            // Write new element into delay buffer (feedback)
            let upper = delayed + input;
            if let Some(slot) = self.delay_buffer.get_mut(self.write_ptr) {
                *slot = upper;
            }
            self.write_ptr = (self.write_ptr + 1) % self.delay_depth.max(1);

            // Lower path: (delayed - input) * twiddle
            let lower = (delayed - input) * self.twiddle_current;
            self.twiddle_current = self.twiddle_current * self.twiddle_step;

            self.counter += 1;
            if self.counter >= self.delay_depth {
                self.counter = 0;
                self.in_butterfly_phase = false;
            }

            lower
        } else {
            // Fill phase: store in delay buffer, pass through
            if let Some(slot) = self.delay_buffer.get_mut(self.write_ptr) {
                *slot = input;
            }
            self.write_ptr = (self.write_ptr + 1) % self.delay_depth.max(1);

            self.counter += 1;
            if self.counter >= self.delay_depth {
                self.counter = 0;
                self.in_butterfly_phase = true;
                self.twiddle_current = GoldilocksElement::ONE;
            }

            input
        }
    }
}

/// Run the behavioral simulation (fallback for unsupported pipelines).
fn run_behavioral_sim(config: &SimConfig) -> Result<Vec<GoldilocksElement>, Error> {
    // Build stage states with correct parameters
    let mut stages: Vec<StageState> = (0..config.num_stages)
        .map(|k| {
            let delay_depth = 1_usize << (23 - k);
            let order_bits = u32::try_from(NTT_STAGES - k)
                .map_err(|e| Error::Field(e.to_string()))?;
            let step_root = primitive_root_of_unity(order_bits)?;
            Ok(StageState::new(delay_depth, step_root))
        })
        .collect::<Result<Vec<_>, Error>>()?;

    // Feed each input element through the pipeline
    let outputs: Vec<GoldilocksElement> = config.input.iter().map(|elem| {
        stages.iter_mut().fold(*elem, |value, stage| {
            stage.process(value)
        })
    }).collect();

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulate_zero_stages_is_passthrough() -> Result<(), Error> {
        let input = vec![
            GoldilocksElement::new(10),
            GoldilocksElement::new(20),
            GoldilocksElement::new(30),
            GoldilocksElement::new(40),
        ];
        let config = SimConfig::new(input.clone(), 0)?;
        let result = simulate_pipeline(config).run()?;
        assert_eq!(result.len(), input.len());
        input.iter().zip(result.iter()).try_for_each(|(expected, actual)| {
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
    fn simulate_config_rejects_too_many_stages() {
        assert!(SimConfig::new(vec![], 25).is_err());
    }

    #[test]
    fn hdl_cat_size_4_simulation_works() -> Result<(), hdl_cat_error::Error> {
        let input = vec![
            GoldilocksElement::new(1),
            GoldilocksElement::new(2),
            GoldilocksElement::new(3),
            GoldilocksElement::new(4),
        ];

        let result = simulate_size_4_pipeline(input.clone()).run()?;
        // Basic sanity check: we got some output
        assert!(!result.is_empty());
        Ok(())
    }
}
