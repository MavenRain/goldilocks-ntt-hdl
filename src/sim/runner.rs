//! Behavioral simulation runner for the NTT pipeline.
//!
//! Simulates the SDF pipeline algorithm using the same stage
//! parameters (delay depths, twiddle roots) as the HDL modules,
//! but computes results using the software [`GoldilocksElement`]
//! arithmetic.  This validates the pipeline structure without
//! depending on `RustHDL`'s signal commit protocol.
//!
//! All side effects are wrapped in [`Io::suspend`].

use comp_cat_rs::effect::io::Io;

use crate::error::Error;
use crate::field::element::GoldilocksElement;
use crate::field::roots::primitive_root_of_unity;
use crate::graph::ntt_graph::NTT_STAGES;

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

/// Build an `Io` that simulates the SDF pipeline behaviorally.
///
/// Feeds one element per cycle through `num_stages` SDF stages,
/// each with the correct delay depth and twiddle root.
/// Returns the collected output elements.
///
/// Nothing executes until [`Io::run`](comp_cat_rs::effect::io::Io::run)
/// is called at the boundary.
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
/// // Zero stages = passthrough
/// let config = SimConfig::new(input.clone(), 0).ok();
/// let result = config.map(|c| simulate_pipeline(c).run());
/// assert!(result.is_some());
/// ```
#[must_use]
pub fn simulate_pipeline(config: SimConfig) -> Io<Error, Vec<GoldilocksElement>> {
    Io::suspend(move || run_behavioral_sim(&config))
}

/// Behavioral SDF stage state.
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

/// Run the behavioral simulation.
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
}
