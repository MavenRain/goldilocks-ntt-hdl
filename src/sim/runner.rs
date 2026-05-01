//! Cycle-accurate simulation runner for the NTT pipeline.
//!
//! Drives [`compose_pipeline`] through hdl-cat's [`Testbench`] to
//! produce a cycle-accurate simulation of the hardware SDF pipeline.
//! Works for any stage count from 0 to [`NTT_STAGES`].
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
use crate::hdl::pipeline::compose_pipeline;

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
    pub fn new(input: Vec<GoldilocksElement>, num_stages: usize) -> Result<Self, Error> {
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
/// Creates the appropriate pipeline via [`compose_pipeline`] and drives
/// it with the input data through hdl-cat's cycle-accurate simulation.
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

/// Compute delay depths for an `num_stages`-stage pipeline.
///
/// Stage `j` has delay depth `2^(num_stages - 1 - j)`.
fn stage_depths(num_stages: usize) -> Vec<usize> {
    (0..num_stages)
        .map(|j| 1_usize << (num_stages - 1 - j))
        .collect()
}

/// Compute step roots for each stage.
///
/// Stage `j` of a `num_stages`-stage pipeline uses the primitive
/// `2^(num_stages - j)`-th root of unity as its step root.
fn step_roots(num_stages: usize) -> Result<Vec<GoldilocksElement>, Error> {
    (0..num_stages)
        .map(|j| {
            let order_bits =
                u32::try_from(num_stages - j).map_err(|e| Error::Field(e.to_string()))?;
            primitive_root_of_unity(order_bits)
        })
        .collect()
}

/// Pack one cycle's input into a [`BitSeq`].
///
/// Wire layout: `data_in:64 ++ valid_in:1 ++ step_root_0:64 ++ ... ++ step_root_{N-1}:64`.
fn pack_input(data: GoldilocksElement, roots: &[GoldilocksElement]) -> BitSeq {
    let base = u64_to_bitseq(data.value()).concat(BitSeq::from_vec(vec![true]));

    roots
        .iter()
        .fold(base, |acc, root| acc.concat(u64_to_bitseq(root.value())))
}

/// Extract `(data, valid)` from a pipeline output [`BitSeq`].
fn unpack_output(bits: &BitSeq) -> Result<(GoldilocksElement, bool), Error> {
    if bits.len() < 65 {
        Err(Error::Field(format!(
            "output too short: expected >= 65 bits, got {}",
            bits.len(),
        )))
    } else {
        let (data_bits, rest) = bits.clone().split_at(64);
        let (valid_bits, _) = rest.split_at(1);
        let data = bitseq_to_u64(&data_bits)?;
        let valid = valid_bits.bit(0);
        Ok((GoldilocksElement::new(data), valid))
    }
}

/// Run the hdl-cat simulation.
fn run_hdl_cat_sim(config: &SimConfig) -> Result<Vec<GoldilocksElement>, Error> {
    if config.num_stages == 0 {
        return Ok(config.input.clone());
    }

    let depths = stage_depths(config.num_stages);
    let roots = step_roots(config.num_stages)?;

    let pipeline = compose_pipeline(&depths)
        .map_err(|e| Error::Field(format!("pipeline construction failed: {e}")))?;

    let inputs: Vec<BitSeq> = config
        .input
        .iter()
        .map(|elem| pack_input(*elem, &roots))
        .collect();

    let testbench = Testbench::new(pipeline);
    let outputs = testbench
        .run(inputs)
        .run()
        .map_err(|e| Error::Field(format!("HDL simulation failed: {e}")))?;

    outputs
        .iter()
        .map(|sample| {
            let (data, valid) = unpack_output(sample.value())?;
            if valid {
                Ok(data)
            } else {
                Ok(GoldilocksElement::ZERO)
            }
        })
        .collect()
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
    fn simulate_config_rejects_too_many_stages() {
        assert!(SimConfig::new(vec![], 25).is_err());
    }

    #[test]
    fn hdl_cat_size_4_simulation_works() -> Result<(), Error> {
        let input = vec![
            GoldilocksElement::new(1),
            GoldilocksElement::new(2),
            GoldilocksElement::new(3),
            GoldilocksElement::new(4),
        ];

        let config = SimConfig::new(input, 2)?;
        let result = simulate_pipeline(config).run()?;
        assert!(!result.is_empty());
        Ok(())
    }

    #[test]
    fn hdl_cat_size_16_simulation_works() -> Result<(), Error> {
        let input: Vec<GoldilocksElement> = (1..=16).map(GoldilocksElement::new).collect();

        let config = SimConfig::new(input, 4)?;
        let result = simulate_pipeline(config).run()?;
        assert_eq!(result.len(), 16);
        Ok(())
    }
}
