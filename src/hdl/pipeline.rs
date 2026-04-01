//! Full 24-stage NTT pipeline.
//!
//! Composes 24 [`SdfStage`] instances into a single streaming
//! pipeline.  Data enters at stage 0 and exits after stage 23.
//! Each stage has its own delay depth and twiddle root, computed
//! from the [`HdlInterpretation`](crate::interpret::hdl_morphism::HdlInterpretation).

use rust_hdl::prelude::*;

use crate::error::Error;
use crate::hdl::common::{u64_to_bits, GOLDILOCKS_WIDTH};
use crate::hdl::stage::SdfStage;
use crate::field::roots::primitive_root_of_unity;
use crate::graph::ntt_graph::NTT_STAGES;

/// The full 24-stage SDF NTT pipeline.
///
/// Constructed via [`NttPipeline::build`], which computes the
/// correct delay depth and twiddle root for each stage.
#[derive(Clone, Debug, Default)]
pub struct NttPipeline {
    /// Clock input.
    pub clock: Signal<In, Clock>,
    /// Streaming data input.
    pub data_in: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Input valid strobe.
    pub valid_in: Signal<In, Bit>,
    /// Streaming data output.
    pub data_out: Signal<Out, Bits<GOLDILOCKS_WIDTH>>,
    /// Output valid strobe.
    pub valid_out: Signal<Out, Bit>,
    stages: Vec<SdfStage>,
}

impl NttPipeline {
    /// Build the 24-stage pipeline with correct parameters.
    ///
    /// Stage k gets delay depth 2^(23-k) and the primitive
    /// 2^(24-k)-th root of unity as its twiddle step root.
    ///
    /// # Errors
    ///
    /// Returns an error if root of unity computation fails.
    pub fn build() -> Result<Self, Error> {
        let stages = (0..NTT_STAGES)
            .map(|k| {
                let delay_depth = 1_usize << (23 - k);
                let order_bits = u32::try_from(NTT_STAGES - k)
                    .map_err(|e| Error::Field(e.to_string()))?;
                let step_root = primitive_root_of_unity(order_bits)?;
                Ok(SdfStage::new(delay_depth, step_root))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(Self {
            clock: Signal::default(),
            data_in: Signal::default(),
            valid_in: Signal::default(),
            data_out: Signal::default(),
            valid_out: Signal::default(),
            stages,
        })
    }

    /// Build a pipeline with only the first `n` stages (for testing).
    ///
    /// # Errors
    ///
    /// Returns an error if `n > 24` or root computation fails.
    pub fn build_partial(n: usize) -> Result<Self, Error> {
        if n > NTT_STAGES {
            Err(Error::Field(format!(
                "requested {n} stages but maximum is {NTT_STAGES}"
            )))
        } else {
            let stages = (0..n)
                .map(|k| {
                    let delay_depth = 1_usize << (23 - k);
                    let order_bits = u32::try_from(NTT_STAGES - k)
                        .map_err(|e| Error::Field(e.to_string()))?;
                    let step_root = primitive_root_of_unity(order_bits)?;
                    Ok(SdfStage::new(delay_depth, step_root))
                })
                .collect::<Result<Vec<_>, Error>>()?;

            Ok(Self {
                clock: Signal::default(),
                data_in: Signal::default(),
                valid_in: Signal::default(),
                data_out: Signal::default(),
                valid_out: Signal::default(),
                stages,
            })
        }
    }

    /// Number of stages in this pipeline.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl Logic for NttPipeline {
    fn update(&mut self) {
        // Chain stages: output of stage k feeds input of stage k+1.
        // Stage 0 reads from pipeline input.
        // Last stage drives pipeline output.
        let n = self.stages.len();

        if n == 0 {
            // Pass-through when no stages
            self.data_out.next = self.data_in.val();
            self.valid_out.next = self.valid_in.val();
        } else {
            // Feed first stage from pipeline input
            if let Some(first) = self.stages.first_mut() {
                first.clock.next = self.clock.val();
                first.data_in.next = self.data_in.val();
                first.valid_in.next = self.valid_in.val();
            }

            // Chain intermediate stages
            // We need to read output of stage k and feed to stage k+1.
            // Collect intermediate values first to satisfy borrow checker.
            let intermediates: Vec<_> = self.stages.iter()
                .map(|s| (s.data_out.val(), s.valid_out.val()))
                .collect();

            (1..n).for_each(|k| {
                if let (Some(stage), Some((data, valid))) =
                    (self.stages.get_mut(k), intermediates.get(k - 1))
                {
                    stage.clock.next = self.clock.val();
                    stage.data_in.next = *data;
                    stage.valid_in.next = *valid;
                }
            });

            // Drive pipeline output from last stage
            if let Some((data, valid)) = intermediates.last() {
                self.data_out.next = *data;
                self.valid_out.next = *valid;
            } else {
                self.data_out.next = u64_to_bits(0);
                self.valid_out.next = false;
            }
        }
    }

    fn connect(&mut self) {
        self.data_out.connect();
        self.valid_out.connect();
    }

    fn hdl(&self) -> Verilog {
        Verilog::Empty
    }
}

impl Block for NttPipeline {
    fn connect_all(&mut self) {
        self.stages.iter_mut().for_each(SdfStage::connect_all);
        self.connect();
    }

    fn update_all(&mut self) {
        self.update();
        self.stages.iter_mut().for_each(SdfStage::update_all);
    }

    fn has_changed(&self) -> bool {
        self.data_out.changed()
            || self.valid_out.changed()
            || self.stages.iter().any(SdfStage::has_changed)
    }

    fn accept(&self, name: &str, probe: &mut dyn Probe) {
        probe.visit_start_scope(name, self);
        probe.visit_atom("data_out", &self.data_out);
        probe.visit_atom("valid_out", &self.valid_out);
        self.stages.iter().enumerate().for_each(|(i, s)| {
            let stage_name = format!("stage_{i}");
            s.accept(&stage_name, probe);
        });
        probe.visit_end_scope(name, self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_full_pipeline() -> Result<(), Error> {
        let pipeline = NttPipeline::build()?;
        assert_eq!(pipeline.stage_count(), 24);
        Ok(())
    }

    #[test]
    fn build_partial_pipeline() -> Result<(), Error> {
        let pipeline = NttPipeline::build_partial(4)?;
        assert_eq!(pipeline.stage_count(), 4);
        Ok(())
    }

    #[test]
    fn build_zero_stages() -> Result<(), Error> {
        let pipeline = NttPipeline::build_partial(0)?;
        assert_eq!(pipeline.stage_count(), 0);
        Ok(())
    }

    #[test]
    fn build_too_many_stages_is_error() {
        assert!(NttPipeline::build_partial(25).is_err());
    }
}
