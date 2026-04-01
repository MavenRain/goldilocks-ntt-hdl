//! Single SDF (Single-path Delay Feedback) stage.
//!
//! Each stage alternates between two phases:
//! - **Fill phase**: incoming data fills the delay line and passes through.
//! - **Butterfly phase**: incoming data pairs with delayed data through
//!   the DIF butterfly.

use rust_hdl::prelude::*;

use crate::field::element::GoldilocksElement;
use crate::hdl::butterfly::DifButterfly;
use crate::hdl::common::{bits32_to_u64, u64_to_bits, GOLDILOCKS_WIDTH};
use crate::hdl::delay::DelayLine;
use crate::hdl::twiddle::TwiddleAccumulator;

/// A single SDF stage of the NTT pipeline.
#[derive(Clone, Debug)]
pub struct SdfStage {
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
    delay: DelayLine,
    butterfly: DifButterfly,
    twiddle_gen: TwiddleAccumulator,
    counter: DFF<Bits<32>>,
    phase: DFF<Bit>,
    delay_depth: u64,
    step_root: u64,
}

impl SdfStage {
    /// Create a new SDF stage.
    ///
    /// - `delay_depth`: delay line depth (2^(23-k) for stage k)
    /// - `step_root`: twiddle accumulator step root
    #[must_use]
    pub fn new(delay_depth: usize, step_root: GoldilocksElement) -> Self {
        Self {
            clock: Signal::default(),
            data_in: Signal::default(),
            valid_in: Signal::default(),
            data_out: Signal::default(),
            valid_out: Signal::default(),
            delay: DelayLine::new(delay_depth),
            butterfly: DifButterfly::default(),
            twiddle_gen: TwiddleAccumulator::default(),
            counter: DFF::default(),
            phase: DFF::default(),
            delay_depth: u64::try_from(delay_depth).unwrap_or(1),
            step_root: step_root.value(),
        }
    }
}

impl Default for SdfStage {
    fn default() -> Self {
        Self::new(1, GoldilocksElement::ONE)
    }
}

impl Logic for SdfStage {
    fn update(&mut self) {
        // Clock sub-blocks
        self.delay.clock.next = self.clock.val();
        self.butterfly.clock.next = self.clock.val();
        self.twiddle_gen.clock.next = self.clock.val();
        self.counter.clock.next = self.clock.val();
        self.phase.clock.next = self.clock.val();

        // Step root for twiddle generator
        self.twiddle_gen.step_root.next = u64_to_bits(self.step_root);

        // Default outputs
        self.data_out.next = bits(0_u64);
        self.valid_out.next = false;

        // Default: hold counter and phase
        self.counter.d.next = self.counter.q.val();
        self.phase.d.next = self.phase.q.val();
        self.twiddle_gen.active.next = false;
        self.twiddle_gen.reset_twiddle.next = false;

        // Delay line always receives incoming data
        self.delay.data_in.next = self.data_in.val();
        self.delay.write_enable.next = self.valid_in.val();

        // Butterfly defaults
        self.butterfly.a.next = bits(0_u64);
        self.butterfly.b.next = bits(0_u64);
        self.butterfly.twiddle.next = self.twiddle_gen.twiddle_out.val();
        self.butterfly.valid_in.next = false;

        // Extract counter as u64 for comparison
        let counter_val = bits32_to_u64(self.counter.q.val());

        if self.valid_in.val() {
            if self.phase.q.val() {
                // BUTTERFLY PHASE
                self.butterfly.a.next = self.delay.data_out.val();
                self.butterfly.b.next = self.data_in.val();
                self.butterfly.valid_in.next = true;
                self.twiddle_gen.active.next = true;

                self.data_out.next = self.butterfly.upper.val();
                self.valid_out.next = self.butterfly.valid_out.val();

                let next_count = counter_val + 1;
                self.counter.d.next = bits(next_count);

                if next_count >= self.delay_depth {
                    self.phase.d.next = false;
                    self.counter.d.next = bits(0_u64);
                }
            } else {
                // FILL PHASE
                self.data_out.next = self.data_in.val();
                self.valid_out.next = true;

                let next_count = counter_val + 1;
                self.counter.d.next = bits(next_count);

                if next_count >= self.delay_depth {
                    self.phase.d.next = true;
                    self.counter.d.next = bits(0_u64);
                    self.twiddle_gen.reset_twiddle.next = true;
                }
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

impl Block for SdfStage {
    fn connect_all(&mut self) {
        self.delay.connect_all();
        self.butterfly.connect_all();
        self.twiddle_gen.connect_all();
        self.counter.connect_all();
        self.phase.connect_all();
        self.connect();
    }

    fn update_all(&mut self) {
        self.update();
        self.delay.update_all();
        self.butterfly.update_all();
        self.twiddle_gen.update_all();
        self.counter.update_all();
        self.phase.update_all();
    }

    fn has_changed(&self) -> bool {
        self.data_out.changed()
            || self.valid_out.changed()
            || self.delay.has_changed()
            || self.butterfly.has_changed()
            || self.twiddle_gen.has_changed()
            || self.counter.has_changed()
            || self.phase.has_changed()
    }

    fn accept(&self, name: &str, probe: &mut dyn Probe) {
        probe.visit_start_scope(name, self);
        probe.visit_atom("data_out", &self.data_out);
        probe.visit_atom("valid_out", &self.valid_out);
        self.delay.accept("delay", probe);
        self.butterfly.accept("butterfly", probe);
        probe.visit_end_scope(name, self);
    }
}
