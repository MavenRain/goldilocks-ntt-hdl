//! On-the-fly twiddle factor generator.
//!
//! Maintains a running accumulator that multiplies by a fixed step
//! root each active cycle, eliminating the need for twiddle ROM.

use rust_hdl::prelude::*;

use crate::field::element::GoldilocksElement;
use crate::hdl::common::{bits_to_u64, u64_to_bits, GOLDILOCKS_WIDTH};

/// On-the-fly twiddle factor accumulator.
///
/// When `active` is asserted, multiplies the current twiddle by the
/// step root.  When `reset_twiddle` is asserted, resets to 1.
#[derive(Clone, Debug, Default, LogicBlock)]
pub struct TwiddleAccumulator {
    /// Clock input.
    pub clock: Signal<In, Clock>,
    /// The step root for this stage.
    pub step_root: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Assert to advance the accumulator.
    pub active: Signal<In, Bit>,
    /// Assert to reset the accumulator to 1.
    pub reset_twiddle: Signal<In, Bit>,
    /// Current twiddle factor output.
    pub twiddle_out: Signal<Out, Bits<GOLDILOCKS_WIDTH>>,
    accum: DFF<Bits<GOLDILOCKS_WIDTH>>,
}

impl Logic for TwiddleAccumulator {
    fn update(&mut self) {
        self.accum.clock.next = self.clock.val();
        self.twiddle_out.next = self.accum.q.val();

        // Default: hold
        self.accum.d.next = self.accum.q.val();

        if self.reset_twiddle.val() {
            self.accum.d.next = u64_to_bits(1);
        } else if self.active.val() {
            let current = GoldilocksElement::new(bits_to_u64(self.accum.q.val()));
            let step = GoldilocksElement::new(bits_to_u64(self.step_root.val()));
            self.accum.d.next = u64_to_bits((current * step).value());
        }
    }
}
