//! DIF butterfly HDL module.
//!
//! Computes:
//!
//! ```text
//! upper = a + b
//! lower = (a - b) * twiddle
//! ```
//!
//! Total latency: 8 cycles (1 for add/sub + 7 for multiply).

use rust_hdl::prelude::*;

use crate::hdl::arithmetic::adder::GoldilocksAdder;
use crate::hdl::arithmetic::multiplier::GoldilocksMul;
use crate::hdl::arithmetic::subtractor::GoldilocksSub;
use crate::hdl::common::GOLDILOCKS_WIDTH;

/// Total butterfly latency: add/sub (1) + multiply (7) = 8 cycles.
pub const BUTTERFLY_LATENCY: usize = 8;

/// DIF butterfly unit.
///
/// Upper path is delayed to align with the lower path's multiplier.
#[derive(Clone, Debug, Default, LogicBlock)]
pub struct DifButterfly {
    /// Clock input.
    pub clock: Signal<In, Clock>,
    /// First input element.
    pub a: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Second input element.
    pub b: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Twiddle factor for the lower path.
    pub twiddle: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Input valid strobe.
    pub valid_in: Signal<In, Bit>,
    /// Upper output: `a + b`.
    pub upper: Signal<Out, Bits<GOLDILOCKS_WIDTH>>,
    /// Lower output: `(a - b) * twiddle`.
    pub lower: Signal<Out, Bits<GOLDILOCKS_WIDTH>>,
    /// Output valid strobe.
    pub valid_out: Signal<Out, Bit>,
    adder: GoldilocksAdder,
    subtractor: GoldilocksSub,
    multiplier: GoldilocksMul,
    // 7-stage delay to align adder output with multiplier output
    upper_d0: DFF<Bits<GOLDILOCKS_WIDTH>>,
    upper_d1: DFF<Bits<GOLDILOCKS_WIDTH>>,
    upper_d2: DFF<Bits<GOLDILOCKS_WIDTH>>,
    upper_d3: DFF<Bits<GOLDILOCKS_WIDTH>>,
    upper_d4: DFF<Bits<GOLDILOCKS_WIDTH>>,
    upper_d5: DFF<Bits<GOLDILOCKS_WIDTH>>,
    upper_d6: DFF<Bits<GOLDILOCKS_WIDTH>>,
}

impl Logic for DifButterfly {
    fn update(&mut self) {
        // Clock sub-blocks and delay registers
        self.adder.clock.next = self.clock.val();
        self.subtractor.clock.next = self.clock.val();
        self.multiplier.clock.next = self.clock.val();
        self.upper_d0.clock.next = self.clock.val();
        self.upper_d1.clock.next = self.clock.val();
        self.upper_d2.clock.next = self.clock.val();
        self.upper_d3.clock.next = self.clock.val();
        self.upper_d4.clock.next = self.clock.val();
        self.upper_d5.clock.next = self.clock.val();
        self.upper_d6.clock.next = self.clock.val();

        // Adder and subtractor in parallel
        self.adder.a.next = self.a.val();
        self.adder.b.next = self.b.val();
        self.adder.valid_in.next = self.valid_in.val();

        self.subtractor.a.next = self.a.val();
        self.subtractor.b.next = self.b.val();
        self.subtractor.valid_in.next = self.valid_in.val();

        // Subtractor output feeds multiplier
        self.multiplier.a.next = self.subtractor.diff.val();
        self.multiplier.b.next = self.twiddle.val();
        self.multiplier.valid_in.next = self.subtractor.valid_out.val();

        // Delay adder output by 7 cycles to match multiplier latency
        self.upper_d0.d.next = self.adder.sum.val();
        self.upper_d1.d.next = self.upper_d0.q.val();
        self.upper_d2.d.next = self.upper_d1.q.val();
        self.upper_d3.d.next = self.upper_d2.q.val();
        self.upper_d4.d.next = self.upper_d3.q.val();
        self.upper_d5.d.next = self.upper_d4.q.val();
        self.upper_d6.d.next = self.upper_d5.q.val();

        // Aligned outputs
        self.upper.next = self.upper_d6.q.val();
        self.lower.next = self.multiplier.product.val();
        self.valid_out.next = self.multiplier.valid_out.val();
    }
}
