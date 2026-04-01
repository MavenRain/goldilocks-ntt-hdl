//! Goldilocks modular multiplier HDL module.
//!
//! Computes `(a * b) mod p` where `p` is the Goldilocks prime.
//! Pipelined with 7-cycle latency to model the real hardware cost.

use rust_hdl::prelude::*;

use crate::field::element::GoldilocksElement;
use crate::hdl::common::{bits_to_u64, u64_to_bits, GOLDILOCKS_WIDTH};

/// The number of pipeline stages for the multiplier.
pub const MULTIPLIER_LATENCY: usize = 7;

/// Pipelined modular multiplier over the Goldilocks field.
///
/// Models a 7-stage pipeline.  In real hardware this decomposes the
/// 64x64 multiply into partial products across DSP slices, then
/// applies Goldilocks reduction.  For simulation, computation happens
/// in stage 0 and the remaining stages model pipeline depth.
#[derive(Clone, Debug, Default, LogicBlock)]
pub struct GoldilocksMul {
    /// Clock input.
    pub clock: Signal<In, Clock>,
    /// First operand.
    pub a: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Second operand.
    pub b: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Input valid strobe.
    pub valid_in: Signal<In, Bit>,
    /// Modular product output (7-cycle latency).
    pub product: Signal<Out, Bits<GOLDILOCKS_WIDTH>>,
    /// Output valid strobe.
    pub valid_out: Signal<Out, Bit>,
    // Data pipeline (7 stages)
    pipe0: DFF<Bits<GOLDILOCKS_WIDTH>>,
    pipe1: DFF<Bits<GOLDILOCKS_WIDTH>>,
    pipe2: DFF<Bits<GOLDILOCKS_WIDTH>>,
    pipe3: DFF<Bits<GOLDILOCKS_WIDTH>>,
    pipe4: DFF<Bits<GOLDILOCKS_WIDTH>>,
    pipe5: DFF<Bits<GOLDILOCKS_WIDTH>>,
    pipe6: DFF<Bits<GOLDILOCKS_WIDTH>>,
    // Valid pipeline (7 stages)
    vpipe0: DFF<Bit>,
    vpipe1: DFF<Bit>,
    vpipe2: DFF<Bit>,
    vpipe3: DFF<Bit>,
    vpipe4: DFF<Bit>,
    vpipe5: DFF<Bit>,
    vpipe6: DFF<Bit>,
}

impl Logic for GoldilocksMul {
    fn update(&mut self) {
        // Clock all pipeline stages
        self.pipe0.clock.next = self.clock.val();
        self.pipe1.clock.next = self.clock.val();
        self.pipe2.clock.next = self.clock.val();
        self.pipe3.clock.next = self.clock.val();
        self.pipe4.clock.next = self.clock.val();
        self.pipe5.clock.next = self.clock.val();
        self.pipe6.clock.next = self.clock.val();
        self.vpipe0.clock.next = self.clock.val();
        self.vpipe1.clock.next = self.clock.val();
        self.vpipe2.clock.next = self.clock.val();
        self.vpipe3.clock.next = self.clock.val();
        self.vpipe4.clock.next = self.clock.val();
        self.vpipe5.clock.next = self.clock.val();
        self.vpipe6.clock.next = self.clock.val();

        // Stage 0: compute product using software model
        let a_elem = GoldilocksElement::new(bits_to_u64(self.a.val()));
        let b_elem = GoldilocksElement::new(bits_to_u64(self.b.val()));
        self.pipe0.d.next = u64_to_bits((a_elem * b_elem).value());
        self.vpipe0.d.next = self.valid_in.val();

        // Stages 1-6: pipeline delay
        self.pipe1.d.next = self.pipe0.q.val();
        self.pipe2.d.next = self.pipe1.q.val();
        self.pipe3.d.next = self.pipe2.q.val();
        self.pipe4.d.next = self.pipe3.q.val();
        self.pipe5.d.next = self.pipe4.q.val();
        self.pipe6.d.next = self.pipe5.q.val();
        self.vpipe1.d.next = self.vpipe0.q.val();
        self.vpipe2.d.next = self.vpipe1.q.val();
        self.vpipe3.d.next = self.vpipe2.q.val();
        self.vpipe4.d.next = self.vpipe3.q.val();
        self.vpipe5.d.next = self.vpipe4.q.val();
        self.vpipe6.d.next = self.vpipe5.q.val();

        // Output from final stage
        self.product.next = self.pipe6.q.val();
        self.valid_out.next = self.vpipe6.q.val();
    }
}
