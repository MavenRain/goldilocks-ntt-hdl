//! Goldilocks modular subtractor HDL module.
//!
//! Computes `(a - b) mod p` where `p` is the Goldilocks prime.
//! Registered output with 1-cycle latency.

use rust_hdl::prelude::*;

use crate::field::element::GoldilocksElement;
use crate::hdl::common::{bits_to_u64, u64_to_bits, GOLDILOCKS_WIDTH};

/// Modular subtractor over the Goldilocks field.
#[derive(Clone, Debug, Default, LogicBlock)]
pub struct GoldilocksSub {
    /// Clock input.
    pub clock: Signal<In, Clock>,
    /// Minuend.
    pub a: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Subtrahend.
    pub b: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Input valid strobe.
    pub valid_in: Signal<In, Bit>,
    /// Modular difference output (1-cycle latency).
    pub diff: Signal<Out, Bits<GOLDILOCKS_WIDTH>>,
    /// Output valid strobe.
    pub valid_out: Signal<Out, Bit>,
    diff_reg: DFF<Bits<GOLDILOCKS_WIDTH>>,
    valid_reg: DFF<Bit>,
}

impl Logic for GoldilocksSub {
    fn update(&mut self) {
        self.diff_reg.clock.next = self.clock.val();
        self.valid_reg.clock.next = self.clock.val();

        let a_elem = GoldilocksElement::new(bits_to_u64(self.a.val()));
        let b_elem = GoldilocksElement::new(bits_to_u64(self.b.val()));
        self.diff_reg.d.next = u64_to_bits((a_elem - b_elem).value());
        self.valid_reg.d.next = self.valid_in.val();

        self.diff.next = self.diff_reg.q.val();
        self.valid_out.next = self.valid_reg.q.val();
    }
}
