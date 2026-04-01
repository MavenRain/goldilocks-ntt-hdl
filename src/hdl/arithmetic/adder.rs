//! Goldilocks modular adder HDL module.
//!
//! Computes `(a + b) mod p` where `p` is the Goldilocks prime.
//! Registered output with 1-cycle latency.

use rust_hdl::prelude::*;

use crate::field::element::GoldilocksElement;
use crate::hdl::common::{bits_to_u64, u64_to_bits, GOLDILOCKS_WIDTH};

/// Modular adder over the Goldilocks field.
#[derive(Clone, Debug, Default, LogicBlock)]
pub struct GoldilocksAdder {
    /// Clock input.
    pub clock: Signal<In, Clock>,
    /// First operand.
    pub a: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Second operand.
    pub b: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Input valid strobe.
    pub valid_in: Signal<In, Bit>,
    /// Modular sum output (1-cycle latency).
    pub sum: Signal<Out, Bits<GOLDILOCKS_WIDTH>>,
    /// Output valid strobe.
    pub valid_out: Signal<Out, Bit>,
    sum_reg: DFF<Bits<GOLDILOCKS_WIDTH>>,
    valid_reg: DFF<Bit>,
}

impl Logic for GoldilocksAdder {
    fn update(&mut self) {
        self.sum_reg.clock.next = self.clock.val();
        self.valid_reg.clock.next = self.clock.val();

        let a_elem = GoldilocksElement::new(bits_to_u64(self.a.val()));
        let b_elem = GoldilocksElement::new(bits_to_u64(self.b.val()));
        self.sum_reg.d.next = u64_to_bits((a_elem + b_elem).value());
        self.valid_reg.d.next = self.valid_in.val();

        self.sum.next = self.sum_reg.q.val();
        self.valid_out.next = self.valid_reg.q.val();
    }
}
