//! Parameterized delay line HDL module.
//!
//! Implements a FIFO-style delay of a configurable number of cycles.
//! For simulation, uses a `Vec`-backed circular buffer.
//! For synthesis, this would map to BRAM, URAM, or shift registers
//! depending on the depth.

use rust_hdl::prelude::*;

use crate::hdl::common::{bits_to_u64, u64_to_bits, GOLDILOCKS_WIDTH};

/// Delay line with configurable depth.
///
/// Delays the input signal by `depth` clock cycles.
/// Uses a circular buffer for simulation efficiency.
///
/// Note: this module does not use `#[hdl_gen]` because it manages
/// its own internal buffer state.  For Verilog generation, a
/// separate RAM-based implementation would be needed.
#[derive(Clone, Debug)]
pub struct DelayLine {
    /// Clock input.
    pub clock: Signal<In, Clock>,
    /// Data input.
    pub data_in: Signal<In, Bits<GOLDILOCKS_WIDTH>>,
    /// Write enable (assert to shift data in).
    pub write_enable: Signal<In, Bit>,
    /// Delayed data output.
    pub data_out: Signal<Out, Bits<GOLDILOCKS_WIDTH>>,
    // Internal circular buffer
    buffer: Vec<u64>,
    write_ptr: usize,
    depth: usize,
}

impl DelayLine {
    /// Create a delay line with the given depth in cycles.
    #[must_use]
    pub fn new(depth: usize) -> Self {
        Self {
            clock: Signal::default(),
            data_in: Signal::default(),
            write_enable: Signal::default(),
            data_out: Signal::default(),
            buffer: vec![0_u64; depth.max(1)],
            write_ptr: 0,
            depth: depth.max(1),
        }
    }
}

impl Default for DelayLine {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Logic for DelayLine {
    fn update(&mut self) {
        // Read from the current write pointer position (oldest entry)
        let read_val = self.buffer.get(self.write_ptr).copied().unwrap_or(0);
        self.data_out.next = u64_to_bits(read_val);

        if self.clock.pos_edge() && self.write_enable.val() {
            let input_val = bits_to_u64(self.data_in.val());
            // Write to current position (overwrites oldest)
            if let Some(slot) = self.buffer.get_mut(self.write_ptr) {
                *slot = input_val;
            }
            // Advance write pointer circularly
            self.write_ptr = (self.write_ptr + 1) % self.depth;
        }
    }

    fn connect(&mut self) {
        self.data_out.connect();
    }

    fn hdl(&self) -> Verilog {
        // Custom Verilog would go here for synthesis
        Verilog::Empty
    }
}

impl Block for DelayLine {
    fn connect_all(&mut self) {
        self.connect();
    }

    fn update_all(&mut self) {
        self.update();
    }

    fn has_changed(&self) -> bool {
        self.data_out.changed()
    }

    fn accept(&self, name: &str, probe: &mut dyn Probe) {
        probe.visit_start_scope(name, self);
        probe.visit_atom(
            "data_out",
            &self.data_out,
        );
        probe.visit_end_scope(name, self);
    }
}
