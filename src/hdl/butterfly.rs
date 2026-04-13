//! DIF (decimation-in-frequency) butterfly HDL module.
//!
//! Computes
//!
//! ```text
//! upper = a + b               (latency: 7 cycles via delay)
//! lower = (a − b) · twiddle    (latency: 7 cycles via delay)
//! ```
//!
//! Both paths are combinational followed by a 7-cycle delay, so the
//! outputs are aligned.  Total butterfly latency: 7 cycles.

use hdl_cat_bits::Bits;
use hdl_cat_circuit::{CircuitArrow, CircuitTensor, Obj};
use hdl_cat_error::Error;
use hdl_cat_ir::HdlGraphBuilder;
use hdl_cat_sync::{compose_sync, par_sync, Sync};

use crate::hdl::delay::delay_n;
use crate::hdl::field_hdl::PrimeFieldHdl;
use crate::hdl::goldilocks_field_hdl::Goldilocks;
use crate::hdl::goldilocks_reduce::goldilocks_mul_reduce_arrow;

/// Total butterfly latency in clock cycles.
pub const BUTTERFLY_LATENCY: usize = 7;

/// Type alias for 64-bit Goldilocks field element.
pub type GoldilocksElement = Bits<64>;

/// Input bundle: two operands and twiddle factor.
pub type ButterflyInput = CircuitTensor<
    CircuitTensor<Obj<GoldilocksElement>, Obj<GoldilocksElement>>,
    Obj<GoldilocksElement>,
>;

/// Output bundle: aligned upper and lower butterfly outputs.
pub type ButterflyOutput = CircuitTensor<Obj<GoldilocksElement>, Obj<GoldilocksElement>>;

/// A pipelined DIF butterfly as a sync machine.
///
/// The state type is opaque; it tracks the internal shift registers
/// for the upper delay and the lower multiply-delay path.
pub type DifButterflySync = Sync<ButterflyOutput, ButterflyInput, ButterflyOutput>;

/// Build a combinational "fork" circuit:
///
/// `((a, b), twiddle)  →  (upper, (diff, twiddle))`
///
/// where `upper = (a + b) mod p` and `diff = (a − b) mod p`.
/// The twiddle factor passes through unchanged.
///
/// Output of the fork: `(upper, (diff, twiddle))`.
type ForkOutput = CircuitTensor<
    Obj<GoldilocksElement>,
    CircuitTensor<Obj<GoldilocksElement>, Obj<GoldilocksElement>>,
>;

/// # Errors
///
/// Returns [`Error`] if IR construction fails.
fn build_fork_circuit() -> Result<CircuitArrow<ButterflyInput, ForkOutput>, Error> {
    let element_ty = Goldilocks::element_wire_ty();

    // ── Inputs ───────────────────────────────────────────────────────
    let (bld, a) = HdlGraphBuilder::new().with_wire(element_ty.clone());
    let (bld, b) = bld.with_wire(element_ty.clone());
    let (bld, tw) = bld.with_wire(element_ty);

    // ── Field constants (via PrimeFieldHdl) ──────────────────────────
    let (bld, consts) = Goldilocks::alloc_constants(bld)?;

    // ── upper = (a + b) mod p ────────────────────────────────────────
    let (bld, upper) = Goldilocks::inline_add(bld, a, b, &consts)?;

    // ── diff = (a − b) mod p ─────────────────────────────────────────
    let (bld, diff) = Goldilocks::inline_sub(bld, a, b, &consts)?;

    // ── Output: (upper, diff, tw) ────────────────────────────────────
    Ok(CircuitArrow::from_raw_parts(
        bld.build(),
        vec![a, b, tw],
        vec![upper, diff, tw],
    ))
}

/// Construct a DIF butterfly sync machine.
///
/// The butterfly takes three inputs: `((a, b), twiddle)` and produces
/// two outputs: `(upper, lower)` where:
/// - `upper = (a + b) mod p` (delayed 7 cycles)
/// - `lower = ((a - b) * twiddle) mod p` (delayed 7 cycles)
///
/// Both paths have identical 7-cycle latency, so outputs are aligned.
///
/// # Errors
///
/// Returns [`Error`] if construction fails.
pub fn dif_butterfly() -> Result<DifButterflySync, Error> {
    // Step 1: Combinational fork: ((a, b), tw) → (upper, (diff, tw))
    let fork = build_fork_circuit()?;
    let fork_sync = Sync::lift_comb(fork);

    // Step 2: 7-cycle delay for upper path
    let upper_delay = delay_n::<GoldilocksElement>(BUTTERFLY_LATENCY)?;

    // Step 3: Combinational multiply + 7-cycle delay for lower path
    let mul_comb = goldilocks_mul_reduce_arrow()?;
    let mul_sync = Sync::lift_comb(mul_comb);
    let lower_delay = delay_n::<GoldilocksElement>(BUTTERFLY_LATENCY)?;
    let lower_path = compose_sync(mul_sync, lower_delay);

    // Step 4: Parallel composition of both paths
    let parallel = par_sync(upper_delay, lower_path);

    // Step 5: Sequential composition: fork ; parallel
    let composed = compose_sync(fork_sync, parallel);

    // Cast to the declared return type via from_raw
    let (graph, inputs, outputs, init, sc) = composed.into_parts();
    Ok(hdl_cat_sync::machine::from_raw(graph, inputs, outputs, init, sc))
}

/// Software reference implementation for testing.
#[must_use]
pub fn reference_butterfly(a: u64, b: u64, twiddle: u64) -> (u64, u64) {
    use crate::hdl::common::GOLDILOCKS_PRIME_U64;
    let p = u128::from(GOLDILOCKS_PRIME_U64);
    let upper = u64::try_from((u128::from(a) + u128::from(b)) % p).unwrap_or(0);
    let diff = (u128::from(a) + p - u128::from(b)) % p;
    let lower = u64::try_from((diff * u128::from(twiddle)) % p).unwrap_or(0);
    (upper, lower)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdl::common::{bitseq_to_u64, u64_to_bitseq};
    use hdl_cat_sim::Testbench;

    #[test]
    fn fork_circuit_builds() -> Result<(), Error> {
        let _fork = build_fork_circuit()?;
        Ok(())
    }

    #[test]
    fn butterfly_builds() -> Result<(), Error> {
        let _butterfly = dif_butterfly()?;
        Ok(())
    }

    #[test]
    fn reference_implementation_works() {
        assert_eq!(reference_butterfly(0, 0, 1), (0, 0));
        let (upper, _lower) = reference_butterfly(3, 5, 1);
        assert_eq!(upper, 8);
    }

    #[test]
    fn butterfly_matches_reference() -> Result<(), Error> {
        let butterfly = dif_butterfly()?;

        let test_cases: Vec<(u64, u64, u64)> = vec![
            (0, 0, 1),
            (3, 5, 1),
            (10, 7, 2),
            (100, 50, 3),
        ];

        // Feed test cases first, then BUTTERFLY_LATENCY flush cycles.
        // Results appear at indices BUTTERFLY_LATENCY .. BUTTERFLY_LATENCY + test_count.
        let test_inputs: Vec<_> = test_cases.iter().map(|&(a, b, tw)| {
            u64_to_bitseq(a).concat(u64_to_bitseq(b)).concat(u64_to_bitseq(tw))
        }).collect();

        let flush: Vec<_> = (0..BUTTERFLY_LATENCY)
            .map(|_| u64_to_bitseq(0).concat(u64_to_bitseq(0)).concat(u64_to_bitseq(1)))
            .collect();

        let all_inputs: Vec<_> = test_inputs.into_iter().chain(flush).collect();
        let testbench = Testbench::new(butterfly);
        let results = testbench.run(all_inputs).run()?;

        // Input on cycle i appears as output on cycle i + BUTTERFLY_LATENCY
        test_cases.iter().enumerate().try_for_each(|(i, &(a, b, tw))| {
            let sample = &results[BUTTERFLY_LATENCY + i];
            let output_bits = sample.value().clone();
            let (upper_bits, lower_bits) = output_bits.split_at(64);
            let upper_out = bitseq_to_u64(&upper_bits)?;
            let lower_out = bitseq_to_u64(&lower_bits)?;
            let (expected_upper, expected_lower) = reference_butterfly(a, b, tw);
            assert_eq!(upper_out, expected_upper,
                "butterfly({a}, {b}, {tw}): upper got {upper_out:#018x}, expected {expected_upper:#018x}");
            assert_eq!(lower_out, expected_lower,
                "butterfly({a}, {b}, {tw}): lower got {lower_out:#018x}, expected {expected_lower:#018x}");
            Ok::<(), hdl_cat_error::Error>(())
        })?;

        Ok(())
    }
}
