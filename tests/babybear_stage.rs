//! `BabyBear` cross-field smoke test: drive a depth-1 SDF stage built
//! via [`sdf_stage_generic::<BabyBear>`] through `hdl-cat`'s
//! [`Testbench`] and verify it matches a software reference SDF state
//! machine over the `BabyBear` prime `p = 2^31 - 2^27 + 1`.
//!
//! This proves the field-generic seam: the same SDF construction code
//! that produces the `Goldilocks` pipeline also produces a working
//! `BabyBear` stage when parameterized over the [`PrimeFieldHdl`]
//! implementation.

use goldilocks_ntt_hdl::error::Error;
use goldilocks_ntt_hdl::hdl::stage::sdf_stage_generic;
use goldilocks_ntt_hdl::hdl::{BABYBEAR_PRIME_U64, BabyBear, PrimeFieldHdl};
use hdl_cat_kind::BitSeq;
use hdl_cat_sim::Testbench;

/// Pack a `BabyBear` stage input cycle into a [`BitSeq`].
///
/// Layout: `data (32) || valid (1) || step_root (32)`.
fn make_input(data: u64, valid: bool, step_root: u64) -> BitSeq {
    BabyBear::to_bitseq(data)
        .concat(BitSeq::from_vec(vec![valid]))
        .concat(BabyBear::to_bitseq(step_root))
}

/// Unpack a stage output cycle (32-bit data, 1-bit valid).
fn read_output(bits: &BitSeq) -> Result<(u64, bool), Error> {
    let (data_bits, valid_bits) = bits.clone().split_at(32);
    let data =
        BabyBear::from_bitseq(&data_bits).map_err(|e| Error::Field(format!("data decode: {e}")))?;
    let valid = valid_bits.bit(0);
    Ok((data, valid))
}

/// Software SDF reference state for `BabyBear`, mirroring the hardware
/// fill/butterfly phase logic in `crate::hdl::stage` but evaluated
/// modulo the `BabyBear` prime.
struct RefState {
    delay: Vec<u64>,
    twiddle: u64,
    counter: usize,
    depth: usize,
}

impl RefState {
    fn new(depth: usize) -> Self {
        Self {
            delay: vec![0; depth],
            twiddle: 1,
            counter: 0,
            depth,
        }
    }

    fn step(self, data_in: u64, step_root: u64) -> Result<(Self, u64), Error> {
        let p = u128::from(BABYBEAR_PRIME_U64);
        let is_butterfly = self.counter >= self.depth;
        let delayed = self
            .delay
            .last()
            .copied()
            .ok_or_else(|| Error::Field("ref delay line empty".to_string()))?;

        let (data_out, delay_in, next_tw) = if is_butterfly {
            let upper_u128 = (u128::from(delayed) + u128::from(data_in)) % p;
            let upper =
                u64::try_from(upper_u128).map_err(|e| Error::Field(format!("upper conv: {e}")))?;
            let diff = (u128::from(delayed) + p - u128::from(data_in)) % p;
            let lower_u128 = (diff * u128::from(self.twiddle)) % p;
            let lower =
                u64::try_from(lower_u128).map_err(|e| Error::Field(format!("lower conv: {e}")))?;
            let tw_u128 = (u128::from(self.twiddle) * u128::from(step_root)) % p;
            let tw = u64::try_from(tw_u128).map_err(|e| Error::Field(format!("tw conv: {e}")))?;
            (lower, upper, tw)
        } else {
            (delayed, data_in, 1)
        };

        let next_delay: Vec<u64> = core::iter::once(delay_in)
            .chain(
                self.delay
                    .iter()
                    .take(self.depth.saturating_sub(1))
                    .copied(),
            )
            .collect();

        let next_counter = if self.counter == 2 * self.depth - 1 {
            0
        } else {
            self.counter + 1
        };

        Ok((
            RefState {
                delay: next_delay,
                twiddle: next_tw,
                counter: next_counter,
                depth: self.depth,
            },
            data_out,
        ))
    }
}

fn fold_reference(depth: usize, data: &[u64], step_root: u64) -> Result<Vec<u64>, Error> {
    let init: Result<(RefState, Vec<u64>), Error> = Ok((RefState::new(depth), Vec::new()));
    data.iter()
        .try_fold(init?, |(state, outs), d| {
            let (next_state, out) = state.step(*d, step_root)?;
            Ok((
                next_state,
                outs.into_iter().chain(core::iter::once(out)).collect(),
            ))
        })
        .map(|(_, outs)| outs)
}

#[test]
fn babybear_depth_1_matches_reference() -> Result<(), Error> {
    let stage =
        sdf_stage_generic::<BabyBear>(1).map_err(|e| Error::Field(format!("build: {e}")))?;
    let step_root: u64 = 7;
    let data: Vec<u64> = vec![10, 20, 30, 40];

    let inputs: Vec<BitSeq> = data
        .iter()
        .map(|d| make_input(*d, true, step_root))
        .collect();
    let tb = Testbench::new(stage);
    let results = tb
        .run(inputs)
        .run()
        .map_err(|e| Error::Field(format!("sim: {e}")))?;

    let expected = fold_reference(1, &data, step_root)?;

    results
        .iter()
        .zip(expected.iter())
        .enumerate()
        .try_for_each(|(i, (sample, exp))| {
            let (actual, _) = read_output(sample.value())?;
            (actual == *exp)
                .then_some(())
                .ok_or_else(|| Error::Field(format!("cycle {i}: got {actual}, expected {exp}")))
        })
}

#[test]
fn babybear_depth_2_matches_reference() -> Result<(), Error> {
    let stage =
        sdf_stage_generic::<BabyBear>(2).map_err(|e| Error::Field(format!("build: {e}")))?;
    let step_root: u64 = 3;
    let data: Vec<u64> = vec![100, 200, 300, 400, 500, 600, 700, 800];

    let inputs: Vec<BitSeq> = data
        .iter()
        .map(|d| make_input(*d, true, step_root))
        .collect();
    let tb = Testbench::new(stage);
    let results = tb
        .run(inputs)
        .run()
        .map_err(|e| Error::Field(format!("sim: {e}")))?;

    let expected = fold_reference(2, &data, step_root)?;

    results
        .iter()
        .zip(expected.iter())
        .enumerate()
        .try_for_each(|(i, (sample, exp))| {
            let (actual, _) = read_output(sample.value())?;
            (actual == *exp)
                .then_some(())
                .ok_or_else(|| Error::Field(format!("cycle {i}: got {actual}, expected {exp}")))
        })
}

#[test]
fn babybear_depth_1_near_prime_no_overflow() -> Result<(), Error> {
    let stage =
        sdf_stage_generic::<BabyBear>(1).map_err(|e| Error::Field(format!("build: {e}")))?;
    let p = BABYBEAR_PRIME_U64;
    let step_root: u64 = 1;
    // Values near the prime exercise the modular reduction in
    // both add and sub paths of the butterfly.
    let data: Vec<u64> = vec![p - 1, 1, p - 2, 2];

    let inputs: Vec<BitSeq> = data
        .iter()
        .map(|d| make_input(*d, true, step_root))
        .collect();
    let tb = Testbench::new(stage);
    let results = tb
        .run(inputs)
        .run()
        .map_err(|e| Error::Field(format!("sim: {e}")))?;

    let expected = fold_reference(1, &data, step_root)?;

    results
        .iter()
        .zip(expected.iter())
        .enumerate()
        .try_for_each(|(i, (sample, exp))| {
            let (actual, _) = read_output(sample.value())?;
            (actual == *exp)
                .then_some(())
                .ok_or_else(|| Error::Field(format!("cycle {i}: got {actual}, expected {exp}")))
        })
}
