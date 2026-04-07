//! Property-based cross-checks of the hdl-cat arithmetic modules against the
//! pure `GoldilocksElement` software model.
//!
//! Each test generates random inputs in `[0, p)`, runs them through the
//! hdl-cat `Testbench` simulation for the module under test, and verifies
//! the outputs match the software reference element-by-element.

use proptest::prelude::*;

use goldilocks_ntt_hdl::field::element::GoldilocksElement;
use goldilocks_ntt_hdl::hdl::arithmetic::{
    goldilocks_add_sync, goldilocks_sub_comb, goldilocks_mul_comb,
};
use goldilocks_ntt_hdl::hdl::common::{u64_to_bitseq, bitseq_to_u64};
use hdl_cat_sim::Testbench;

const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

fn canonical() -> impl Strategy<Value = u64> {
    0_u64..GOLDILOCKS_PRIME
}

fn drive_adder(pairs: &[(u64, u64)]) -> Result<Vec<u64>, hdl_cat_error::Error> {
    let adder = goldilocks_add_sync()?;
    let testbench = Testbench::new(adder);

    let inputs: Vec<_> = pairs
        .iter()
        .map(|&(a, b)| {
            let a_bits = u64_to_bitseq(a);
            let b_bits = u64_to_bitseq(b);
            a_bits.concat(b_bits) // Concatenate the two inputs into one BitSeq
        })
        .collect();

    let outputs = testbench.run(inputs).run()?;

    outputs
        .iter()
        .map(|timed_sample| bitseq_to_u64(timed_sample.value()))
        .collect::<Result<Vec<_>, _>>()
}

fn drive_subtractor(pairs: &[(u64, u64)]) -> Result<Vec<u64>, hdl_cat_error::Error> {
    let subtractor = goldilocks_sub_comb()?;
    let testbench = Testbench::new(hdl_cat_sync::Sync::lift_comb(subtractor));

    let inputs: Vec<_> = pairs
        .iter()
        .map(|&(a, b)| {
            let a_bits = u64_to_bitseq(a);
            let b_bits = u64_to_bitseq(b);
            a_bits.concat(b_bits) // Concatenate the two inputs into one BitSeq
        })
        .collect();

    let outputs = testbench.run(inputs).run()?;

    outputs
        .iter()
        .map(|timed_sample| bitseq_to_u64(timed_sample.value()))
        .collect::<Result<Vec<_>, _>>()
}

fn drive_multiplier(pairs: &[(u64, u64)]) -> Result<Vec<u64>, hdl_cat_error::Error> {
    let multiplier = goldilocks_mul_comb()?;
    let testbench = Testbench::new(hdl_cat_sync::Sync::lift_comb(multiplier));

    let inputs: Vec<_> = pairs
        .iter()
        .map(|&(a, b)| {
            let a_bits = u64_to_bitseq(a);
            let b_bits = u64_to_bitseq(b);
            a_bits.concat(b_bits) // Concatenate the two inputs into one BitSeq
        })
        .collect();

    let outputs = testbench.run(inputs).run()?;

    outputs
        .iter()
        .map(|timed_sample| bitseq_to_u64(timed_sample.value()))
        .collect::<Result<Vec<_>, _>>()
}

proptest! {
    #[test]
    fn adder_matches_software_model(
        pairs in prop::collection::vec((canonical(), canonical()), 1..32)
    ) {
        let actual = drive_adder(&pairs).unwrap();
        let expected: Vec<u64> = pairs
            .iter()
            .map(|&(a, b)| {
                (GoldilocksElement::new(a) + GoldilocksElement::new(b)).value()
            })
            .collect();
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn subtractor_matches_software_model(
        pairs in prop::collection::vec((canonical(), canonical()), 1..32)
    ) {
        let actual = drive_subtractor(&pairs).unwrap();
        let expected: Vec<u64> = pairs
            .iter()
            .map(|&(a, b)| {
                (GoldilocksElement::new(a) - GoldilocksElement::new(b)).value()
            })
            .collect();
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn multiplier_matches_software_model(
        pairs in prop::collection::vec((canonical(), canonical()), 1..32)
    ) {
        let actual = drive_multiplier(&pairs).unwrap();
        let expected: Vec<u64> = pairs
            .iter()
            .map(|&(a, b)| {
                (GoldilocksElement::new(a) * GoldilocksElement::new(b)).value()
            })
            .collect();
        prop_assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn basic_arithmetic_operations_work() -> Result<(), hdl_cat_error::Error> {
        // Basic sanity tests
        let pairs = vec![(1, 2), (3, 5), (10, 20)];

        let add_results = drive_adder(&pairs)?;
        assert_eq!(add_results, vec![3, 8, 30]);

        let sub_results = drive_subtractor(&pairs)?;
        let expected_subs: Vec<u64> = pairs
            .iter()
            .map(|&(a, b)| (GoldilocksElement::new(a) - GoldilocksElement::new(b)).value())
            .collect();
        assert_eq!(sub_results, expected_subs);

        let mul_results = drive_multiplier(&pairs)?;
        assert_eq!(mul_results, vec![2, 15, 200]);

        Ok(())
    }
}
