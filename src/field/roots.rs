//! Roots of unity in the Goldilocks field.
//!
//! The Goldilocks prime p = 2^64 - 2^32 + 1 has multiplicative group
//! of order p - 1 = 2^32 * (2^32 - 1).  This means primitive roots
//! of unity of order 2^k exist for all k <= 32, which is exactly what
//! NTT of size up to 2^32 requires.

use crate::error::Error;
use crate::field::element::{GOLDILOCKS_PRIME, GoldilocksElement};

/// A known generator of the multiplicative group GF(p)*.
///
/// The element 7 is a primitive root modulo the Goldilocks prime.
const MULTIPLICATIVE_GENERATOR: GoldilocksElement = GoldilocksElement::new_const(7);

/// Compute a primitive 2^k-th root of unity in the Goldilocks field.
///
/// Raises the multiplicative generator g to the power (p - 1) / 2^k,
/// yielding an element of exact multiplicative order 2^k.
///
/// # Errors
///
/// Returns an error if `k > 32`, since the Goldilocks field only
/// has roots of unity of order up to 2^32.
///
/// # Examples
///
/// ```
/// use goldilocks_ntt_hdl::field::element::GoldilocksElement;
/// use goldilocks_ntt_hdl::field::roots::primitive_root_of_unity;
///
/// // The 2nd root of unity satisfies w^2 = 1, w != 1
/// let w = primitive_root_of_unity(1).ok()
///     .unwrap_or(GoldilocksElement::ONE);
/// assert_eq!(w * w, GoldilocksElement::ONE);
/// assert_ne!(w, GoldilocksElement::ONE);
/// ```
#[must_use = "this computes a root of unity; use the returned value"]
pub fn primitive_root_of_unity(k: u32) -> Result<GoldilocksElement, Error> {
    if k > 32 {
        Err(Error::Field(format!(
            "Goldilocks field has no primitive 2^{k}-th root of unity (max k = 32)"
        )))
    } else {
        // (p - 1) / 2^k = (p - 1) >> k
        let exponent = (GOLDILOCKS_PRIME - 1) >> k;
        Ok(MULTIPLICATIVE_GENERATOR.pow(exponent))
    }
}

/// Compute all twiddle factors for an NTT of size 2^k.
///
/// Returns a `Vec` of length 2^k containing `[w^0, w^1, ..., w^(2^k - 1)]`
/// where `w` is the primitive 2^k-th root of unity.
///
/// # Errors
///
/// Returns an error if `k > 32`.
pub fn twiddle_factors(k: u32) -> Result<Vec<GoldilocksElement>, Error> {
    let root = primitive_root_of_unity(k)?;
    let n = 1_usize << k;
    Ok(
        std::iter::successors(Some(GoldilocksElement::ONE), |acc| Some(*acc * root))
            .take(n)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_of_unity_order_1() -> Result<(), Error> {
        // 2^0 = 1st root of unity is always 1
        let w = primitive_root_of_unity(0)?;
        assert_eq!(w, GoldilocksElement::ONE);
        Ok(())
    }

    #[test]
    fn root_of_unity_order_2() -> Result<(), Error> {
        // 2^1 = 2nd root of unity: w^2 = 1, w != 1, so w = p-1
        let w = primitive_root_of_unity(1)?;
        assert_eq!(w * w, GoldilocksElement::ONE);
        assert_ne!(w, GoldilocksElement::ONE);
        assert_eq!(w.value(), GOLDILOCKS_PRIME - 1);
        Ok(())
    }

    #[test]
    fn root_of_unity_has_exact_order() -> Result<(), Error> {
        // For k = 8: w^256 = 1 but w^128 != 1
        let w = primitive_root_of_unity(8)?;
        assert_eq!(w.pow(256), GoldilocksElement::ONE);
        assert_ne!(w.pow(128), GoldilocksElement::ONE);
        Ok(())
    }

    #[test]
    fn root_of_unity_k32_exists() -> Result<(), Error> {
        let w = primitive_root_of_unity(32)?;
        // w^(2^32) = 1
        assert_eq!(w.pow(1_u64 << 32), GoldilocksElement::ONE);
        Ok(())
    }

    #[test]
    fn root_of_unity_k33_fails() {
        assert!(primitive_root_of_unity(33).is_err());
    }

    #[test]
    fn twiddle_factors_length() -> Result<(), Error> {
        let tw = twiddle_factors(4)?;
        assert_eq!(tw.len(), 16);
        Ok(())
    }

    #[test]
    fn twiddle_factors_first_is_one() -> Result<(), Error> {
        let tw = twiddle_factors(4)?;
        assert_eq!(
            tw.first().copied().unwrap_or(GoldilocksElement::ZERO),
            GoldilocksElement::ONE
        );
        Ok(())
    }

    #[test]
    fn twiddle_factors_are_consecutive_powers() -> Result<(), Error> {
        let tw = twiddle_factors(4)?;
        let w = primitive_root_of_unity(4)?;
        tw.iter().enumerate().try_for_each(|(i, ti)| {
            let expected = w.pow(u64::try_from(i).map_err(|e| Error::Field(e.to_string()))?);
            if *ti == expected {
                Ok(())
            } else {
                Err(Error::Field(format!(
                    "twiddle[{i}] mismatch: got {ti}, expected {expected}"
                )))
            }
        })
    }
}
