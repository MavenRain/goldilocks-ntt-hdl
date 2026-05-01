//! The Goldilocks field element newtype.
//!
//! The Goldilocks prime is p = 2^64 - 2^32 + 1 = `0xFFFF_FFFF_0000_0001`.
//! This field is popular in zero-knowledge proof systems because its
//! structure allows efficient modular arithmetic and it has roots of
//! unity of order up to 2^32.

use crate::error::Error;

/// The Goldilocks prime: p = 2^64 - 2^32 + 1.
pub const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// An element of the Goldilocks field GF(p) where p = 2^64 - 2^32 + 1.
///
/// Internally stored as a `u64` in the canonical range `[0, p)`.
///
/// # Examples
///
/// ```
/// use goldilocks_ntt_hdl::field::element::GoldilocksElement;
///
/// let a = GoldilocksElement::new(42);
/// let b = GoldilocksElement::new(7);
/// let c = a + b;
/// assert_eq!(c.value(), 49);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct GoldilocksElement(u64);

impl GoldilocksElement {
    /// The additive identity (zero).
    pub const ZERO: Self = Self(0);

    /// The multiplicative identity (one).
    pub const ONE: Self = Self(1);

    /// Const-compatible constructor for use in `const` contexts.
    ///
    /// The caller must ensure `value < p`.
    pub const fn new_const(value: u64) -> Self {
        Self(value)
    }

    /// Create a new field element, reducing modulo p if necessary.
    pub fn new(value: u64) -> Self {
        if value >= GOLDILOCKS_PRIME {
            Self(value.wrapping_sub(GOLDILOCKS_PRIME))
        } else {
            Self(value)
        }
    }

    /// Create a field element from a value known to be in canonical range.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not in `[0, p)`.
    pub fn from_canonical(value: u64) -> Result<Self, Error> {
        if value < GOLDILOCKS_PRIME {
            Ok(Self(value))
        } else {
            Err(Error::Field(format!(
                "value {value:#018x} is not in canonical range [0, {GOLDILOCKS_PRIME:#018x})"
            )))
        }
    }

    /// The underlying `u64` value in canonical form.
    #[must_use]
    pub fn value(self) -> u64 {
        self.0
    }

    /// Compute the multiplicative inverse via Fermat's little theorem:
    /// a^(-1) = a^(p-2) mod p.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is zero.
    pub fn inverse(self) -> Result<Self, Error> {
        if self.0 == 0 {
            Err(Error::Field("cannot invert zero".to_owned()))
        } else {
            Ok(self.pow(GOLDILOCKS_PRIME - 2))
        }
    }

    /// Exponentiation by squaring.
    pub fn pow(self, exponent: u64) -> Self {
        // Iterative square-and-multiply via fold over bits.
        // We process from the MSB down.  Find the highest set bit,
        // then fold over remaining bits.
        if exponent == 0 {
            Self::ONE
        } else {
            let highest_bit = 63 - u64::from(exponent.leading_zeros());
            (0..highest_bit).rev().fold(self, |acc, i| {
                let squared = acc * acc;
                if (exponent >> i) & 1 == 1 {
                    squared * self
                } else {
                    squared
                }
            })
        }
    }
}

impl std::fmt::Display for GoldilocksElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#018x}", self.0)
    }
}

impl From<u64> for GoldilocksElement {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

/// Modular addition: (a + b) mod p.
///
/// Uses the fact that a, b < p, so a + b < 2p < 2^65.
/// If the sum overflows u64 or exceeds p, subtract p.
impl std::ops::Add for GoldilocksElement {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let (sum, carry) = self.0.overflowing_add(rhs.0);
        // If carry occurred or sum >= p, subtract p.
        if carry || sum >= GOLDILOCKS_PRIME {
            Self(sum.wrapping_sub(GOLDILOCKS_PRIME))
        } else {
            Self(sum)
        }
    }
}

/// Modular subtraction: (a - b) mod p.
///
/// If a < b, add p to wrap around.
impl std::ops::Sub for GoldilocksElement {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            Self(self.0 - rhs.0)
        } else {
            Self(self.0.wrapping_add(GOLDILOCKS_PRIME).wrapping_sub(rhs.0))
        }
    }
}

/// Modular negation: (-a) mod p = (p - a) mod p.
impl std::ops::Neg for GoldilocksElement {
    type Output = Self;

    fn neg(self) -> Self {
        if self.0 == 0 {
            Self::ZERO
        } else {
            Self(GOLDILOCKS_PRIME - self.0)
        }
    }
}

/// Modular multiplication: (a * b) mod p.
///
/// Computes the full 128-bit product, then reduces using the
/// Goldilocks structure: since 2^64 = 2^32 - 1 (mod p), we have
/// `c_hi * 2^64 + c_lo = c_lo + c_hi * (2^32 - 1) (mod p)`.
impl std::ops::Mul for GoldilocksElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let product = u128::from(self.0) * u128::from(rhs.0);
        reduce128(product)
    }
}

/// Split a `u128` into its low and high 64-bit halves.
///
/// Returns `(low_64, high_64)`.  Uses irrefutable pattern
/// destructuring to avoid both `as` casts and panicking indexing.
#[must_use]
fn split_u128(value: u128) -> (u64, u64) {
    let [
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        b8,
        b9,
        b10,
        b11,
        b12,
        b13,
        b14,
        b15,
    ] = value.to_le_bytes();
    let lo = u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
    let hi = u64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]);
    (lo, hi)
}

/// Reduce a 128-bit value modulo the Goldilocks prime.
///
/// The software model uses native `u128` modular arithmetic for
/// correctness.  The HDL layer implements the shift-based reduction
/// exploiting `2^64 = 2^32 - 1 (mod p)` separately.
fn reduce128(value: u128) -> GoldilocksElement {
    let p = u128::from(GOLDILOCKS_PRIME);
    let (lo, _hi) = split_u128(value % p);
    // value % p < p < 2^64, so _hi == 0 and lo is the canonical result.
    GoldilocksElement(lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_additive_identity() -> Result<(), Error> {
        let a = GoldilocksElement::from_canonical(42)?;
        assert_eq!(a + GoldilocksElement::ZERO, a);
        assert_eq!(GoldilocksElement::ZERO + a, a);
        Ok(())
    }

    #[test]
    fn one_is_multiplicative_identity() -> Result<(), Error> {
        let a = GoldilocksElement::from_canonical(12345)?;
        assert_eq!(a * GoldilocksElement::ONE, a);
        assert_eq!(GoldilocksElement::ONE * a, a);
        Ok(())
    }

    #[test]
    fn addition_wraps_at_prime() {
        let a = GoldilocksElement::new(GOLDILOCKS_PRIME - 1);
        let b = GoldilocksElement::new(2);
        assert_eq!((a + b).value(), 1);
    }

    #[test]
    fn subtraction_wraps_below_zero() {
        let a = GoldilocksElement::new(1);
        let b = GoldilocksElement::new(3);
        assert_eq!((a - b).value(), GOLDILOCKS_PRIME - 2);
    }

    #[test]
    fn negation_is_self_inverse() -> Result<(), Error> {
        let a = GoldilocksElement::from_canonical(999)?;
        assert_eq!(a + (-a), GoldilocksElement::ZERO);
        Ok(())
    }

    #[test]
    fn negation_of_zero_is_zero() {
        assert_eq!(-GoldilocksElement::ZERO, GoldilocksElement::ZERO);
    }

    #[test]
    fn multiplication_basic() {
        let a = GoldilocksElement::new(3);
        let b = GoldilocksElement::new(7);
        assert_eq!((a * b).value(), 21);
    }

    #[test]
    fn multiplication_near_prime() {
        let a = GoldilocksElement::new(GOLDILOCKS_PRIME - 1);
        let b = GoldilocksElement::new(GOLDILOCKS_PRIME - 1);
        // (p-1)^2 mod p = 1
        assert_eq!((a * b).value(), 1);
    }

    #[test]
    fn inverse_of_one_is_one() -> Result<(), Error> {
        assert_eq!(GoldilocksElement::ONE.inverse()?, GoldilocksElement::ONE);
        Ok(())
    }

    #[test]
    fn inverse_round_trip() -> Result<(), Error> {
        let a = GoldilocksElement::from_canonical(12345)?;
        let a_inv = a.inverse()?;
        assert_eq!(a * a_inv, GoldilocksElement::ONE);
        Ok(())
    }

    #[test]
    fn inverse_of_zero_is_error() {
        assert!(GoldilocksElement::ZERO.inverse().is_err());
    }

    #[test]
    fn pow_zero_exponent() -> Result<(), Error> {
        let a = GoldilocksElement::from_canonical(42)?;
        assert_eq!(a.pow(0), GoldilocksElement::ONE);
        Ok(())
    }

    #[test]
    fn pow_one_exponent() -> Result<(), Error> {
        let a = GoldilocksElement::from_canonical(42)?;
        assert_eq!(a.pow(1), a);
        Ok(())
    }

    #[test]
    fn fermats_little_theorem() -> Result<(), Error> {
        let a = GoldilocksElement::from_canonical(7)?;
        // a^(p-1) = 1 for a != 0
        assert_eq!(a.pow(GOLDILOCKS_PRIME - 1), GoldilocksElement::ONE);
        Ok(())
    }

    #[test]
    fn from_canonical_rejects_out_of_range() {
        assert!(GoldilocksElement::from_canonical(GOLDILOCKS_PRIME).is_err());
        assert!(GoldilocksElement::from_canonical(u64::MAX).is_err());
    }

    #[test]
    fn new_reduces_values_at_or_above_prime() {
        assert_eq!(GoldilocksElement::new(GOLDILOCKS_PRIME).value(), 0);
        assert_eq!(GoldilocksElement::new(GOLDILOCKS_PRIME + 1).value(), 1);
    }
}
