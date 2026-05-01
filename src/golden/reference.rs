//! Pure recursive DIF (decimation-in-frequency) NTT over the Goldilocks field.
//!
//! This is the software golden model used to verify the HDL pipeline.
//! The algorithm recursively splits the transform in half, applying
//! butterfly operations at each level.

use crate::error::Error;
use crate::field::element::GoldilocksElement;
use crate::field::roots::primitive_root_of_unity;

/// Compute the DIF NTT of `data`, returning the transformed vector.
///
/// Input length must be a power of two.  The output is in
/// bit-reversed order (standard DIF property).
///
/// # Errors
///
/// Returns an error if `data.len()` is not a power of two or exceeds 2^32.
///
/// # Examples
///
/// ```
/// use goldilocks_ntt_hdl::field::element::GoldilocksElement;
/// use goldilocks_ntt_hdl::golden::reference::dif_ntt;
///
/// let input: Vec<GoldilocksElement> = (1..=4)
///     .map(GoldilocksElement::new)
///     .collect();
/// let ntt = dif_ntt(&input);
/// assert!(ntt.is_ok());
/// assert_eq!(ntt.map(|v| v.len()).unwrap_or(0), 4);
/// ```
pub fn dif_ntt(data: &[GoldilocksElement]) -> Result<Vec<GoldilocksElement>, Error> {
    let n = data.len();
    if n == 0 || (n & (n - 1)) != 0 {
        Err(Error::Field(format!(
            "NTT length {n} is not a power of two"
        )))
    } else {
        let log_n = n.trailing_zeros();
        let root = primitive_root_of_unity(log_n)?;
        Ok(dif_ntt_recursive(data, root))
    }
}

/// Recursive DIF NTT core.
///
/// Given `data` of length n and `root` a primitive n-th root of unity,
/// returns the NTT in bit-reversed order.
fn dif_ntt_recursive(
    data: &[GoldilocksElement],
    root: GoldilocksElement,
) -> Vec<GoldilocksElement> {
    if data.len() <= 1 {
        data.to_vec()
    } else {
        let half = data.len() / 2;
        let root_sq = root * root;
        let (first_half, second_half) = data.split_at(half);

        // DIF butterfly: for each pair (a, b) across the two halves,
        //   u = a + b
        //   v = (a - b) * twiddle
        let twiddles = std::iter::successors(Some(GoldilocksElement::ONE), |acc| Some(*acc * root));

        let (upper, lower): (Vec<_>, Vec<_>) = first_half
            .iter()
            .zip(second_half.iter())
            .zip(twiddles)
            .map(|((a, b), tw)| (*a + *b, (*a - *b) * tw))
            .unzip();

        // Recurse on both halves with root^2
        let even = dif_ntt_recursive(&upper, root_sq);
        let odd = dif_ntt_recursive(&lower, root_sq);

        // Bit-reversed output: evens then odds
        even.into_iter().chain(odd).collect()
    }
}

/// Compute the inverse NTT, recovering original data from
/// the bit-reversed output of [`dif_ntt`].
///
/// The round-trip identity is:
/// `inverse_ntt(dif_ntt(x)) == x` for any power-of-two length input.
///
/// Internally: `BR(DIF(BR(y), w^{-1})) / n` where `BR` is
/// bit-reversal and `DIF` is the forward DIF kernel.
///
/// # Examples
///
/// ```
/// use goldilocks_ntt_hdl::field::element::GoldilocksElement;
/// use goldilocks_ntt_hdl::golden::reference::{dif_ntt, inverse_ntt};
///
/// let input: Vec<GoldilocksElement> = (1..=8)
///     .map(GoldilocksElement::new)
///     .collect();
///
/// let forward = dif_ntt(&input).ok();
/// let recovered = forward.and_then(|f| inverse_ntt(&f).ok());
///
/// // Round-trip recovers the original
/// assert_eq!(recovered.as_deref(), Some(input.as_slice()));
/// ```
///
/// # Errors
///
/// Returns an error if `data.len()` is not a power of two or exceeds 2^32.
pub fn inverse_ntt(data: &[GoldilocksElement]) -> Result<Vec<GoldilocksElement>, Error> {
    let n = data.len();
    if n == 0 || (n & (n - 1)) != 0 {
        Err(Error::Field(format!(
            "NTT length {n} is not a power of two"
        )))
    } else {
        let log_n = n.trailing_zeros();
        let root = primitive_root_of_unity(log_n)?;
        let root_inv = root.inverse()?;
        let n_inv =
            GoldilocksElement::from(u64::try_from(n).map_err(|e| Error::Field(e.to_string()))?)
                .inverse()?;

        // Step 1: Bit-reverse input (undo forward DIF's bit-reversal)
        let br_input = bit_reverse_permutation(data)?;
        // Step 2: DIF with inverse root (produces bit-reversed output)
        let dif_result = dif_ntt_recursive(&br_input, root_inv);
        // Step 3: Bit-reverse output and scale by 1/n
        let br_output = bit_reverse_permutation(&dif_result)?;
        Ok(br_output.into_iter().map(|x| x * n_inv).collect())
    }
}

/// Bit-reverse a vector of length 2^k.
///
/// This reorders elements so that element at index `i` moves to
/// index `bit_reverse(i, k)`.
///
/// # Errors
///
/// Returns an error if `data.len()` is not a power of two.
pub fn bit_reverse_permutation(
    data: &[GoldilocksElement],
) -> Result<Vec<GoldilocksElement>, Error> {
    let n = data.len();
    if n == 0 || (n & (n - 1)) != 0 {
        Err(Error::Field(format!("length {n} is not a power of two")))
    } else {
        let log_n = n.trailing_zeros();
        (0..n)
            .map(|i| {
                let i_u64 = u64::try_from(i).map_err(|e| Error::Field(e.to_string()))?;
                let rev = bit_reverse(i_u64, log_n);
                let rev_idx = usize::try_from(rev).map_err(|e| Error::Field(e.to_string()))?;
                data.get(rev_idx)
                    .copied()
                    .ok_or_else(|| Error::Field(format!("index {rev_idx} out of bounds")))
            })
            .collect()
    }
}

/// Reverse the lowest `bits` bits of `value`.
#[must_use]
fn bit_reverse(value: u64, bits: u32) -> u64 {
    (0..bits).fold(0_u64, |acc, i| acc | (((value >> i) & 1) << (bits - 1 - i)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntt_of_single_element() -> Result<(), Error> {
        let data = vec![GoldilocksElement::new(42)];
        let result = dif_ntt(&data)?;
        assert_eq!(result.len(), 1);
        assert_eq!(
            result
                .first()
                .copied()
                .unwrap_or(GoldilocksElement::ZERO)
                .value(),
            42
        );
        Ok(())
    }

    #[test]
    fn ntt_of_two_elements() -> Result<(), Error> {
        let a = GoldilocksElement::new(3);
        let b = GoldilocksElement::new(5);
        let result = dif_ntt(&[a, b])?;
        // DIF NTT of [a, b] with w = p-1 (2nd root of unity):
        // output[0] = a + b = 8
        // output[1] = (a - b) * w^0 = a - b = p - 2
        assert_eq!(result.len(), 2);
        assert_eq!(
            result
                .first()
                .copied()
                .unwrap_or(GoldilocksElement::ZERO)
                .value(),
            8
        );
        Ok(())
    }

    #[test]
    fn ntt_of_zeros_is_zeros() -> Result<(), Error> {
        // NTT of all-zeros must be all-zeros (linearity).
        let zeros = vec![GoldilocksElement::ZERO; 4];
        let ntt_zeros = dif_ntt(&zeros)?;
        assert!(ntt_zeros.iter().all(|x| *x == GoldilocksElement::ZERO));
        Ok(())
    }

    #[test]
    fn ntt_inverse_round_trip_size_8() -> Result<(), Error> {
        let original: Vec<_> = (1..=8).map(GoldilocksElement::new).collect();

        // Forward DIF NTT (bit-reversed output)
        let forward = dif_ntt(&original)?;
        // Inverse NTT recovers original
        let recovered = inverse_ntt(&forward)?;

        original
            .iter()
            .zip(recovered.iter())
            .try_for_each(|(orig, rec)| {
                if orig == rec {
                    Ok(())
                } else {
                    Err(Error::Field(format!(
                        "round-trip mismatch: original {orig}, recovered {rec}"
                    )))
                }
            })
    }

    #[test]
    fn ntt_rejects_non_power_of_two() {
        let data = vec![GoldilocksElement::ZERO; 3];
        assert!(dif_ntt(&data).is_err());
    }

    #[test]
    fn ntt_rejects_empty() {
        assert!(dif_ntt(&[]).is_err());
    }

    #[test]
    fn bit_reverse_basics() {
        assert_eq!(bit_reverse(0b000, 3), 0b000);
        assert_eq!(bit_reverse(0b001, 3), 0b100);
        assert_eq!(bit_reverse(0b010, 3), 0b010);
        assert_eq!(bit_reverse(0b011, 3), 0b110);
        assert_eq!(bit_reverse(0b100, 3), 0b001);
    }

    #[test]
    fn bit_reverse_permutation_is_involution() -> Result<(), Error> {
        let data: Vec<_> = (0..8).map(GoldilocksElement::new).collect();
        let once = bit_reverse_permutation(&data)?;
        let twice = bit_reverse_permutation(&once)?;
        assert_eq!(data, twice);
        Ok(())
    }
}
