pub use num_integer::sqrt as isqrt;
use num_integer::Integer;
use num_traits::cast::ToPrimitive;
use num_traits::identities::One;
use std::ops::{Shl, Shr};
use std::str::FromStr;

pub use bnum::types::{I1024, U1024, U256, U512};
use bnum::{BInt, BUint};

// Multi-precision (fixed width) integer arithmetic

pub trait Num:
    Integer
    + One
    + Copy
    + Clone
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + From<u64>
    + FromStr
{
    fn bits(&self) -> u32;

    fn to_u64(&self) -> Option<u64>;
    fn low_u64(&self) -> u64;
}

impl Num for u64 {
    fn bits(&self) -> u32 {
        u64::BITS - u64::leading_zeros(*self)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(*self)
    }
    fn low_u64(&self) -> u64 {
        *self
    }
}

impl<const N: usize> Num for BInt<N> {
    fn bits(&self) -> u32 {
        Self::bits(&self)
    }

    fn to_u64(&self) -> Option<u64> {
        ToPrimitive::to_u64(self)
    }

    fn low_u64(&self) -> u64 {
        self.to_bits().digits()[0]
    }
}

impl<const N: usize> Num for BUint<N> {
    fn bits(&self) -> u32 {
        Self::bits(&self)
    }

    fn to_u64(&self) -> Option<u64> {
        ToPrimitive::to_u64(self)
    }

    fn low_u64(&self) -> u64 {
        self.digits()[0]
    }
}

/// Square root modulo a prime number p
pub fn sqrt_mod<T: Num>(n: T, p: T) -> Option<T> {
    let n = n % p;
    let one = T::one();
    if p == T::from(2) {
        Some(n % p)
    } else if p % T::from(4) == T::from(3) {
        // n = r^2
        // n^((p+1)/2) = r^((p+1)/4) = n^1/2
        let r = pow_mod(n, (p >> 2) + one, p);
        if mulmod(r, r, p) == n {
            Some(r)
        } else {
            None
        }
    } else {
        // p>>1 is (p-1)/2
        if pow_mod(n, p >> 1, p) != one {
            None
        } else {
            let exp2 = (p.low_u64() - 1).trailing_zeros();
            assert!(exp2 < 20);
            // Simplified Tonelli-Shanks
            // O(2^k log(p)) where p-1 = q*2^k
            let mut q = p >> 1;
            while q.low_u64() % 2 == 0 {
                q = q >> 1
            }
            let q1 = (q >> 1) + one; // (q+1)/2
            for k in 1..(4 << exp2) {
                // n*k*k has order q with probability q/(p-1)
                let k = T::from(k);
                let nk = mulmod(mulmod(n, k, p), k, p);
                let root = pow_mod(nk, q1, p);
                if mulmod(root, root, p) == nk {
                    return Some(mulmod(root, pow_mod(k, p - T::from(2), p), p));
                }
            }
            None
        }
    }
}

pub fn inv_mod64(n: u64, p: u64) -> Option<u64> {
    let e = Integer::extended_gcd(&(n as i64), &(p as i64));
    if e.gcd == 1 {
        let x = if e.x < 0 { e.x + p as i64 } else { e.x };
        assert!(x >= 0);
        Some(x as u64 % p)
    } else {
        None
    }
}

pub fn inv_mod<const N: usize>(n: BUint<N>, p: BUint<N>) -> Option<BUint<N>> {
    // Not generic, we need to switch to signed realm.
    let e = Integer::extended_gcd(&BInt::from_bits(n), &BInt::from_bits(p));
    if e.gcd.is_one() {
        let x = if e.x.is_negative() {
            e.x + BInt::from_bits(p)
        } else {
            e.x
        };
        assert!(!x.is_negative());
        Some(x.to_bits() % p)
    } else {
        None
    }
}

/// Modular exponentiation
pub fn pow_mod<T: Num>(n: T, k: T, p: T) -> T {
    let mut res: T = T::one();
    let zero = T::zero();
    let mut nn = n % p;
    let mut k = k;
    while k > zero {
        if k.low_u64() % 2 == 1 {
            res = (res * nn) % p;
        }
        nn = (nn * nn) % p;
        k = k >> 1;
    }
    res
}

fn mulmod<T: Num>(a: T, b: T, p: T) -> T {
    (a * b) % p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_mod() {
        for i in 2..997u64 {
            assert_eq!(pow_mod(i, 996, 997), 1)
        }
        for i in 2..996u64 {
            assert_eq!(pow_mod(5, i, 997) * pow_mod(5, 996 - i, 997) % 997, 1)
        }
    }

    #[test]
    fn test_sqrt_mod() {
        const PRIMES: &[u32] = &[2503, 2521, 2531, 2539, 2500213, 2500363, 300 * 1024 + 1];
        for &p in PRIMES {
            let p = p as u64;
            for k in 1..p / 2 {
                let k = k as u64;
                if k > 5000 {
                    break;
                }
                if let Some(r) = sqrt_mod(k, p) {
                    assert_eq!(k, mulmod(r, r, p));
                }
                let r = sqrt_mod(k * k, p);
                assert!(
                    r == Some(k) || r == Some(p - k),
                    "failed sqrt({}) mod {}",
                    (k * k) % p,
                    p
                )
            }
        }
    }

    #[test]
    fn test_inv_mod() {
        let n =
            U1024::from_str("2953951639731214343967989360202131868064542471002037986749").unwrap();
        for k in 1..100u64 {
            let k = U1024::from(k);
            let kinv = inv_mod(k, n).unwrap();
            assert_eq!((kinv * k) % n, U1024::one());
        }
    }

    #[test]
    fn test_isqrt() {
        for k in 1..1000u64 {
            let n = (U256::from(k) << 192) + U256::from(1234_5678_1234_5678 as u64);
            let r = isqrt(n);
            assert!(r * r <= n, "sqrt({}) = incorrect {}", n, r);
            assert!(
                n < (r + U256::one()) * (r + U256::one()),
                "sqrt({}) = incorrect {}",
                n,
                r
            );
        }

        for k in 1..1000u64 {
            let n = (U256::from(k) << 64) + U256::from(1234_5678_1234_5678 as u64);
            assert_eq!(isqrt(n * n), n);
            assert_eq!(isqrt(n * n + U256::one()), n);
            assert_eq!(isqrt(n * n - U256::one()), n - U256::one());
        }
    }
}
