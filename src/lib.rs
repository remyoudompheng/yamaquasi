pub mod poly;

use std::ops::{Add, Div, Mul, Rem, Shl, Sub};
use uint::construct_uint;

// Multi-precision (fixed width) integers

pub type Uint = U512;

construct_uint! {
    pub struct U256(4);
}

construct_uint! {
    pub struct U512(8);
}

construct_uint! {
    pub struct U1024(16);
}

pub trait Num:
    From<u32>
    + AsRef<[u64]>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Rem<Self, Output = Self>
    + Shl<usize, Output = Self>
    + PartialOrd
    + Copy
{
    fn bits(&self) -> usize;
}

impl Num for U256 {
    fn bits(&self) -> usize {
        U256::bits(self)
    }
}

impl Num for U512 {
    fn bits(&self) -> usize {
        U512::bits(self)
    }
}

impl Num for U1024 {
    fn bits(&self) -> usize {
        U1024::bits(self)
    }
}

/// Rounded down integer square root
pub fn isqrt<T: Num>(n: T) -> T {
    let one = T::from(1);
    let two = T::from(2);
    let mut r = one << (n.bits() / 2);
    r = (r + n / r) / two;
    // (r + n/r)^2 = 2n + r^2 + n^2/r^2 > 4n
    while (r - one) * (r - one) > n {
        r = (r + n / r) / two;
    }
    if r * r <= n {
        r
    } else {
        r - one
    }
}

/// Square root modulo a prime number p
pub fn sqrt_mod<T: Num>(n: T, p: u32) -> Option<u32> {
    let np: u32 = (n % T::from(p)).as_ref()[0] as u32;
    if p == 2 {
        Some(np % p)
    } else if p % 4 == 3 {
        // n = r^2
        // n^((p+1)/2) = r^((p+1)/4) = n^1/2
        let r = pow_mod(np, (p + 1) / 4, p);
        if mulmod(r, r, p) == np {
            Some(r)
        } else {
            None
        }
    } else {
        if pow_mod(np, (p - 1) / 2, p) != 1 {
            None
        } else {
            // Simplified Tonelli-Shanks
            // O(2^k log(p)) where p-1 = q*2^k
            let mut q = (p - 1) / 2;
            while q % 2 == 0 {
                q /= 2
            }
            for k in 1..p {
                // n*k*k has order q with probability q/(p-1)
                let nk = mulmod(mulmod(np, k, p), k, p);
                let root = pow_mod(nk, (q + 1) / 2, p);
                if mulmod(root, root, p) == nk {
                    return Some(mulmod(root, pow_mod(k, p - 2, p), p));
                }
            }
            None
        }
    }
}

/// Modular exponentiation
pub fn pow_mod(n: u32, k: u32, p: u32) -> u32 {
    let mut res: u64 = 1;
    let mut nn: u64 = n as u64;
    let p64 = p as u64;
    let mut k = k;
    while k > 0 {
        if k % 2 == 1 {
            res = (res * nn) % p64;
        }
        nn = (nn * nn) % p64;
        k /= 2;
    }
    res as u32
}

fn mulmod(a: u32, b: u32, p: u32) -> u32 {
    return ((a as u64 * b as u64) % p as u64) as u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_mod() {
        for i in 2..997 {
            assert_eq!(pow_mod(i, 996, 997), 1)
        }
        for i in 2..996 {
            assert_eq!(pow_mod(5, i, 997) * pow_mod(5, 996 - i, 997) % 997, 1)
        }
    }

    #[test]
    fn test_sqrt_mod() {
        const PRIMES: &[u32] = &[2503, 2521, 2531, 2539, 2500213, 2500363, 300 * 1024 + 1];
        for &p in PRIMES {
            for k in 1..p / 2 {
                if k > 5000 {
                    break;
                }
                if let Some(r) = sqrt_mod(U256::from(k), p) {
                    assert_eq!(k, mulmod(r, r, p));
                }
                let r = sqrt_mod(U256::from(k as u64 * k as u64), p);
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
    fn test_isqrt() {
        for k in 1..1000 {
            let n = U256([1234, 5678, 1234, k]);
            let r = isqrt(n);
            assert!(r * r <= n, "sqrt({}) = incorrect {}", n, r);
            assert!(n < (r + 1) * (r + 1), "sqrt({}) = incorrect {}", n, r);
        }

        for k in 1..1000 {
            let n = U256([1234, k, 0, 0]);
            assert_eq!(isqrt(n * n), n);
            assert_eq!(isqrt(n * n + 1), n);
            assert_eq!(isqrt(n * n - 1), n - 1);
        }
    }
}
