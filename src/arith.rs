use std::ops::{Add, Div, Mul, Rem, Shl, Shr, Sub};
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
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Rem<Self, Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + PartialOrd
    + Copy
{
    const BITS: usize;
    fn bits(&self) -> usize;
    fn low_u64(&self) -> u64;
}

impl Num for u32 {
    const BITS: usize = 32;
    fn bits(&self) -> usize {
        (Self::BITS - self.leading_zeros()) as usize
    }
    fn low_u64(&self) -> u64 {
        *self as u64
    }
}

impl Num for u64 {
    const BITS: usize = 64;
    fn bits(&self) -> usize {
        (Self::BITS - self.leading_zeros()) as usize
    }
    fn low_u64(&self) -> u64 {
        *self
    }
}

impl Num for U256 {
    const BITS: usize = 256;
    fn bits(&self) -> usize {
        U256::bits(self)
    }
    fn low_u64(&self) -> u64 {
        Self::low_u64(&self)
    }
}

impl Num for U512 {
    const BITS: usize = 512;
    fn bits(&self) -> usize {
        U512::bits(self)
    }
    fn low_u64(&self) -> u64 {
        Self::low_u64(&self)
    }
}

impl Num for U1024 {
    const BITS: usize = 1024;
    fn bits(&self) -> usize {
        U1024::bits(self)
    }
    fn low_u64(&self) -> u64 {
        Self::low_u64(&self)
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
pub fn sqrt_mod<T: Num>(n: T, p: T) -> Option<T> {
    let n = n % p;
    let one = T::from(1);
    if p == T::from(2) {
        Some(n % p)
    } else if p.low_u64() % 4 == 3 {
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
                    return Some(mulmod(root, inv_mod(k, p).unwrap(), p));
                }
            }
            None
        }
    }
}

pub fn inv_mod<T: Num>(n: T, p: T) -> Option<T> {
    if n.bits() == 0 {
        None
    } else {
        Some(pow_mod(n, p - T::from(2), p))
    }
}

/// Modular exponentiation
pub fn pow_mod<T: Num>(n: T, k: T, p: T) -> T {
    assert!(2 * p.bits() < T::BITS);
    let mut res: T = T::from(1);
    let zero = T::from(0);
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
    assert!(2 * p.bits() < T::BITS);
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
