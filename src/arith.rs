// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

impl Num for u128 {
    fn bits(&self) -> u32 {
        u128::BITS - u128::leading_zeros(*self)
    }

    fn to_u64(&self) -> Option<u64> {
        if *self >= (1 << 64) {
            None
        } else {
            Some(*self as u64)
        }
    }

    fn low_u64(&self) -> u64 {
        *self as u64
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
pub fn sqrt_mod<N, T: Num>(n: N, p: T) -> Option<T>
where
    N: std::ops::Rem<T, Output = T>,
{
    let n: T = n % p;
    if n == T::zero() {
        return Some(T::zero());
    }
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
            assert!(exp2 < 24);
            // Simplified Tonelli-Shanks
            // O(2^k log(p)) where p-1 = q*2^k
            let mut q = p >> 1;
            while q.low_u64() % 2 == 0 {
                q = q >> 1
            }
            let q1 = (q >> 1) + one; // (q+1)/2
            for k in 1..(1 << 24) {
                // n*k*k has order q with probability q/(p-1)
                // Some k must satisfy that property.
                let k = T::from(k);
                let nk = mulmod(mulmod(n, k, p), k, p);
                let root = pow_mod(nk, q1, p);
                if mulmod(root, root, p) == nk {
                    return Some(mulmod(root, pow_mod(k, p - T::from(2), p), p));
                }
            }
            unreachable!("sqrt_mod fail")
        }
    }
}

#[allow(dead_code)]
fn inv_mod64(n: u64, p: u64) -> Option<u64> {
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

#[derive(Clone, Debug)]
pub struct Dividers {
    p: u32,
    // Multiplier for 16-bit division
    // This is used for small primes during sieving.
    m16: u32,
    s16: usize,
    // Multiplier for 31-bit division
    pub div31: Divider31,
    // Multiplier for 64-bit division
    m64: u64,
    r64: u64,
    s64: usize,
}

impl Dividers {
    // Compute m and s such that x/p = (x*m) >> s
    // p is assumed to be a prime number.
    //
    // https://gmplib.org/~tege/divcnst-pldi94.pdf
    //
    // m must be 32 bit for u31 and 65-bit for u64
    // to obtain correct results. We use 32-bit/64-bit mantissas
    // and thus need to correct the result in the case of u64.
    pub fn new(p: u32) -> Self {
        if p == 2 {
            return Dividers {
                p,
                m16: 1,
                s16: 1,
                div31: Divider31 {
                    p: 2,
                    m31: 1,
                    s31: 1,
                },
                m64: 1,
                r64: 0,
                s64: 1,
            };
        }
        // Compute 2^k / p rounded up
        let m128: U256 = (U256::one() << 128) / (p as u64);
        let sz = m128.bits();
        // For 16 bits we can use the exact 17-bit multiplier
        let m16 = (m128 >> (sz - 17)).low_u64() as u32 + 1; // 17 bits
        let s16 = 128 + 17 - sz as usize; // m16 >> s16 = m128 >> 128
        let m64 = (m128 >> (sz - 64)).low_u64() + 1; // 64 bits
        let r64 = (U256::one() << 64) % (p as u64);
        let s64 = 128 + 64 - sz as usize; // m64 >> s64 = m128 >> 128
        assert_eq!(r64, {
            let m = (m64 - 1) >> (s64 - 64);
            !m.wrapping_mul(p as u64) + 1
        });
        Dividers {
            p,
            m16,
            s16,
            div31: Divider31::new(p),
            m64,
            r64,
            s64,
        }
    }

    pub fn modu16(&self, n: u16) -> u16 {
        if self.p == 2 {
            return n & 1;
        }
        let nm = (n as u64) * (self.m16 as u64);
        let q = (nm >> self.s16) as u16;
        n - q * self.p as u16
    }

    #[inline]
    pub fn divmod64(&self, n: u64) -> (u64, u64) {
        let nm = (n as u128) * (self.m64 as u128);
        let q = (nm >> self.s64) as u64;
        let qp = q * self.p as u64;
        if qp > n {
            (q - 1, self.p as u64 - (qp - n))
        } else {
            (q, n - qp)
        }
    }

    pub fn modi64(&self, n: i64) -> u64 {
        if n < 0 {
            let m = self.divmod64((-n) as u64).1;
            if m == 0 {
                return 0;
            }
            self.p as u64 - m
        } else {
            self.divmod64(n as u64).1
        }
    }

    pub fn divmod_uint<const N: usize>(&self, n: &BUint<N>) -> (BUint<N>, u64) {
        if self.p == 2 {
            return (n >> 1, n.low_u64() & 1);
        }
        let mut digits = n.digits().clone();
        let rem = self.divmod_uint_inplace(&mut digits);
        debug_assert!((n % BUint::<N>::from(self.p)).low_u64() == rem);
        (BUint::from_digits(digits), rem)
    }

    pub fn mod_uint<const N: usize>(&self, n: &BUint<N>) -> u64 {
        if self.p == 2 {
            return n.low_u64() & 1;
        }
        let mut digits = n.digits().clone();
        self.divmod_uint_inplace(&mut digits)
    }

    #[inline]
    fn divmod_uint_inplace<const N: usize>(&self, digits: &mut [u64; N]) -> u64 {
        let mut carry: u64 = 0;
        // self.m64 = ceil(2^k / p) >> s == ceil(2^(k-s) / p)
        // Compute the actual quotient of 2^64 by p.
        let m64 = (self.m64 - 1) >> (self.s64 - 64);
        for i in 0..N {
            let i = N - 1 - i;
            let d = digits[i];
            if d == 0 && carry == 0 {
                continue;
            }
            let (mut q, r) = self.divmod64(d);
            debug_assert!(q == d / self.p as u64);
            if carry != 0 {
                q += carry * m64;
                let (cq, cr) = self.divmod64(carry * self.r64 + r);
                q += cq;
                carry = cr;
            } else {
                carry = r;
            }
            digits[i] = q;
        }
        carry
    }

    /// Modular inverse. Prime number is supposed to be small (<= 32 bits).
    /// The algorithm is an extended binary GCD.
    pub fn inv(&self, k: u64) -> Option<u64> {
        if self.p == 2 {
            if k % 2 == 0 {
                return None;
            } else {
                return Some(1);
            }
        }
        let k = if k >= self.p as u64 {
            self.divmod64(k).1
        } else {
            k
        };
        if k == 0 {
            return None;
        }
        // x and y can never be both divisible by 2.
        // x is always less than y
        let (mut x, mut y) = (k, self.p as u64);
        let (mut a, mut b) = (1i64, 0i64); // x = ak + bp
        let (mut c, mut d) = (0i64, 1i64); // y = ck + dp

        let k = k as i64;
        let p = self.p as i64;

        // Use hardware division if bit size is very different.
        while x != 1 && y > (x << 8) {
            let (q, r) = (y as u32 / x as u32, y as u32 % x as u32);
            let q = q as i64;
            (x, y) = (r as u64, x);
            (a, b, c, d) = (c - q * a, d - q * b, a, b);
        }
        loop {
            debug_assert!(x as i64 == a * k + b * p);
            debug_assert!(y as i64 == c * k + d * p);
            debug_assert!(-p < a && a < p);
            if x == 1 {
                if a < 0 {
                    a += p;
                }
                return Some(a as u64);
            } else if x > y {
                (x, y) = (y, x);
                (a, b, c, d) = (c, d, a, b);
            } else if x % 2 == 0 {
                x >>= 1;
                if a % 2 == 0 {
                    // both a, b even
                    (a, b) = (a >> 1, b >> 1);
                } else {
                    // adjust by an odd term, both become even
                    if a < 0 {
                        (a, b) = ((a + p) / 2, (b - k) / 2);
                    } else {
                        (a, b) = ((a - p) / 2, (b + k) / 2);
                    }
                }
            } else if y % 2 == 0 {
                y >>= 1;
                if c % 2 == 0 {
                    (c, d) = (c >> 1, d >> 1);
                } else {
                    if c < 0 {
                        (c, d) = ((c + p) / 2, (d - k) / 2);
                    } else {
                        (c, d) = ((c - p) / 2, (d + k) / 2);
                    }
                }
            } else {
                (x, y) = (x, y - x);
                (a, b, c, d) = (a, b, c - a, d - b);
            }
        }
    }
}

// A divider for 31-bit integers.
// It uses a 32-bit mantissa.
#[derive(Clone, Debug)]
pub struct Divider31 {
    pub p: u32,
    pub m31: u32,
    pub s31: u32,
}

impl Divider31 {
    pub const fn new(p: u32) -> Self {
        let m64 = (1u64 << 63) / p as u64;
        let sz = 64 - u64::leading_zeros(m64);
        let m31 = (m64 >> (sz - 32)) as u32 + 1; // 32 bits
        let s31 = 63 + 32 - sz as u32; // m31 >> s31 = m63 >> 63
        Divider31 { p, m31, s31 }
    }

    #[inline]
    pub fn modi32(&self, n: i32) -> u32 {
        if n < 0 {
            let m = self.modu31((-n) as u32);
            if m == 0 {
                return 0;
            }
            self.p as u32 - m
        } else {
            self.modu31(n as u32)
        }
    }

    #[inline]
    pub fn divu31(&self, n: u32) -> u32 {
        let nm = (n as u64) * (self.m31 as u64);
        (nm >> self.s31) as u32
    }

    #[inline]
    pub fn modu31(&self, n: u32) -> u32 {
        let nm = (n as u64) * (self.m31 as u64);
        let q = (nm >> self.s31) as u32;
        n - q * self.p as u32
    }
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
        const PRIMES: &[u32] = &[
            2473,
            2503,
            2521,
            2531,
            2539,
            63977,
            2500213,
            2500363,
            300 * 1024 + 1,
            (7 << 20) + 1,
            (13 << 20) + 1,
        ];
        for &p in PRIMES {
            let p = p as u64;
            for k in 1..p / 2 {
                let k = k as u64;
                if k > 50000 || k * p > 1 << 28 {
                    break;
                }
                if let Some(r) = sqrt_mod(k, p) {
                    assert_eq!(k, mulmod(r, r, p));
                }
                let r = sqrt_mod(k * k, p);
                assert!(
                    r == Some(k) || r == Some(p - k),
                    "failed sqrt({}) mod {} got {:?}",
                    (k * k) % p,
                    p,
                    r
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

    #[test]
    fn test_dividers() {
        const M32: u32 = 100_000_000;
        const M64: u64 = 100_000_000_000_000_000;
        use crate::fbase::primes;
        let ps = primes(2000);
        for p in ps {
            let d = Dividers::new(p);
            for n in M32..M32 + std::cmp::max(1000, 2 * p) {
                let n = n as u32;
                assert_eq!(n % p, d.div31.modu31(n as u32));
            }
            let p = p as u64;
            for n in M64..M64 + std::cmp::max(1000, 2 * p) {
                assert_eq!((n / p, n % p), d.divmod64(n));
            }
            let signed = -(M64 as i64);
            let p = p as i64;
            if p != 2 && p != 5 {
                assert_eq!(p + (signed % p), d.modi64(signed) as i64);
            } else {
                assert_eq!(d.modi64(signed), 0);
            }
        }
    }

    #[test]
    fn test_dividers_u16() {
        use crate::fbase::primes;
        let ps = primes(10000);
        for p in ps {
            if p >= 1 << 16 {
                break;
            }
            let d = Dividers::new(p);
            if p < 100 {
                for i in 0..4 * p {
                    let n = i as u16;
                    assert_eq!(n % (p as u16), d.modu16(n));
                }
            }
            for i in 0..1000u64 {
                let n = ((12345 * i) & 0xffff) as u16;
                assert_eq!(n % (p as u16), d.modu16(n));
            }
        }
    }

    #[test]
    fn test_dividers_uint() {
        use crate::fbase::primes;

        let n0s: &[U1024] = &[
            // Tricky carry
            (U1024::one() << 64) + U1024::from(1_234_567_890u64),
            (U1024::one() << 65) + U1024::from(1_234_567_890u64),
            pow_mod(
                U1024::from(65537u64),
                U1024::from(1_234_567_890u64),
                (U1024::one() << 384) + U1024::one(),
            ),
        ];
        for n0 in n0s {
            let ps = primes(2000);
            for p in ps {
                let d = Dividers::new(p);
                for i in 0..100u64 {
                    let n = n0 + U1024::from(i);
                    assert_eq!((n / (p as u64), n % (p as u64)), d.divmod_uint(&n));
                }
            }
        }

        // Regression test: d.m64 ends with many zero bits.
        let d = Dividers::new(274177);
        let n = U1024::from_str("37714305606241449883").unwrap();
        assert_eq!(d.mod_uint(&n), 0);
        assert_eq!(d.divmod_uint(&n), (U1024::from(137554592858779 as u64), 0));
    }

    #[test]
    fn test_dividers_inv() {
        use crate::fbase::primes;
        let ps = primes(200);
        for p in ps {
            let d = Dividers::new(p);
            let p = p as u64;
            for i in 0..=2 * p {
                match d.inv(i) {
                    None => assert_eq!(i % p, 0),
                    Some(j) => assert_eq!((i * j) % p, 1),
                }
            }
        }
    }
}
