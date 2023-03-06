// Copyright 2022,2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! A collection of functions working on multi-precision
//! and/or modular arithmetic.

use std::ops::{Shl, Shr};
use std::str::FromStr;

pub use num_integer::sqrt as isqrt;
use num_integer::{Integer, Roots};
use num_traits::{One, Pow, ToPrimitive};

pub use bnum::types::{I1024, I256, U1024, U256, U512};
use bnum::{BInt, BUint};

/// Trait for types that can be used for integer-like arithmetic.
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
        Self::bits(self)
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
        Self::bits(self)
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

/// Modular inversion for 64-bit moduli.
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

/// A precomputed structure to divide by a static prime number
/// via Barrett reduction. This is used for primes from the factor base.
#[derive(Clone, Copy, Debug)]
pub struct Dividers {
    // Multiplier for 64-bit division
    pub(crate) div64: Divider64,
    // Multiplier for 16-bit division
    // This is used for small primes during sieving.
    m16: u32,
    s16: usize,
    // Multiplier for 31-bit division
    pub div31: Divider31,
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
    pub const fn new(p: u32) -> Self {
        if p == 2 {
            return Dividers {
                div64: Divider64 {
                    p: 2,
                    m64: 1,
                    r64: 0,
                    s64: 1,
                },
                m16: 1,
                s16: 1,
                div31: Divider31 {
                    p: 2,
                    m31: 1,
                    s31: 1,
                },
            };
        }
        // Compute 2^k / p rounded up
        let m64 = (1u64 << 63) / p as u64;
        let sz = 64 - u64::leading_zeros(m64);
        // For 16 bits we can use the exact 17-bit multiplier
        let m16 = (m64 >> (sz - 17)) as u32 + 1; // 17 bits
        let s16 = 63 + 17 - sz as usize; // m16 >> s16 = m128 >> 128
        Dividers {
            div64: Divider64::new(p as u64),
            m16,
            s16,
            div31: Divider31::new(p),
        }
    }

    pub fn modu16(&self, n: u16) -> u16 {
        if self.div64.p == 2 {
            return n & 1;
        }
        let nm = (n as u64) * (self.m16 as u64);
        let q = (nm >> self.s16) as u16;
        n - q * self.div64.p as u16
    }

    #[inline]
    pub fn divmod64(&self, n: u64) -> (u64, u64) {
        self.div64.divmod64(n)
    }

    pub fn modi64(&self, n: i64) -> u64 {
        self.div64.modi64(n)
    }

    pub fn divmod_uint<const N: usize>(&self, n: &BUint<N>) -> (BUint<N>, u64) {
        self.div64.divmod_uint(n)
    }

    pub fn mod_uint<const N: usize>(&self, n: &BUint<N>) -> u64 {
        self.div64.mod_uint(n)
    }
}

// Constants for Barrett reduction division by a constant prime.
// The 64-bit multiplier can only exactly divide 63-bit integers.
#[derive(Clone, Copy, Debug)]
pub struct Divider64 {
    p: u64,
    m64: u64,
    r64: u64,
    s64: u32,
}

impl Divider64 {
    pub const fn new(p: u64) -> Self {
        // Compute 2^127 / p
        let m127 = (1_u128 << 127) / p as u128;
        let sz = u128::BITS - u128::leading_zeros(m127);
        let m64 = (m127 >> (sz - 64)) as u64 + 1; // 64 bits
        let r64 = ((1_u128 << 64) % (p as u128)) as u64;
        let s64 = 127 + 64 - sz; // m64 >> s64 = m127 >> 127

        // Sanity check
        let m = (m64 - 1) >> (s64 - 64);
        let mp = (!m.wrapping_mul(p)).wrapping_add(1);
        if mp != r64 {
            panic!("incorrect divider");
        }
        Divider64 { p, m64, r64, s64 }
    }

    #[inline]
    pub fn divmod64(&self, n: u64) -> (u64, u64) {
        let nm = (n as u128) * (self.m64 as u128);
        let q = (nm >> self.s64) as u64;
        let qp = q * self.p;
        if qp > n {
            (q - 1, self.p - (qp - n))
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
            self.p - m
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
            debug_assert!(q == d / self.p);
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
}

/// Tests whether n can be written as p^k for k <= 20.
/// This is enough to filter square factors before quadratic sieve,
/// because trial division by the factor base will already catch
/// the case where k > 20 (p < 2^20 if n has 400 bits).
pub fn perfect_power<N>(n: N) -> Option<(N, u32)>
where
    N: Copy + Roots + Pow<u32, Output = N>,
{
    for k in [2, 3, 5, 7, 11, 13, 17, 19_u32] {
        let r = n.nth_root(k);
        if r.pow(k) == n {
            if let Some((rr, kk)) = perfect_power(r) {
                return Some((rr, k * kk));
            }
            return Some((r, k));
        }
    }
    None
}

/// A divider for 31-bit integers.
/// It uses a 32-bit mantissa.
#[derive(Clone, Copy, Debug)]
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
        let s31 = 63 + 32 - sz; // m31 >> s31 = m63 >> 63
        Divider31 { p, m31, s31 }
    }

    #[inline]
    pub fn modi32(&self, n: i32) -> u32 {
        if n < 0 {
            let m = self.modu31((-n) as u32);
            if m == 0 {
                return 0;
            }
            self.p - m
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
        n - q * self.p
    }
}

/// A precomputed structure to compute faster modular inverses
/// via "Montgomery modular inverse".
///
///
/// Because this is a hot path of MPQS, the memory footprint
/// of computing modular inverses modulo many bases must be kept small.
///
/// We precompute modular inverses of -2^k only for k a multiple of 8.
/// We assume that p is at most 28 bits, so that p*p << 8 fits in u64.
pub struct Inverter {
    // invpow2[k] is -2^(8k+8) mod p
    invpow2: [u32; 8],
}

impl Inverter {
    pub fn new(p: u32) -> Inverter {
        if p == 2 {
            // Return a summy value.
            return Inverter { invpow2: [0; 8] };
        }
        debug_assert!(p >> 28 == 0);
        // Inverse powers of 2 (-2^8..-2^64)
        let mut invpow2 = [0; 8];
        let mut x: u32 = 1;
        for k in 0..=8 * invpow2.len() {
            if k >= 8 && k % 8 == 0 {
                invpow2[k / 8 - 1] = p - x;
            }
            if x % 2 == 0 {
                x /= 2;
            } else {
                x = (x + p) / 2;
            }
        }
        Inverter { invpow2 }
    }

    /// Computation of Montgomery "almost inverse",
    /// using a precomputed inverse power of 2.
    ///
    /// It requires a companion Divider64 structure.
    ///
    /// The implementation follows the presentation by Lórencz
    /// <https://doi.org/10.1007/3-540-36400-5_6>
    /// and is attributed to Kaliski.
    ///
    /// See also Joppe W. Bos, Constant Time Modular Inversion
    /// <https://www.joppebos.com/files/CTInversion.pdf>
    /// <http://dx.doi.org/10.1007/s13389-014-0084-8>
    pub fn invert(&self, x: u32, div: &Divider64) -> u32 {
        if div.p == 2 {
            return x % 2;
        }
        debug_assert!(div.p >> 28 == 0);
        let p = div.p as u32;
        assert!(x != 0);
        // Similar to binary GCD, with invariants:
        // rx = -u*2^k, sx = v*2^k, gcd(u,v)=gcd(x,p)=1
        let (mut u, mut v) = (p, x);
        let (mut r, mut s) = (0_u32, 1_u32);
        let mut k = 0_u32;
        // Initially p is off and x might be even.
        if v & 1 == 0 {
            let vtz = v.trailing_zeros();
            (v, r) = (v >> vtz, r << vtz);
            k += vtz;
        }
        // Now both u and v are odd. At each loop iteration
        // they will still be odd, and both r,s <= (1<<k).
        loop {
            // Loop at most 2*p.bits() times
            let diff = (u as i64) - (v as i64);
            if diff == 0 {
                // Now u=v so they are necessarily 1.
                break;
            }
            let dtz = diff.trailing_zeros();
            k = k + dtz;
            debug_assert!(dtz > 0);
            if diff > 0 {
                // Combination of:
                // (u, r, s) = ((u-v)/2, r+s, 2s)
                // and (u, s) = (u >> dtz-1, s << dtz-1)
                (u, r, s) = ((diff as u32) >> dtz, r + s, s << dtz);
            } else {
                // Combination of:
                // (v, r, s) = (v-u)/2, 2r, r+s)
                // and (v, r) = (v >> dtz-1, r << dtz-1)
                (v, r, s) = (((-diff) as u32) >> dtz, r << dtz, r + s);
            }
            debug_assert!(((r as u64) * (x as u64) + ((u as u64) << k)) % div.p == 0);
        }
        debug_assert!(u == 1);
        debug_assert!(r <= 2 * p);
        // Now u = 1, rx = -2^k and r is smaller than 2p.
        // (k is at most twice the bit size of p).
        // Normalize k to a multiple of 8.
        let powidx = k as usize / 8;
        debug_assert!(powidx < self.invpow2.len());
        // k + (8 - k % 8) == 8 * (k / 8) + 8
        let r = (r as u64) << (8 - k % 8);
        let n = unsafe {
            r * *self.invpow2.get_unchecked(powidx) as u64
        };
        div.divmod64(n).1 as u32
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
    fn test_inv_mod_fast() {
        const PRIMES: &[u32] = &[2473, 63977, 2500363, 300 * 1024 + 1];
        for &p in PRIMES {
            let div = Divider64::new(p as u64);
            let inv = Inverter::new(p);
            for k in 1..p / 2 {
                if k > 50000 {
                    break;
                }
                let kinv = inv.invert(k, &div);
                assert_eq!(
                    (k as u64 * kinv as u64) % (p as u64),
                    1,
                    "p={p} k={k} k^-1={kinv}"
                );
            }
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
    fn test_perfect_power() {
        assert_eq!(perfect_power(6669042837601_u64), Some((1607, 4)));
        assert_eq!(perfect_power(8650415919381337933_u64), Some((13, 17)));
    }
}
