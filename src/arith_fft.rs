// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! An implementation of classical Schonhäge-Strassen multiplication
//! for polynomials. It is only used to implement quasilinear
//! multipoint evaluation for P-1/ECM algorithms.
//!
//! It uses Kronecker substitution so that multiple polynomial coefficient
//! are packed into a single input coefficient for the FFT convolution.
//! When FFT elements are larger than 1024 bits, Karatsuba multiplication is used.
//! The resulting complexity is expected to be about O(n^1.29)
//!
//! The typical FFT sizes are:
//! - N=16 (1024 bits, FFT size 4096):
//!   multiply degree 2048 polynomials with 500 bit coeffs
//! - N=64 (4096 bits, FFT size 16384):
//!   multiply degree 16k polynomials with 500-bit coeffs (pack 2 into 2048b)
//! - N=128 (8192 bits, FFT size 32768):
//!   multiply degree 64k polynomials with 500-bit coeffs (pack 4 into 4096b)
//! - N=256 (16k bits, FFT size 64k):
//!   multiply degree 256k polynomials with 500-bit coeffs (pack 8 into 8192b)
//!
//! Larger FFT sizes would be useful for polynomial degrees over 1 million,
//! but they are not used in practice for ECM.

use crate::arith_montgomery::{MInt, ZmodN};

/// Convolution product modulo n and X^size-1 where size <= 1M
/// The Kronecker substitution method can be used to pack multiple
/// coefficients in each FFT input element. The result will be
/// filled with (p1*p2)[offset..]
pub fn convolve_modn(
    zn: &ZmodN,
    size: usize,
    p1: &[MInt],
    p2: &[MInt],
    res: &mut [MInt],
    offset: usize,
) {
    // Convolution coefficients (size * n²) must not exceed 1024 bits.
    assert!(zn.n.bits() <= 500);
    match size {
        ..=4096 => _convolve_modn::<16>(zn, size, p1, p2, res, offset),
        // Don't use F2048 because we cannot fit 2 coeffs per element.
        4097..=32768 => _convolve_modn::<64>(zn, size, p1, p2, res, offset),
        32769..=131072 => _convolve_modn::<128>(zn, size, p1, p2, res, offset),
        131073..=524288 => _convolve_modn::<256>(zn, size, p1, p2, res, offset),
        _ => panic!("unsupported"),
    }
}

// Actual implementation of convolve_modn for a specific FFT size.
fn _convolve_modn<const N: usize>(
    zn: &ZmodN,
    size: usize,
    p: &[MInt],
    q: &[MInt],
    res: &mut [MInt],
    offset: usize,
) {
    // Multiply xR * yR => xyR²
    let msize: usize = p[0].0.len();
    // If N is above 64, copy 8 MInt words at offset 17*k
    // N=64 can contain 2 MInts (0..8, 17..25)
    // N=128 can contain 4 MInts (0..8, 17..25, 34..42, 51..59)
    // N=256 can contain 8 MInts (0..8, 17..25, ... 119..127)
    if N < 64 {
        // Inputs map 1-to-1 to FFT inputs.
        let mut vp = vec![FInt::<N>::default(); size];
        let mut vq = vec![FInt::<N>::default(); size];
        for i in 0..p.len() {
            vp[i].0[..msize].copy_from_slice(&p[i].0[..]);
        }
        for i in 0..q.len() {
            vq[i].0[..msize].copy_from_slice(&q[i].0[..]);
        }
        let vpq = mulfft(&vp, &vq);
        for i in 0..res.len() {
            let mut digits = [0u64; 16];
            digits.copy_from_slice(&vpq[offset + i].0[..16]);
            res[i] = zn.redc_large(&digits);
        }
    } else {
        // N/32 inputs map to 1 input (degree N/32-1 polynomial)
        let logpack = N.trailing_zeros() - 5;
        let mut vp = vec![FInt::<N>::default(); size >> logpack];
        let mut vq = vec![FInt::<N>::default(); size >> logpack];
        // Inputs map 1-to-1 to FFT inputs.
        for i in 0..p.len() {
            let j = i & (N / 32 - 1);
            vp[i >> logpack].0[17 * j..17 * j + msize].copy_from_slice(&p[i].0[..]);
        }
        for i in 0..q.len() {
            let j = i & (N / 32 - 1);
            vq[i >> logpack].0[17 * j..17 * j + msize].copy_from_slice(&q[i].0[..]);
        }
        let vpq = mulfft(&vp, &vq);
        // Each output maps to N/16-1 coefficients (degree N/16-2)
        for i in 0..vpq.len() {
            for j in 0..N / 16 - 1 {
                let idx = (i << logpack) + j;
                let idx = if offset <= idx && idx < offset + res.len() {
                    idx - offset
                } else {
                    continue;
                };
                let mut digits = [0u64; 16];
                // FIXME: read the 17th words for n > 500 bits
                // Last j is offset n + n/16 - 34 .. n + n/16 - 18 < n
                digits.copy_from_slice(&vpq[i].0[17 * j..17 * j + 16]);
                res[idx] = zn.add(res[idx], zn.redc_large(&digits));
            }
        }
    }
}

/// Convolution product (polynomial product modulo X^(2^k) - 1)
pub fn mulfft<const N: usize>(p1: &[FInt<N>], p2: &[FInt<N>]) -> Vec<FInt<N>> {
    let l = p1.len();
    assert_eq!(l, p2.len());
    assert_eq!(l & (l - 1), 0);
    assert!(l <= 256 * N);
    let k = l.trailing_zeros();
    // Forward FFT of size 2^k
    let mut fp1 = vec![FInt::default(); l];
    let mut fp2 = vec![FInt::default(); l];
    fft(&p1, &mut fp1, 0, k, true);
    fft(&p2, &mut fp2, 0, k, true);
    // Pointwise product
    for i in 0..(1 << k) {
        fp1[i] = fp1[i].mul(&fp2[i]);
    }
    // Inverse FFT
    fft(&fp1, &mut fp2, 0, k, false);
    fp2
}

/// Compute the forward Fourier transform of src into dst.
/// dst[j] = Sum(src[i] ω^ij) where ω is a primitive 2^k root of unity.
///
/// The inverse Fourier transform replaces ω by ω^-1
/// and divides by 2^k.
fn fft<const N: usize>(src: &[FInt<N>], dst: &mut [FInt<N>], depth: u32, k: u32, fwd: bool) {
    debug_assert!(dst.len() == 1 << k);
    // At each stage, transform src[i<<depth] for i in 0..2^k into dst.
    if k == 0 {
        dst[0] = src[0].clone();
        if !fwd {
            // For inverse FFT, divide by 2^depth.
            dst[0].shr(depth);
        }
        return;
    } else if k == 1 {
        dst[0] = src[0].clone();
        dst[1] = src[0].clone();
        dst[0].add_assign(&src[1 << depth]);
        dst[1].sub_assign(&src[1 << depth]);
        if !fwd {
            // For inverse FFT, divide by 2^(depth+1).
            dst[0].shr(depth + 1);
            dst[1].shr(depth + 1);
        }
        return;
    }
    // Transform odd/even indices
    let half = 1 << (k - 1);
    fft(src, &mut dst[..half], depth + 1, k - 1, fwd);
    fft(&src[1 << depth..], &mut dst[half..], depth + 1, k - 1, fwd);
    // Twiddle
    let (dst1, dst2) = dst.split_at_mut(half);
    for idx in 0..half {
        // Compute X + ω^idx Y, X - ω^idx Y where ω is the 2^k primitive root of unity.
        if fwd {
            dst2[idx].twiddle(idx as u32, k);
        } else {
            dst2[idx].twiddle((1 << k) - idx as u32, k);
        }
        butterfly(&mut dst1[idx], &mut dst2[idx]);
    }
}

// Arithmetic modulo 2^64N+1

/// An integer modulo 2^64N + 1.
/// It is always assumed to be normalized either x[N] == 0
/// or x[N] == 1 and x[i] == 0 for i < N.
///
/// Modulo 2^64N + 1, 2 is a 128N-th root of unity and
/// ω = sqrt(2) is a 256N-th root of unity.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FInt<const N: usize>(pub [u64; N], u64);

impl<const N: usize> Default for FInt<N> {
    fn default() -> Self {
        FInt([0; N], 0)
    }
}

fn _add_slices(z: &mut [u64], x: &[u64]) -> u64 {
    let mut carry = 0_u64;
    for i in 0..x.len() {
        unsafe {
            let xi = *x.get_unchecked(i) as u128;
            let zi = z.get_unchecked_mut(i);
            let sum = xi + (*zi as u128) + (carry as u128);
            *zi = sum as u64;
            carry = (sum >> 64) as u64;
        }
    }
    carry
}

fn _sub_slices(z: &mut [u64], x: &[u64]) -> u64 {
    let mut carry = 0;
    for i in 0..x.len() {
        unsafe {
            let xi = *x.get_unchecked(i);
            let zi = z.get_unchecked_mut(i);
            let (xxi, mut cc) = xi.overflowing_add(carry);
            if cc {
                // unlikely
                *zi = (*zi).wrapping_sub(xxi);
                carry = 1;
            } else {
                (*zi, cc) = (*zi).overflowing_sub(xxi);
                carry = if cc { 1 } else { 0 };
            }
        }
    }
    u64::from(carry)
}

/// Replace x, y by x+y, x-y
fn butterfly<const N: usize>(x: &mut FInt<N>, y: &mut FInt<N>) {
    if x.1 == 1 || y.1 == 1 {
        (*x, *y) = (x.add(y), x.sub(y));
        return;
    }
    let mut carry_add = 0_u64;
    let mut carry_sub = 1_u64;
    for i in 0..N {
        unsafe {
            let xi = x.0.get_unchecked_mut(i);
            let yi = y.0.get_unchecked_mut(i);
            let add = (*xi as u128) + (*yi as u128) + (carry_add as u128);
            let sub = (*xi as u128) + ((!*yi) as u128) + (carry_sub as u128);
            *xi = add as u64;
            *yi = sub as u64;
            carry_add = (add >> 64) as u64;
            carry_sub = (sub >> 64) as u64;
        }
    }
    x.1 += carry_add;
    if carry_sub == 0 {
        // Add 1.
        y.add_small(1);
    }
    x.reduce();
    y.reduce();
}

impl<const N: usize> FInt<N> {
    #[cfg(test)]
    const N: usize = N;

    fn reduce(&mut self) {
        let z = &mut self.0;
        if z[0] >= self.1 {
            // common case
            z[0] -= self.1;
            self.1 = 0;
            return;
        } else {
            // z[0] < self.1
            let mut carry = self.1;
            self.1 = 0;
            for i in 0..N {
                let (zz, c) = z[i].overflowing_sub(carry);
                z[i] = zz;
                if !c {
                    carry = 0;
                    break;
                } else {
                    carry = 1;
                }
            }
            if carry == 1 {
                // We still have to subtract 2^64N == add 1
                for i in 0..N {
                    let (zz, c) = z[i].overflowing_add(carry);
                    z[i] = zz;
                    if !c {
                        carry = 0;
                        break;
                    } else {
                        carry = 1;
                    }
                }
                self.1 = carry;
            }
        }
    }

    fn is_reduced(&self) -> bool {
        if self.1 == 1 {
            self.0.iter().all(|&v| v == 0)
        } else {
            true
        }
    }

    fn add(&self, rhs: &FInt<N>) -> FInt<N> {
        debug_assert!(self.is_reduced(), "{:?}", self);
        debug_assert!(rhs.is_reduced(), "{:?}", rhs);
        let mut z = self.clone();
        z.add_assign(rhs);
        z
    }

    fn add_assign(&mut self, rhs: &FInt<N>) {
        let carry = _add_slices(&mut self.0, &rhs.0);
        self.1 += carry + rhs.1;
        self.reduce();
    }

    /// Adds a small integer. The result is NOT normalized.
    fn add_small(&mut self, x: u64) {
        #[allow(unused_assignments)]
        let mut c = false;
        (self.0[0], c) = self.0[0].overflowing_add(x);
        for i in 1..N {
            (self.0[i], c) = self.0[i].overflowing_add(u64::from(c));
            if !c {
                break;
            }
        }
        if c {
            self.1 += 1
        }
    }

    fn sub(&self, rhs: &FInt<N>) -> FInt<N> {
        debug_assert!(self.is_reduced(), "{:?}", self);
        debug_assert!(rhs.is_reduced(), "{:?}", rhs);
        let mut z = self.clone();
        z.sub_assign(rhs);
        z
    }

    fn sub_assign(&mut self, rhs: &FInt<N>) {
        let carry = if rhs.1 == 1 {
            1
        } else {
            _sub_slices(&mut self.0, &rhs.0)
        };
        // If carry, z = self - rhs + 2^64N = self - rhs - 1
        if carry == 1 {
            self.add_small(1);
        }
        self.reduce();
    }

    fn mul(&self, rhs: &FInt<N>) -> FInt<N> {
        debug_assert!(self.is_reduced(), "{:?}", self);
        debug_assert!(rhs.is_reduced(), "{:?}", rhs);
        if self.1 == 1 {
            return FInt::default().sub(rhs);
        } else if rhs.1 == 1 {
            return FInt::default().sub(self);
        }
        // Karatsuba multiplication. We only use exact powers of 2.
        fn karatsuba(z: &mut [u64], p: &[u64], q: &[u64], tmp: &mut [u64]) {
            z.fill(0);
            let n = p.len();
            if n <= 16 {
                mulbasic(z, p, q);
                return;
            }
            let half = n / 2;
            // Compute
            // plo*qlo phi*qhi
            //     +middle
            // where middle = (plo+phi)*(qlo+qhi) - plo qlo - phi qhi
            // Compute middle first:
            let carrymid = {
                let (zl, zmr) = z.split_at_mut(half);
                let (zmid, zr) = zmr.split_at_mut(n);
                zl.copy_from_slice(&p[0..half]);
                let carryp = _add_slices(zl, &p[half..n]);
                zr.copy_from_slice(&q[0..half]);
                let carryq = _add_slices(zr, &q[half..n]);
                karatsuba(zmid, zl, zr, tmp);
                // add extra terms
                let mut carry = carryp & carryq;
                if carryq == 1 {
                    carry += _add_slices(&mut zmid[half..], zl);
                }
                if carryp == 1 {
                    carry += _add_slices(&mut zmid[half..], zr);
                }
                zl.fill(0);
                zr.fill(0);
                carry
            };
            // Compute plo qlo and phi qhi
            let (buf, tmphi) = tmp.split_at_mut(2 * n);
            let (blo, bhi) = buf.split_at_mut(n);
            karatsuba(blo, &p[0..half], &q[0..half], tmphi);
            karatsuba(bhi, &p[half..n], &q[half..n], tmphi);
            // Subtract from middle
            let carrylo = _sub_slices(&mut z[half..half + n], blo);
            let carryhi = _sub_slices(&mut z[half..half + n], bhi);
            debug_assert!(carrymid - (carrylo + carryhi) <= 1);
            // must be 0 or 1
            z[half + n] = carrymid - (carrylo + carryhi);
            // Combine result
            let carry1 = _add_slices(&mut z[0..n], blo);
            debug_assert!(bhi[0] != !0);
            bhi[0] += carry1; // cannot overflow
            let carry2 = _add_slices(&mut z[n..], bhi);
            // cannot overflow
            debug_assert!(carry2 == 0);
        }
        fn mulbasic(z: &mut [u64], p: &[u64], q: &[u64]) {
            for i in 0..p.len() {
                let mut carry = 0;
                for j in 0..q.len() {
                    unsafe {
                        let xi = *p.get_unchecked(i) as u128;
                        let yj = *q.get_unchecked(j) as u128;
                        let xy = xi * yj + (carry as u128);
                        let z1 = xy as u64;
                        let z2 = (xy >> 64) as u64;
                        let zij = z.get_unchecked_mut(i + j);
                        let (zz1, c) = zij.overflowing_add(z1);
                        *zij = zz1;
                        carry = z2 + (if c { 1 } else { 0 });
                    }
                }
                z[i + q.len()] = carry;
            }
        }
        fn mulk<const N: usize>(z: &mut [u64], p: &[u64], q: &[u64]) {
            let mut tmp = [[0_u64; N]; 4];
            let tmp_ptr = &mut tmp[0][0] as *mut u64;
            let tmpmut = unsafe { std::slice::from_raw_parts_mut(tmp_ptr, 4 * N) };
            karatsuba(z, p, q, tmpmut);
        }
        let mut z = [[0; N]; 2];
        let z_ptr = &mut z[0][0] as *mut u64;
        let zmut = unsafe { std::slice::from_raw_parts_mut(z_ptr, 2 * N) };
        mulk::<N>(zmut, &self.0, &rhs.0);
        FInt(z[0], 0).sub(&FInt(z[1], 0))
    }

    /// Multiply self by ω^i where ω is a primitive 2^k-th root of unity
    /// ω = 2^(64n/2^K) where 2^(1/2) = 2^48n - 2^16n
    fn twiddle(&mut self, i: u32, k: u32) {
        if k == 0 {
            return;
        }
        let shift = (128 * i * N as u32) >> k;
        let halfshift = (i % 2 == 1) && (1 << k == 256 * N);
        if halfshift {
            // Multiply by sqrt(2)
            let mut y = self.clone();
            y.shl(16 * N as u32);
            self.shl(48 * N as u32);
            *self = self.sub(&y);
        }
        self.shl(shift);
    }

    /// Multiply by 2^s
    fn shl(&mut self, s: u32) {
        debug_assert!(self.is_reduced(), "{:?}", self);
        let s = s % (128 * N as u32);
        if self.1 == 1 {
            // -1 << s
            let mut z = FInt::default();
            z.0[0] = 1;
            z.shl(s);
            *self = FInt::default().sub(&z);
            return;
        }
        let nn = 64 * N as u32;
        let mut zlo = [0; N];
        let mut zhi = [0; N];
        let mut carry = 0u64;
        let sabs = if s >= nn { s - nn } else { s };
        let shi = sabs as usize / 64;
        let slo = sabs % 64;
        for i in 0..N - shi {
            let xi = ((self.0[i] as u128) << slo) | (carry as u128);
            zlo[i + shi] = xi as u64;
            carry = (xi >> 64) as u64;
        }
        for i in N - shi..N {
            let xi = ((self.0[i] as u128) << slo) | (carry as u128);
            zhi[i + shi - N] = xi as u64;
            carry = (xi >> 64) as u64;
        }
        // Top word
        zhi[shi] = carry;
        if s >= nn {
            let c = _sub_slices(&mut zhi, &zlo);
            self.0 = zhi;
            self.1 = 0;
            if c == 1 {
                self.add_small(1);
            }
        } else {
            let c = _sub_slices(&mut zlo, &zhi);
            self.0 = zlo;
            self.1 = 0;
            if c == 1 {
                self.add_small(1);
            }
        }
    }

    fn shr(&mut self, s: u32) {
        if s == 0 {
            return;
        }
        self.shl(128 * N as u32 - s);
    }
}

#[cfg(test)]
type F1024 = FInt<16>;

#[test]
fn test_fint_reduce() {
    let zero = F1024::default();
    // Reduce zero is a noop
    let mut z = zero.clone();
    z.reduce();
    assert_eq!(zero, z);
    // Reduce 2^64N is a noop.
    let mut z1 = F1024::default();
    z1.1 = 1;
    let mut z2 = z1.clone();
    z2.reduce();
    assert_eq!(z1, z2);
    // 2^64N+1 reduces to zero
    let mut z1 = F1024::default();
    z1.0[0] = 1;
    z1.1 = 1;
    z1.reduce();
    assert_eq!(z1, zero);
    // 4x 2^64N+1 reduces to zero
    let mut z1 = F1024::default();
    z1.0[0] = 4;
    z1.1 = 4;
    z1.reduce();
    assert_eq!(z1, zero);
    // 2 * 2^64N + 1 reduces to 2^64N
    let mut z1 = F1024::default();
    z1.0[0] = 1;
    z1.1 = 2;
    z1.reduce();
    assert_eq!(z1, FInt([0; 16], 1));
    // 1 * 2^64N + 2 reduces to 1
    let mut z1 = F1024::default();
    z1.0[0] = 2;
    z1.1 = 1;
    z1.reduce();
    let mut one = F1024::default();
    one.0[0] = 1;
    assert_eq!(z1, one);
    // 3 * 2^64N + 1 reduces to -2
    let mut z1 = F1024::default();
    z1.0[0] = 1;
    z1.1 = 3;
    z1.reduce();
    assert_eq!(z1, FInt([!0; 16], 0));
}

#[test]
fn test_fint_add() {
    let mut z1 = F1024::default();
    let mut z2 = F1024::default();
    z1.1 = 1;
    z2.1 = 1;
    assert_eq!(z1.add(&z2), FInt([!0; F1024::N], 0));
}

#[test]
fn test_fint_sub() {
    let z1 = FInt([0; F1024::N], 1);
    let mut z2 = FInt([0; F1024::N], 0);
    z2.0[0] = 1;
    assert_eq!(z1.sub(&z2), FInt([!0; F1024::N], 0));
}

#[test]
fn test_fint_butterfly() {
    let mut z1 = F1024::default();
    let mut z2 = F1024::default();
    for i in 0..F1024::N {
        z1.0[i] = (1 + i as u64).wrapping_mul(123_456_789_123_456_789);
        z2.0[i] = (2 + i as u64).wrapping_mul(987_654_321_987_654_321);
    }
    let x = z1.add(&z2);
    let y = z1.sub(&z2);
    butterfly(&mut z1, &mut z2);
    assert_eq!(z1, x);
    assert_eq!(z2, y);
}

#[test]
fn test_fint_mul() {
    let mut z1 = FInt([0; F1024::N], 0);
    let mut z2 = FInt([0; F1024::N], 0);
    // z1 = 123<<960 + 456
    z1.0[0] = 456;
    z1.0[F1024::N - 1] = 123;
    // z2 = 456<<960 + 789
    z2.0[0] = 789;
    z2.0[F1024::N - 1] = 456;
    let z = z1.mul(&z2);
    let mut expect = FInt([0; F1024::N], 0);
    expect.0[F1024::N - 1] = 0x4a756;
    expect.0[F1024::N - 2] = !0xdb17;
    expect.0[0] = 0x57d68;
    assert_eq!(z, expect);
}

#[test]
fn test_fint_shift() {
    const N: usize = 16;
    let mut z = FInt([0; N], 0);
    for i in 0..N {
        z.0[i] = (123 + i as u64).wrapping_mul(123456789_123456789);
    }
    // z << k << 1024-k == -z
    for k in 0..1024 {
        let mut zz = z.clone();
        zz.shl(k);
        zz.shl(1024 - k);
        assert_eq!(z.add(&zz), FInt::default(), "shift {k}");
    }
    // z << k << 2048-k == z
    for k in 0..=2048 {
        let mut zz = z.clone();
        zz.shl(k);
        zz.shl(2048 - k);
        assert_eq!(z, zz);
    }
    // z << k >> k == z
    for k in 0..=2048 {
        let mut zz = z.clone();
        zz.shl(k);
        zz.shr(k);
        assert_eq!(z, zz);
    }
    for k in 0..1024 {
        let mut zz = z.clone();
        zz.twiddle(k, 10);
        zz.twiddle(1024 - k, 10);
        assert_eq!(z, zz);
    }
    for k in 0..2048 {
        let mut zz = z.clone();
        zz.twiddle(k, 11);
        zz.twiddle(2048 - k, 11);
        assert_eq!(z, zz);
    }
    for k in 0..4096 {
        let mut zz = z.clone();
        zz.twiddle(k, 12);
        zz.twiddle(4096 - k, 12);
        assert_eq!(z, zz);
    }
}

#[test]
fn test_fft() {
    for w in [4, 16, 256, 512, 1024, 2048, 4096] {
        let mut v1 = vec![F1024::default(); w];
        let mut v2 = vec![F1024::default(); w];
        let mut v3 = vec![F1024::default(); w];
        for idx in 0..w {
            v1[idx].0[0] = 123 * idx as u64 + 456;
        }
        fft(&v1, &mut v2, 0, w.trailing_zeros(), true);
        fft(&v2, &mut v3, 0, w.trailing_zeros(), false);
        for i in 0..v1.len() {
            assert_eq!(v1[i], v3[i], "FAIL w={w} index={i}");
        }
    }
}

#[test]
fn test_mulfft() {
    for w in [4, 16, 256, 512, 1024, 2048] {
        let mut v1 = vec![F1024::default(); 2 * w];
        let mut v2 = vec![F1024::default(); 2 * w];
        for idx in 0..w {
            v1[idx].0[0] = 123 * idx as u64 + 456;
            v2[idx].0[0] = 456 * idx as u64 + 789;
        }

        let v = mulfft(&v1, &v2);
        for i in 0..2 * w {
            let mut expect = 0;
            for idx1 in 0..=i {
                expect += v1[idx1].0[0] * v2[i - idx1].0[0];
            }
            assert_eq!(expect, v[i].0[0], "index {i}");
        }
    }
}
