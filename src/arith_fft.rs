// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! An implementation of classical Schonhäge-Strassen multiplication
//! for polynomials. It is only used to implement quasilinear
//! multipoint evaluation for P-1/ECM algorithms.
//!
//! # Fermat transform
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
//!
//! # Many prime NTT transform
//!
//! An alternate way of computing polynomial convolution is to perform
//! convolution in Z/pZ[X] for several u64-sized primes and lift the result
//! to Z[X] using CRT. It doesn't require multiprecision arithmetic during the FFT itself
//! and achieves true O(n log n) complexity with no additional code.
//!
//! Using primes p = a*2^k+1 enables FFT for all sizes that are reasonable powers of two.
//! There is no restriction on the size of original modulus N (currently bounded to 512 bits)
//! except that the CRT has quadratic complexity wrt the size of N.

use std::cmp::min;

use crate::arith;
use crate::arith_montgomery::{self, MInt, ZmodN};

/// Convolution product modulo n and X^size-1 where size <= 1M
///
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
    // If coefficients are small, they can fit in a size 2^S FFT array element
    // as A base 2^B digits
    // If (2A-1)(2B+S) < 64N (convolution result must fit)
    //    size / A < 2^S
    let (fsize, logpack, stride) = match (zn.n.bits(), size) {
        // Fit in F1024, max FFT size 4096
        // A=2 (3x 5 words, bits < 150)
        (0..=150, 0..=8192) => (1024, 1, 5),
        // A=1
        (0..=500, 0..=4096) => (1024, 0, 0),
        // Fit in F2048, max FFT size 8192
        // A=2 (3x 10 words, bits < 310)
        (0..=310, 0..=16384) => (2048, 1, 10),
        // A=1 actually not interesting.
        //(0..=512, 0..=8192) => (2048, 0),
        // Fit in F4096, max FFT size 16384
        // A=4 (7x 9 words, bits < 280)
        (0..=280, 0..=65536) => (4096, 2, 9),
        // A=2 (3x 20 words)
        (0..=512, 0..=32768) => (4096, 1, 17),
        // Fit in F8192, max FFT size 32768
        // A=8 (15x 8 words)
        (0..=245, 0..=262144) => (8192, 3, 8),
        // A=4 (7x 18 words)
        (0..=512, 0..=131072) => (8192, 2, 17),
        // Fit in F16384, max FFT size 65536
        // A=8 (15x 17 words)
        (0..=512, 0..=524288) => (16384, 3, 17),
        (nbits, _) => panic!("cannot fit size {size} convolution with {nbits} bit coefficients"),
    };
    assert!(zn.n.bits() <= 500);
    match fsize {
        1024 => _convolve_modn::<16>(zn, size, logpack, stride, p1, p2, res, offset),
        2048 => _convolve_modn::<32>(zn, size, logpack, stride, p1, p2, res, offset),
        4096 => _convolve_modn::<64>(zn, size, logpack, stride, p1, p2, res, offset),
        8192 => _convolve_modn::<128>(zn, size, logpack, stride, p1, p2, res, offset),
        16384 => _convolve_modn::<256>(zn, size, logpack, stride, p1, p2, res, offset),
        _ => unreachable!("impossible"),
    }
}

// Compute convolution of 2 elements of Z/nZ[X] using multiple prime NTT.
pub fn convolve_modn_ntt(
    mzp: &MultiZmodP,
    size: usize,
    p1: &[MInt],
    p2: &[MInt],
    res: &mut [MInt],
    offset: usize,
) {
    assert_eq!(size & (size - 1), 0);
    let logsize = size.trailing_zeros();
    assert!(mzp.k >= logsize);
    // Map p1 and p2 to residues.
    let w = mzp.w;
    let mut f1 = vec![0; w * size];
    let mut f2 = vec![0; w * size];
    for i in 0..p1.len() {
        mzp.from_mint(&mut f1[w * i..w * (i + 1)], &p1[i]);
    }
    for i in 0..p2.len() {
        mzp.from_mint(&mut f2[w * i..w * (i + 1)], &p2[i]);
    }
    // Forward NTT
    let mut ntt1 = vec![0; w * size];
    mzp.ntt(&f1, &mut ntt1, 0, logsize, true);
    let mut ntt2 = f1;
    mzp.ntt(&f2, &mut ntt2, 0, logsize, true);
    // Pointwise multiply
    for i in 0..size {
        mzp.mul(&mut ntt1[w * i..w * (i + 1)], &ntt2[w * i..w * (i + 1)]);
    }
    // Inverse NTT
    let mut f12 = ntt2;
    mzp.ntt(&ntt1, &mut f12, 0, logsize, false);
    // Extract result
    res.fill(MInt::default());
    for i in offset..min(size, offset + res.len()) {
        let idx = i - offset;
        res[idx] = mzp.redc(&f12[w * i..w * (i + 1)]);
    }
}

// Actual implementation of convolve_modn for a specific FFT size.
fn _convolve_modn<const N: usize>(
    zn: &ZmodN,
    size: usize,
    logpack: u32,
    stride: usize,
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
    if stride == 0 {
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
        let mask = (1 << logpack) - 1;
        let mut vp = vec![FInt::<N>::default(); size >> logpack];
        let mut vq = vec![FInt::<N>::default(); size >> logpack];
        // Inputs map 1-to-1 to FFT inputs.
        for i in 0..p.len() {
            let j = i & mask;
            vp[i >> logpack].0[stride * j..stride * j + msize].copy_from_slice(&p[i].0[..]);
        }
        for i in 0..q.len() {
            let j = i & mask;
            vq[i >> logpack].0[stride * j..stride * j + msize].copy_from_slice(&q[i].0[..]);
        }
        let vpq = mulfft(&vp, &vq);
        // Each output maps to N/16-1 coefficients (degree N/16-2)
        res.fill(MInt::default());
        for i in 0..vpq.len() {
            for j in 0..((2 << logpack) - 1) {
                let idx = (i << logpack) + j;
                let idx = if offset <= idx && idx < offset + res.len() {
                    idx - offset
                } else {
                    continue;
                };
                let digits = &vpq[i].0[stride * j..stride * (j + 1)];
                res[idx] = zn.add(res[idx], zn.redc_large(digits));
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
    fft(p1, &mut fp1, 0, k, true);
    fft(p2, &mut fp2, 0, k, true);
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
///
/// It is always assumed to be normalized either `x[N] == 0`
/// or `x[N] == 1` and `x[i] == 0` for i < N.
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

fn _mul_slice(z: &mut [u64], x: &[u64], w: u64) {
    let mut carry = 0;
    for i in 0..z.len() {
        unsafe {
            let zi = z[..].get_unchecked_mut(i);
            let xi = x[..].get_unchecked(i);
            let mut zw = (*xi as u128) * (w as u128);
            zw += carry as u128;
            *zi = zw as u64;
            carry = (zw >> 64) as u64;
        }
    }
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
        if s == 0 {
            return;
        }
        // Small shift: shl(x, s) = (x << s) - (x >> (64N-s))
        // Large shift: shl(x, s) = - shl(x, s - 64N)
        //                        = (x >> (128N-s)) - (x << (s-64N))

        // Shift whole words.
        let sw = s as usize / 64;
        if sw == 0 {
            // nothing to do
        } else if sw < N {
            let mut zhi = [0; N];
            zhi[0..sw].copy_from_slice(&self.0[N - sw..N]);
            self.0.copy_within(0..N - sw, sw);
            // subtract zhi
            if zhi[0] != 0 && self.0[sw] > 0 {
                // common case without carries:
                // fill with !(zhi-1) and subtract one for carry
                zhi[0] -= 1;
                for i in 0..sw {
                    self.0[i] = !zhi[i];
                }
                self.0[sw] -= 1;
            } else {
                self.0[0..sw].fill(0);
                let c = _sub_slices(&mut self.0, &zhi);
                if c == 1 {
                    self.add_small(1);
                }
            }
        } else if sw == N {
            // negate: -x = not(x) + 2
            for i in 0..N {
                self.0[i] = !self.0[i]
            }
            self.add_small(2);
        } else {
            let swhi = sw - N;
            let mut zlo = [0; N];
            zlo[swhi..].copy_from_slice(&self.0[0..N - swhi]);
            self.0.copy_within(N - swhi.., 0);
            if zlo[swhi] != 0 && !self.0[0] != 0 {
                // common case without carries
                zlo[swhi] -= 1;
                for i in swhi..N {
                    self.0[i] = !zlo[i];
                }
                self.0[0] += 1; // add -2^64N == 1
            } else {
                self.0[swhi..].fill(0);
                let c = _sub_slices(&mut self.0, &zlo);
                if c == 1 {
                    self.add_small(1);
                }
            }
        }
        // Small shift
        let sb = s % 64;
        if sb > 0 {
            let mut carry = 0u64;
            for i in 0..N {
                unsafe {
                    let xi = self.0.get_unchecked_mut(i);
                    let wlo = (*xi << sb) | carry;
                    carry = *xi >> (64 - sb);
                    *xi = wlo;
                }
            }
            self.1 = (self.1 << sb) | carry;
        }
        self.reduce();
    }

    fn shr(&mut self, s: u32) {
        if s == 0 {
            return;
        }
        self.shl(128 * N as u32 - s);
    }
}

/// All 59-bit primes that can be written as (a << 49) + 1
/// They are such that 2p does not overflow a u64.
/// The negated inverse of p modulo 2^64 is p-2
/// because (p-1)² = p(p-2)+1 = 0.
///
/// Since p = 1 mod 2^k for small k,
/// the inverse of 2^k mod p is (1-p)/2^k
///
/// The size of primes is such that Sum(k[i] * CRT[i]) < Sum(primes[i] * P)
/// is always less than 2^64 * P where P = product(primes)
///
/// 25 primes are enough for 720-bit moduli.
/// FFTs of size 2^32 should be enough for anybody.
const NTT_PRIMES: &[(u64, u64)] = &[
    // (p, element of order 2^32)
    (0x418000000000001, 0x20fd3b8a794b05),
    (0x426000000000001, 0x4084d82734bec34),
    (0x438000000000001, 0x579a4b725f365f),
    (0x462000000000001, 0x23651b8e1c71b4c),
    (0x508000000000001, 0x1a48f116441cc57),
    (0x528000000000001, 0x3cb61e87e38117d),
    (0x534000000000001, 0x1974562d8fd5c5d),
    (0x54c000000000001, 0x21f1eda46470a9d),
    (0x594000000000001, 0x179d1af7574c097),
    (0x598000000000001, 0x1d1c1fc5e7b3d60),
    (0x59e000000000001, 0x2d57ea1b0a0e112),
    (0x5d0000000000001, 0xdd7e419ce8fb03),
    (0x5da000000000001, 0x3324d5abb734d7e),
    (0x5e2000000000001, 0x592118fb16fb9bd),
    (0x62e000000000001, 0x54a898ba0a1aa32),
    (0x65a000000000001, 0x64861cf6b565d35),
    (0x660000000000001, 0x2bd2a0ea379229a),
    (0x67c000000000001, 0x4b387afa00fb99f),
    (0x6d8000000000001, 0xfe882fd3b98e8c),
    (0x6fc000000000001, 0x445172f17c94689),
    (0x712000000000001, 0x62e45fa11c058a3),
    (0x718000000000001, 0x1a94877d1f5b919),
    (0x744000000000001, 0x4e743f5672da4f),
    (0x774000000000001, 0x68d1f0d4eb9df05),
    (0x7a4000000000001, 0x58d7e6c4064ea76),
    (0x7da000000000001, 0x7344ff51d92abdf),
];

use bnum::types::U2048;

// A residue number system for bounded size computations.
// We select w NTT friendly primes such that 62k is more than the
// maximal expected integer size.
//
// Roots of unity are precomputed for a FFT of size 2^k.
//
// Values (which are themselves in Montgomery form mod n)
// are represented as length w slices [u64], in Montgomery form.
#[doc(hidden)]
#[derive(Clone)]
pub struct MultiZmodP<'a> {
    // primes.len() == w
    primes: &'a [(u64, u64)],
    // roots[i].len() == w << i for i = 1..k
    // This uses 2x the space of 2^k roots of unity.
    roots: Vec<Box<[u64]>>,
    // rpowers[i][j] = R^(j+1) mod p[i]
    rpowers: Vec<Vec<u64>>,
    // CRT coefficients:
    // pinv[i] = (P/pi)^-1 mod pi
    // crt[i] = pinv[i] * (P/pi)
    // where P is the product of selected primes.
    crt: Vec<U2048>,
    pprod: U2048,
    w: usize,
    k: u32,
    zn: &'a ZmodN,
}

impl<'a> MultiZmodP<'a> {
    pub fn new(zn: &'a ZmodN, logsize: u32) -> Self {
        // We need 2 logn + k bits.
        let need = 2 * zn.n.bits() + logsize;
        // 58w > 2 logn + k
        let w = need as usize / 58 + 1;
        let primes = &NTT_PRIMES[..w];
        assert!(w as u128 * (primes[w - 1].0 as u128) < 1 << 64);
        // Compute R^2 and R^3 = 2^192 mod pi
        let mut rpowers = vec![];
        for i in 0..w {
            let pi = primes[i].0;
            let r = (1_u128 << 64) % (pi as u128);
            let r2 = (r * r) % (pi as u128);
            let mut rs = vec![r as u64, r2 as u64];
            let mut rj = r2 as u64;
            for _ in 0..w {
                rj = arith_montgomery::mg_mul(pi, pi - 2, rj, r2 as u64);
                rs.push(rj);
            }
            rpowers.push(rs);
        }
        // Compute CRT coefficients. This is the same code
        // as siqs::select_siqs_factors.
        let mut crt = vec![U2048::default(); w];
        let mut pprod = U2048::from(1_u64);
        for i in 0..w {
            // Compute product(pj for j != i) modulo pi and modulo n.
            let mut mi = 1_u128;
            let pi = primes[i].0 as u128;
            let mut m = U2048::from(1_u64);
            for j in 0..w {
                if j == i {
                    continue;
                }
                let pj = primes[j].0;
                mi = (mi * pj as u128) % pi;
                m *= U2048::from(pj);
            }
            // Invert modulo pi.
            let mi_inv = arith::inv_mod64(mi as u64, pi as u64).unwrap();
            crt[i] = m * U2048::from(mi_inv);
            pprod *= U2048::from(pi);
        }
        // Prepare roots of unity (in Montgomery form).
        let mut ωs = vec![0_u64; w];
        let mut roots = vec![0u64; w << logsize];
        for i in 0..w {
            roots[i] = rpowers[i][0];
            // Build a root of order 2^logsize from an order 2^32 element.
            let pi = primes[i].0;
            let mut ω = primes[i].1 as u128;
            for _ in logsize..32 {
                ω = (ω * ω) % (pi as u128);
            }
            ωs[i] = arith_montgomery::mg_mul(pi, pi - 2, ω as u64, rpowers[i][1]);
        }
        for j in 1..(1 << logsize) {
            for i in 0..w {
                let pi = primes[i].0;
                roots[j * w + i] =
                    arith_montgomery::mg_mul(pi, pi - 2, roots[(j - 1) * w + i], ωs[i]);
            }
        }
        // Precompute packed arrays of roots for each power of 2.
        let mut roots_packed = vec![];
        for log in 0..=logsize {
            let mut v = vec![];
            for idx in 0..(1 << log) {
                let offset = w * idx << (logsize - log);
                v.extend_from_slice(&roots[offset..offset + w]);
            }
            assert!(v.len() == w << log);
            roots_packed.push(v.into_boxed_slice());
        }
        // Sanity check
        for i in 0..w {
            let pi = primes[i].0;
            let ω_pow_n =
                arith_montgomery::mg_mul(pi, pi - 2, roots[((1 << logsize) - 1) * w + i], ωs[i]);
            debug_assert!(
                arith_montgomery::mg_redc(pi, pi - 2, ω_pow_n as u128) == 1,
                "ω={} ω^(n-1)={} p={pi}",
                arith_montgomery::mg_redc(pi, pi - 2, ωs[i] as u128),
                arith_montgomery::mg_redc(pi, pi - 2, roots[((1 << logsize) - 1) * w + i] as u128)
            );
        }
        MultiZmodP {
            primes,
            roots: roots_packed,
            crt,
            pprod,
            rpowers,
            w,
            k: logsize,
            zn,
        }
    }

    // Converts a reduced Z/nZ element to a slice of residues.
    pub fn from_mint(&self, z: &mut [u64], x: &MInt) {
        let sz = self.zn.words();
        assert!(sz <= 8); // help inlining
        for i in 0..self.w {
            unsafe {
                let pi = self.primes.get_unchecked(i).0;
                // Compute x = sum(x[j] R^j mod pi)
                let ri = &self.rpowers.get_unchecked(i)[..];
                let mut zi = x.0[0] as u128;
                debug_assert!(sz >= x.0.len() || x.0[sz] == 0);
                for j in 1..sz {
                    let xj = *x.0.get_unchecked(j);
                    let rij = *ri.get_unchecked(j - 1);
                    zi += xj as u128 * rij as u128;
                }
                // Reduce mod pi.
                let mut zi_red = arith_montgomery::mg_mul(pi, pi - 2, zi as u64, ri[1])
                    + arith_montgomery::mg_mul(pi, pi - 2, (zi >> 64) as u64, ri[2]);
                if zi_red > pi {
                    zi_red -= pi;
                }
                z[i] = zi_red;
            }
        }
    }

    pub fn to_mint(&self, x: &[u64]) -> MInt {
        const WORDS: usize = U2048::BITS as usize / 64;
        let mut res = [0; WORDS];
        self._crt(&mut res, x);
        // m = res/R
        let m = self.zn.redc_large(&res[..self.w + 1]);
        // m = (res/R)*R
        let m = self.zn.from_int(crate::Uint::from(m));
        m
    }

    // Returns x/R (especially when x = aR*bR)
    pub fn redc(&self, x: &[u64]) -> MInt {
        const WORDS: usize = U2048::BITS as usize / 64;
        let mut res = [0; WORDS];
        self._crt(&mut res, x);
        self.zn.redc_large(&res[..self.w + 1])
    }

    pub fn _crt(&self, res: &mut [u64], x: &[u64]) {
        debug_assert!(x.len() == self.w);
        // Reconstruct number as (sum xi CRT[i] - kP)
        // We compute modulo n but kP must be subtracted.
        // k = sum(xi pinv[i] // pi)
        // where pinv[i] = (P/pi)^-1 mod pi
        let w = self.w;
        debug_assert!(res.len() >= w + 1);
        const WORDS: usize = U2048::BITS as usize / 64;
        let mut buf = [0; WORDS];
        let mut xs = [0; NTT_PRIMES.len()];
        for i in 0..w {
            let pi = self.primes[i].0;
            xs[i] = arith_montgomery::mg_redc(pi, pi - 2, x[i] as u128);
        }
        let mut carry = 0;
        for i in 0..w {
            let mut z = carry as u128;
            for j in 0..w {
                z += xs[j] as u128 * self.crt[j].digits()[i] as u128;
            }
            res[i] = z as u64;
            carry = (z >> 64) as u64;
        }
        res[w] = carry;
        // Reduce modulo pprod.
        // Product of primes has size ~59w bits, it has w or w-1 words
        let pdigits = self.pprod.digits();
        let plen: usize = {
            let mut i = w;
            loop {
                if pdigits[i] != 0 {
                    break i;
                }
                i -= 1;
            }
        };
        debug_assert!(pdigits[w + 1] == 0);
        // Use top 64 bits to compute an approximate quotient.
        // This quotient must be exact up to 1 unit.
        // Because the quotient is at most 63 bits.
        fn topwords(r: &[u64], p: &[u64], plen: usize) -> (u128, u64) {
            let pad = p[plen].leading_zeros();
            let mut rtop = ((r[plen + 1] as u128) << (64 + pad)) + ((r[plen] as u128) << pad);
            let mut ptop = p[plen] << pad;
            if pad > 0 {
                rtop += (r[plen - 1] as u128) >> (64 - pad);
                ptop += p[plen - 1] >> (64 - pad);
            }
            (rtop, ptop)
        }
        let qapprox: u128 = if plen == 0 {
            let restop = ((res[plen + 1] as u128) << 64) + res[plen] as u128;
            restop / (pdigits[0] as u128)
        } else {
            let (restop, ptop) = topwords(res, pdigits, plen);
            restop / (ptop as u128 + 1)
        };
        debug_assert!(qapprox >> 64 == 0);
        _mul_slice(&mut buf[..w + 1], &pdigits[..w + 1], qapprox as u64);
        _sub_slices(&mut res[..w + 1], &buf[..w + 1]);
        // Values must be less than P/2 in all cases, so the top word must be less than pprod.
        if plen > 0 {
            loop {
                let (restop, ptop) = topwords(res, pdigits, plen);
                if restop >= ptop as u128 {
                    _sub_slices(&mut res[..w + 1], &pdigits[..w + 1]);
                } else {
                    break;
                }
            }
        }
        debug_assert!({
            let mut r = [0; WORDS];
            r[..w + 1].copy_from_slice(&res[..w + 1]);
            U2048::from_digits(r) < self.pprod
        });
    }

    // Arithmetic for FFT:
    // - butterfly (x,y) -> (x+y,x-y)
    // - multiply by ω^i
    // - divide by 2^l

    fn addsub(&self, a: &mut [u64], b: &mut [u64], x: &[u64], y: &[u64]) {
        debug_assert!(x.len() == self.w);
        debug_assert!(y.len() == self.w);
        for i in 0..self.w {
            unsafe {
                let xi = *x.get_unchecked(i);
                let yi = *y.get_unchecked(i);
                let pi = self.primes.get_unchecked(i).0;
                let add = if xi + yi >= pi { xi + yi - pi } else { xi + yi };
                let sub = if xi >= yi { xi - yi } else { xi + pi - yi };
                *a.get_unchecked_mut(i) = add;
                *b.get_unchecked_mut(i) = sub;
            }
        }
    }

    fn addsub_inplace(&self, x: &mut [u64], y: &mut [u64]) {
        debug_assert!(x.len() == self.w);
        debug_assert!(y.len() == self.w);
        for i in 0..self.w {
            unsafe {
                let xi_ref = x.get_unchecked_mut(i);
                let yi_ref = y.get_unchecked_mut(i);
                let (xi, yi) = (*xi_ref, *yi_ref);
                let pi = self.primes.get_unchecked(i).0;
                let add = if xi + yi >= pi { xi + yi - pi } else { xi + yi };
                let sub = if xi >= yi { xi - yi } else { xi + pi - yi };
                *xi_ref = add;
                *yi_ref = sub;
            }
        }
    }

    // Multiply x in place, by y.
    fn mul(&self, x: &mut [u64], y: &[u64]) {
        debug_assert!(x.len() == self.w);
        debug_assert!(y.len() == self.w);
        for i in 0..self.w {
            unsafe {
                let xi = x.get_unchecked_mut(i);
                let yi = y.get_unchecked(i);
                let pi = self.primes.get_unchecked(i).0;
                *xi = arith_montgomery::mg_mul(pi, pi - 2, *xi, *yi);
            }
        }
    }

    // Multiply x in place, by ω^i where ω is a 2^k primitive root of unity.
    fn twiddle(&self, x: &mut [u64], i: usize, k: u32) {
        debug_assert!(x.len() == self.w);
        let i = i & ((1 << k) - 1);
        self.mul(x, &self.roots[k as usize][self.w * i..self.w * (i + 1)]);
    }

    // Divide x by 2^k
    // The inverse of 2^k mod p is (1-p)/2^k
    fn div_pow2(&self, x: &mut [u64], k: u32) {
        for i in 0..self.w {
            unsafe {
                let xi = x.get_unchecked_mut(i);
                let pi = self.primes.get_unchecked(i).0;
                // xR -> REDC(xR * R/2^k)
                *xi = arith_montgomery::mg_redc(pi, pi - 2, (*xi as u128) << (64 - k));
            }
        }
    }

    fn ntt(&self, src: &[u64], dst: &mut [u64], depth: u32, k: u32, fwd: bool) {
        // At each stage, transform src[i<<depth] for i in 0..2^k into dst.
        assert!(dst.len() == self.w << k);
        let w = self.w;
        if k == 0 {
            dst.copy_from_slice(src);
            if !fwd {
                // For inverse FFT, divide by 2^depth.
                self.div_pow2(&mut dst[0..w], depth);
            }
            return;
        } else if k == 1 {
            // dst = (src0 + src1, src0 - src1)
            let (dst1, dst2) = dst.split_at_mut(w);
            self.addsub(dst1, dst2, &src[..w], &src[w << depth..(w << depth) + w]);
            if !fwd {
                // For inverse FFT, divide by 2^(depth+1).
                self.div_pow2(dst1, depth + 1);
                self.div_pow2(dst2, depth + 1);
            }
            return;
        }
        // Transform odd/even indices
        let half = 1 << (k - 1);
        self.ntt(src, &mut dst[..w * half], depth + 1, k - 1, fwd);
        self.ntt(
            &src[w << depth..],
            &mut dst[w * half..],
            depth + 1,
            k - 1,
            fwd,
        );
        // Twiddle
        let (dst1, dst2) = dst.split_at_mut(w * half);
        for idx in 0..half {
            // Compute X + ω^idx Y, X - ω^idx Y where ω is the 2^k primitive root of unity.
            if fwd {
                self.twiddle(&mut dst2[w * idx..w * idx + w], idx, k);
            } else {
                self.twiddle(&mut dst2[w * idx..w * idx + w], (1 << k) - idx, k);
            }
            self.addsub_inplace(
                &mut dst1[w * idx..w * idx + w],
                &mut dst2[w * idx..w * idx + w],
            );
        }
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

#[test]
fn test_multimodn() {
    use crate::Uint;
    use std::str::FromStr;

    // Tiny modulus
    let n = Uint::from_str("29396397").unwrap();
    let zn = ZmodN::new(n);
    let mzn = MultiZmodP::new(&zn, 3);
    assert!(mzn.w == 1);
    for i in 1..1000_u64 {
        let x = zn.from_int((12345 * i).into());
        let mut mx = vec![0; mzn.w];
        mzn.from_mint(&mut mx, &x);
        assert_eq!(mzn.to_mint(&mx), x);
    }

    // Large modulus
    let n = Uint::from_str("2953951639731214343967989360202131868064542471002037986749").unwrap();
    let zn = ZmodN::new(n);
    let mzn = MultiZmodP::new(&zn, 3);
    eprintln!("Using {} primes for {}-bit modulus", mzn.w, n.bits());
    for i in 1..1000_u64 {
        let x = zn.from_int((12345 * i).into());
        let mut mx = vec![0; mzn.w];
        mzn.from_mint(&mut mx, &x);
        assert_eq!(mzn.to_mint(&mx), x);

        // Compute xxR = REDC(xR*xR)
        let mut mxx = mx.clone();
        mzn.mul(&mut mxx, &mx);
        let xx = zn.from_int((12345 * 12345 * i * i).into());
        assert_eq!(mzn.redc(&mxx), xx);
    }
}

#[test]
fn test_multintt() {
    use crate::Uint;
    use std::str::FromStr;

    let n = Uint::from_str("2953951639731214343967989360202131868064542471002037986749").unwrap();
    let zn = ZmodN::new(n);
    let zr = MultiZmodP::new(&zn, 16);
    for w in [4, 16, 256, 512, 1024, 2048] {
        eprintln!("Test convolve {w}*{w} => {}", 2 * w);
        let mut v1 = vec![MInt::default(); 2 * w];
        let mut v2 = vec![MInt::default(); 2 * w];
        for idx in 0..w {
            v1[idx] = zn.from_int((123 * idx as u64 + 456).into());
            v2[idx] = zn.from_int((456 * idx as u64 + 789).into());
        }

        // NTT round trip
        let logsize = (2 * w).trailing_zeros();
        let mzp = MultiZmodP::new(&zn, logsize);
        let mw = mzp.w;
        let mut f1 = vec![0; 2 * w * mw];
        for i in 0..v1.len() {
            mzp.from_mint(&mut f1[mw * i..mw * (i + 1)], &v1[i]);
        }
        let mut t1 = vec![0; 2 * w * mw];
        let mut invt1 = vec![0; 2 * w * mw];
        mzp.ntt(&f1, &mut t1, 0, logsize, true);
        mzp.ntt(&t1, &mut invt1, 0, logsize, false);
        for idx in 0..w {
            eprintln!(
                "index {idx}: {:?} == {:?}",
                &f1[mw * idx..mw * (idx + 1)],
                &invt1[mw * idx..mw * (idx + 1)]
            );
            assert_eq!(
                &f1[mw * idx..mw * (idx + 1)],
                &invt1[mw * idx..mw * (idx + 1)],
                "index {idx}"
            );
        }

        // NTT convolution
        let mut v = vec![MInt::default(); 2 * w];
        convolve_modn_ntt(&zr, 2 * w, &v1, &v2, &mut v, 0);
        for i in 0..2 * w {
            let mut expect = 0;
            for idx1 in 0..=i {
                let x1 = if idx1 < w { 123 * idx1 as u64 + 456 } else { 0 };
                let x2 = if i - idx1 < w {
                    456 * (i - idx1) as u64 + 789
                } else {
                    0
                };
                expect += x1 * x2;
            }
            assert_eq!(
                zn.from_int(expect.into()),
                v[i],
                "w={w} index {i} expect={expect} got={}",
                zn.to_int(v[i])
            );
        }
    }
}
