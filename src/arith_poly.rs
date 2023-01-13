// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Implementation of polynomial arithmetic over Z/nZ
//!
//! The sole purpose of this library is to provide subquadratic
//! multipoint evaluation for ECM and Pollard P-1 implementations,
//! using scaled remainder trees as defined by D.J. Bernstein.
//! <https://cr.yp.to/arith/scaledmod-20040820.pdf>
//!
//! Products use the Karatsuba method and become quickly more effective
//! than quadratic multiplication. However the overhead of computing
//! product and remainder trees is large compared to simple polynomial
//! evaluation.

use std::cmp::max;
use std::ops::{AddAssign, MulAssign, SubAssign};

use crate::arith_fft::convolve_modn;
use crate::arith_montgomery::{MInt, ZmodN};

// Fast polynomial evaluation:
//
// We want to compute P at points b1 ... bn
//
// For this compute the Laurent series F = P/Q = f(x) + g(1/x) + o(1/x^n)
// Then P(bj) is the residue of P/(X-bj) at bj
// Recursively multiply by (X-bj) to obtain the leaves P/(X-bj)
// Each multiplication is realized by a recursive middle product.

#[derive(Clone)]
pub struct Poly<'a> {
    zn: &'a ZmodN,
    pub c: Vec<MInt>,
}

impl<'a> AddAssign<&Poly<'a>> for Poly<'a> {
    fn add_assign(&mut self, rhs: &Poly<'a>) {
        if self.c.len() < rhs.c.len() {
            self.c.resize(rhs.c.len(), MInt::default());
        }
        Self::_add(&self.zn, &mut self.c, &rhs.c);
    }
}

impl<'a> SubAssign<&Poly<'a>> for Poly<'a> {
    fn sub_assign(&mut self, rhs: &Poly<'a>) {
        if self.c.len() < rhs.c.len() {
            self.c.resize(rhs.c.len(), MInt::default());
        }
        Self::_sub(&self.zn, &mut self.c, &rhs.c);
    }
}

impl<'a> MulAssign<&MInt> for Poly<'a> {
    fn mul_assign(&mut self, rhs: &MInt) {
        let zn = &self.zn;
        for i in 0..self.c.len() {
            self.c[i] = zn.mul(self.c[i], *rhs);
        }
    }
}

const USE_FFT: bool = true;
const FFT_THRESHOLD: usize = 64;

impl<'a> Poly<'a> {
    pub fn new(zn: &'a ZmodN, coefs: Vec<MInt>) -> Self {
        Poly { zn, c: coefs }
    }

    // Computes polynomial product(x-r for r in roots)
    pub fn _product_tree(zn: &'a ZmodN, roots: &[MInt]) -> Vec<Vec<MInt>> {
        // Smallest power of two >= len(roots).
        let logn = usize::BITS - usize::leading_zeros(roots.len() - 1);
        let n = 1_usize << logn;
        let mut layers = vec![];
        let mut buf1 = vec![MInt::default(); 2 * n];
        let mut buf2 = vec![MInt::default(); 2 * n];
        for i in 0..n {
            if i < roots.len() {
                // polynomial x-r
                buf1[2 * i] = zn.sub(MInt::default(), roots[i]);
                buf1[2 * i + 1] = zn.one();
            } else {
                // pad with x so that all polynomials have degree 1
                buf1[2 * i + 1] = zn.one();
            }
        }
        layers.push(buf1.clone());
        // Repeatedly multiply pairs of polynomials in buf1 into buf2.
        // To build layer i, multiply degree 2^{i-1} polynomials from previous layer.
        let mut tmp = vec![MInt::default(); 6 * n];
        for i in 1..=logn {
            for j in 0..(1 << (logn - i)) {
                // Indices from 0 to 2<<logn
                // At layer i, the degree of polynomials to be multiplied is 2^(i-1)
                // Polynomials are monic so no need to multiply by the leading "1".
                let deg = 1 << (i - 1);
                let idx1 = (2 * j) << i;
                let idx2 = (2 * j + 1) << i;
                //let idx3 = (2 * j + 2) << i;
                Self::_longmul(
                    zn,
                    &mut buf2[idx1..idx1 + 2 * deg],
                    &buf1[idx1..idx1 + deg],
                    &buf1[idx2..idx2 + deg],
                    &mut tmp,
                );
                // Add x^d * P2 and x^d * P1
                Self::_add(
                    zn,
                    &mut buf2[idx1 + deg..idx1 + 2 * deg],
                    &buf1[idx1..idx1 + deg],
                );
                Self::_add(
                    zn,
                    &mut buf2[idx1 + deg..idx1 + 2 * deg],
                    &buf1[idx2..idx2 + deg],
                );
                // Set leading coefficient
                buf2[idx1 + 2 * deg] = zn.one();
            }
            debug_assert!(buf2[1 << i] == zn.one());
            std::mem::swap(&mut buf1, &mut buf2);
            layers.push(buf1.clone());
        }
        assert_eq!(layers.len() as u32, logn + 1);
        layers
    }

    pub fn from_roots(zn: &'a ZmodN, roots: Vec<MInt>) -> Self {
        let deg = roots.len();
        let layers = Self::_product_tree(zn, &roots);
        let product = layers.last().unwrap();
        // Last layer has degree n.
        let n = product.len() / 2;
        Poly {
            zn,
            c: product[n - deg..n + 1].to_vec(),
        }
    }

    // Evaluate a polynomial at a set of points. The set of points
    // is usually larger (4x) than the polynomial degree.
    pub fn multi_eval(&self, a: Vec<MInt>) -> Vec<MInt> {
        let plen = self.c.len();
        let alen = a.len();
        let logn = usize::BITS - usize::leading_zeros(plen + 1);
        let n = 1_usize << logn;
        let chunks = (alen - 1) / n + 1;
        let chunklen = a.len() / chunks + 1;
        let mut vals = vec![];
        for chk in a.chunks(chunklen) {
            //eprintln!("chunk size {}", chk.len());
            vals.append(&mut self._multi_eval(chk));
        }
        assert_eq!(vals.len(), alen);
        vals
    }

    pub fn _multi_eval(&self, a: &[MInt]) -> Vec<MInt> {
        // Compute P(x)/product(x-ai) as a power series in 1/x
        // assume that degree(P) <= len(a)
        // Let t = 1/x
        // P(x)/product(x-ai) = revP(t) / product(1-ai t) has residue P(ai) at ai
        let zn = &self.zn;
        //       assert!(self.c.len() <= 2 * n);
        // Compute the product tree of x - ai as above.
        let layers = Self::_product_tree(zn, a);
        let n = layers[layers.len() - 1].len() / 2;
        let logn = n.trailing_zeros();
        assert_eq!(layers.len() as u32, logn + 1);

        // The last layer is Q=product(x-ai) of degree n
        // To get P/Q as a power series in 1/x compute revP/revQ
        // where rev(F) = t^n F(1/t)
        let q = layers.last().unwrap();
        let mut revp = vec![MInt::default(); n + 1];
        let mut revq = vec![MInt::default(); n + 1];
        for i in 0..=n {
            if i < self.c.len() {
                revp[n - i] = self.c[i];
            }
            revq[n - i] = q[i];
        }
        // If Q is monic, revQ is 1 + O(t)
        debug_assert!(revq[0] == zn.one());

        let mut node = vec![MInt::default(); 2 * n];
        let mut dst = vec![MInt::default(); 2 * n];
        let mut tmp = vec![MInt::default(); 10 * n];
        Self::_div_mod_xn(zn, &mut dst, &revp[..n + 1], &revq[..n + 1], &mut tmp);
        // Rewrite the power series:
        // P/Q = p0 + p1/x + ... + p[n]/x^n
        // P/Q = p[0] + (p[1] x^n-1 + ... p[n]) / x^n
        // The constant can be dropped for the residue computation.
        for i in 0..=n {
            node[i] = dst[n - i];
        }
        // For each layer:
        // express P/Q1Q2 as polynomial / 2^(k+1)
        // multiply by Q1 (degree 2^k)
        // keep coefficients 1/x ... 1/x^k => this is the same as the middle product!
        // at layer N-i
        // 2^i polynomials Q
        // length of Q is 2^(N+1-i)
        // degree of Q is 2^(N-i)
        for i in 1..=logn {
            let layer = &layers[(logn - i) as usize];
            for j in 0..(1 << (i - 1)) {
                // Location of parent F
                let idx1 = (2 * j) << (logn + 1 - i);
                let idx2 = (2 * j + 1) << (logn + 1 - i);
                let idx3 = (2 * j + 2) << (logn + 1 - i);
                // Split node=P/Q1Q2 into P/Q1 and P/Q2
                // node[i1..i3] => node[i1..i2] and node[i2..i3]
                let q1 = &layer[idx1..idx2];
                let q2 = &layer[idx2..idx3];
                let degq = 1 << (logn - i);
                // always ignore the constant term to get an array of length 2k-1
                assert!(idx1 + degq + 1 <= idx2);
                debug_assert!(q1[degq] == zn.one());
                debug_assert!(q2[degq] == zn.one());
                // on the left, multiply by Q2 to get P/Q1
                // Beware that the convention is slightly different than middlemul:
                // F = P/Q1Q2 = a[2k-1] / x + ... + a[0] / x^2k
                // Q2 = x^k + b[k-1] x^k-1 + ... + b[0]
                // middlemul(F/x, Q2) = coefficients of F*(Q2-x^k) for 1/x .. 1/x^k
                Self::_middlemul(
                    zn,
                    &mut dst[idx1..idx1 + degq],
                    &node[idx1 + 1..idx1 + 2 * degq],
                    &q2[..degq],
                    &mut tmp,
                );
                Self::_middlemul(
                    zn,
                    &mut dst[idx2..idx2 + degq],
                    &node[idx1 + 1..idx1 + 2 * degq],
                    &q1[..degq],
                    &mut tmp,
                );
                // Re-add terms for the leading x^k of Q2.
                // Beware to use node[idx1] before writing to it.
                for i in 0..degq {
                    node[idx2 + i] = zn.add(&node[idx1 + i], &dst[idx2 + i]);
                    node[idx1 + i] = zn.add(&node[idx1 + i], &dst[idx1 + i]);
                }
            }
        }
        // Now each leaf contains the results
        let mut vals = vec![];
        for i in 0..a.len() {
            // a + b/x => residue=b
            vals.push(node[2 * i]);
        }
        vals
    }

    fn _add(zn: &ZmodN, z: &mut [MInt], x: &[MInt]) {
        assert_eq!(z.len(), x.len());
        for i in 0..z.len() {
            z[i] = zn.add(&z[i], &x[i]);
        }
    }

    fn _sub(zn: &ZmodN, z: &mut [MInt], x: &[MInt]) {
        assert_eq!(z.len(), x.len());
        for i in 0..z.len() {
            z[i] = zn.sub(&z[i], &x[i]);
        }
    }

    fn _basic_mul(zn: &ZmodN, z: &mut [MInt], p: &[MInt], q: &[MInt]) {
        z[p.len() + q.len() - 1..].fill(MInt::default());
        for i in 0..p.len() {
            for j in 0..q.len() {
                if i == 0 || j + 1 == q.len() {
                    // first term
                    z[i + j] = zn.mul(&p[i], &q[j]);
                } else {
                    z[i + j] = zn.add(&z[i + j], &zn.mul(&p[i], &q[j]));
                }
            }
        }
    }

    /// Multiply 2 polynomials into a double length polynomial.
    /// The lengths of p,q are preferably both close to 2^k from below.
    fn _fft_longmul(zn: &ZmodN, z: &mut [MInt], p: &[MInt], q: &[MInt]) {
        // FIXME: reuse scratch buffer.
        let degp = p.len() - 1;
        let degq = q.len() - 1;
        let logsize = usize::BITS - usize::leading_zeros(max(degp, degq));
        convolve_modn(zn, 2 << logsize, p, q, z, 0);
    }

    /// Middle product of polynomials of degree 2N-2 and N-1
    /// using a length 2N FFT.
    /// The full product is:
    /// (x^3n-2 + ... + x^2n-1) + (x^2n-2 + ... + x^n-1) + (x^n-2 + ... 1)
    /// and low and high parts will wrap together during convolution.
    fn _fft_midmul(zn: &ZmodN, z: &mut [MInt], p: &[MInt], q: &[MInt]) {
        // FIXME: reuse scratch buffer.
        let qlen = q.len();
        assert!(qlen & (qlen - 1) == 0);
        assert!(p.len() == 2 * qlen - 1);
        let logsize = qlen.trailing_zeros();
        // Compute convolution product and extract
        // coefficients n-1 .. 2n-1
        convolve_modn(zn, 2 << logsize, p, q, z, qlen - 1);
    }

    fn _longmul(zn: &ZmodN, z: &mut [MInt], p: &[MInt], q: &[MInt], tmp: &mut [MInt]) {
        if USE_FFT && p.len() >= FFT_THRESHOLD {
            Self::_fft_longmul(zn, z, p, q)
        } else {
            Self::karatsuba(zn, z, p, q, tmp)
        }
    }

    // Computes the multiplication of polynomials p and q in array z,
    // using tmp as scratch buffer.
    fn karatsuba(zn: &ZmodN, z: &mut [MInt], p: &[MInt], q: &[MInt], tmp: &mut [MInt]) {
        if p.len() <= 20 && q.len() <= 20 {
            Self::_basic_mul(zn, z, p, q);
            return;
        }
        debug_assert!(z.len() >= p.len() + q.len());
        // Invariant: tmp has 3 times the length of p or q
        let half = (max(p.len(), q.len()) + 1) / 2;
        assert!(tmp.len() >= 4 * half);
        // Add:
        // plo*qlo
        //     (pl qh + ph ql)
        //          ph qh
        let (plo, phi) = (&p[..half], &p[half..]);
        let (qlo, qhi) = (&q[..half], &q[half..]);

        let (tmplo, tmphi) = tmp.split_at_mut(2 * half);
        debug_assert!(2 * half >= p.len());
        debug_assert!(2 * half >= q.len());
        // middle first: plo*qhi+phi*qlo = (plo+phi)*(qlo+qhi) -plo qlo - phi qhi
        // put the sums in tmp[2h..3h] and tmp[3h..4h]
        // put the product in tmp[..2h] using z as scratch.
        // Because len(z) == 2l > 3half
        tmphi[..half].copy_from_slice(plo);
        Self::_add(zn, &mut tmphi[..phi.len()], phi);
        tmphi[half..2 * half].copy_from_slice(qlo);
        Self::_add(zn, &mut tmphi[half..half + qhi.len()], qhi);
        Self::karatsuba(
            zn,
            &mut tmplo[..2 * half],
            &tmphi[..half],
            &tmphi[half..2 * half],
            z,
        );
        // then low and high, put it in z[..n] and z[n..2n]
        // Use tmphi as scratch (len(tmphi) = 4half > 3half)
        Self::karatsuba(zn, &mut z[..2 * half], plo, qlo, tmphi);
        Self::karatsuba(zn, &mut z[2 * half..], phi, qhi, tmphi);
        let hilen = phi.len() + qhi.len() - 1;
        // then subtract and add middle
        Self::_sub(zn, &mut tmplo[..2 * half], &z[..2 * half]);
        Self::_sub(zn, &mut tmplo[..hilen], &z[2 * half..2 * half + hilen]);
        Self::_add(zn, &mut z[half..3 * half], &tmplo[..2 * half]);
        // Total cost K(n/2) + 4n ADD
    }

    /// Fast middle product
    ///
    /// Hanrot, Quercia, Zimmerman
    /// The Middle Product Algorithm, I.
    /// https://hal.inria.fr/inria-00071921/document
    fn _middlemul(zn: &ZmodN, z: &mut [MInt], p: &[MInt], q: &[MInt], tmp: &mut [MInt]) {
        // Page 9 of [Hanrot-Quercia-Zimmerman]
        // Karatsuba-like method to get coefficients [n-1..2n-1]
        // from a product where deg p = 2n-2, deg q = n-1
        // P = P3 P2 P1 P0
        // Q =       Q1 Q0
        // compute: a = middlemul(P2 P1 + P1 P0, Q1)
        //          b = middlemul(P2 P1, Q1 - Q0)
        //          c = middlemul(P3 P2 + P2 P1, Q0)
        // return a-b = middle(P2 P1, Q0) + middle(P1 P0, Q1)
        //        c+b = middle(P3 P2, Q0) + middle(P2 P1, Q1)
        // output size is n
        assert!(p.len() == 2 * q.len() - 1);
        if q.len() == 1 {
            z[0] = zn.mul(&p[0], &q[0]);
            return;
        } else if q.len() == 2 {
            // p = p2 p1 p0
            // q =    q1 q0
            // want (p2 q0 + p1 q1, p1 q0 + p0 q1)
            z[0] = zn.add(&zn.mul(&p[1], &q[0]), &zn.mul(&p[0], &q[1]));
            z[1] = zn.add(&zn.mul(&p[2], &q[0]), &zn.mul(&p[1], &q[1]));
            return;
        } else if USE_FFT && q.len() >= FFT_THRESHOLD {
            if q.len() & (q.len() - 1) == 0 {
                // exact power of 2
                Self::_fft_midmul(zn, z, p, q);
                return;
            } else if (q.len() - 1) & (q.len() - 2) == 0 {
                // Power series inversion creates inputs of size 2^k + 1
                // P = p[2n] X^2n + ... + p[0]
                // Q = q[n] X^n + ... + q[0]
                // and we want MP(P,Q) = r[2n] + ... + r[n]
                //
                // Instead compute a 2^k FFT:
                // P = p[2n] X^2n + P' X + p0
                // Q = Q' X + q0
                // PQ = ... + p[2n] q0 X^2n + P' Q' X^2 + q0 P' X + p0 Q' X + ...
                //
                // MP(P, Q) = MP(P', Q') X
                //          + p[2n] q[0] X^n
                //          + q0 P' X
                //          + p0 Q' X => only p0 qn X^n in middle product
                let n = q.len() - 1;
                // z[k+1] = sum(p[i+1] q[j+1] where i+j = k+n-1)
                Self::_fft_midmul(zn, &mut z[1..], &p[1..2 * n], &q[1..]);
                for i in n..=2 * n {
                    z[i - n] = zn.add(z[i - n], zn.mul(p[i], q[0]));
                }
                z[0] = zn.mul(p[0], q[n]);
                for i in 1..=n {
                    z[0] = zn.add(z[0], zn.mul(p[i], q[n - i]));
                }
                return;
            }
        }
        assert!(tmp.len() >= 2 * p.len());

        // Compute a and c
        let half = q.len() / 2; // q = 2 half or 2 half + 1
        let half_up = q.len() - half;
        let (tmplo, tmphi) = tmp.split_at_mut(p.len());
        // Compute P + x^half_up P (containing P3 P2 + P2 P1 and P2 P1 + P1 P0)
        tmplo[..p.len()].copy_from_slice(p);
        Self::_add(zn, &mut tmplo[..p.len() - half_up], &p[half_up..p.len()]);
        // Compute a and c (multiply by Q0) using tmphi as scratch
        Self::_middlemul(
            zn,
            &mut z[..half_up],
            &mut tmplo[..2 * half_up - 1],
            &q[half..], // length half_up
            tmphi,
        );
        Self::_middlemul(
            zn,
            &mut z[half_up..q.len()],
            &mut tmplo[half_up..p.len() - half_up], // length 2half-1
            &q[..half],
            tmphi,
        );

        let (tmp1, tmp2) = tmplo.split_at_mut(half_up);
        // Compute Q1-Q0 in tmp2 (aligned right)
        tmp2[..half_up].copy_from_slice(&q[half..]);
        Self::_sub(zn, &mut tmp2[half_up - half..half_up], &q[..half]);
        // Compute b in tmp1, len(tmplo) > 4half
        // Output length = half_up
        Self::_middlemul(
            zn,
            tmp1,
            &p[half_up..3 * half_up - 1],
            &tmp2[..half_up],
            tmphi,
        );
        Self::_sub(zn, &mut z[..half_up], &tmp1[..half_up]);
        Self::_add(zn, &mut z[half_up..q.len()], &tmp1[..half]);
    }

    /// Slow method for testing purposes.
    pub fn mul_basic(p: &'a Poly<'a>, q: &'a Poly<'a>) -> Poly<'a> {
        let mut z = vec![];
        z.resize(2 * p.c.len(), MInt::default());
        Self::_basic_mul(&p.zn, &mut z, &p.c, &q.c);
        Poly { zn: &p.zn, c: z }
    }

    /// Only intended for tests and benchmarks.
    pub fn mul_karatsuba(p: &'a Poly<'a>, q: &'a Poly<'a>) -> Poly<'a> {
        let mut z = vec![];
        let mut tmp = vec![];
        // FIXME: fix unbalanced inputs.
        // Assumes similar degrees (all degrees more than max(deg p, deg q)/2)
        z.resize(2 * p.c.len(), MInt::default());
        tmp.resize(6 * p.c.len(), MInt::default());
        Self::karatsuba(&p.zn, &mut z, &p.c, &q.c, &mut tmp);
        Poly { zn: &p.zn, c: z }
    }

    /// Only intended for tests and benchmarks.
    pub fn mul_fft(p: &'a Poly<'a>, q: &'a Poly<'a>) -> Poly<'a> {
        let mut pq = vec![MInt::default(); p.c.len() + q.c.len() - 1];
        Self::_fft_longmul(&p.zn, &mut pq, &p.c, &q.c);
        Poly { zn: &p.zn, c: pq }
    }

    pub fn middlemul(p: &'a Poly<'a>, q: &Poly<'a>) -> Poly<'a> {
        let n = q.c.len();
        assert!(p.c.len() == 2 * n - 1);
        let mut vp = p.c.clone();
        vp.resize(2 * n - 1, MInt::default());

        let mut z = vec![];
        let mut tmp = vec![];
        // Assumes similar degrees (all degrees more than max(deg p, deg q)/2)
        z.resize(n, MInt::default());
        tmp.resize(2 * vp.len() + 16, MInt::default());
        Self::_middlemul(&p.zn, &mut z, &vp, &q.c, &mut tmp);
        Poly { zn: &p.zn, c: z }
    }

    // Inverse and quotient modulo x^n
    // See [Hanrot-Quercia-Zimmerman, section 5.1]
    fn _inv_mod_xn(zn: &ZmodN, z: &mut [MInt], p: &[MInt], tmp: &mut [MInt]) {
        assert!(tmp.len() >= 4 * p.len());
        if p.len() == 1 {
            z[0] = zn.inv(p[0]).unwrap();
            return;
        }
        // 1 / (P0 + P1x) = 1 / P0 - P1 x / P0^2
        // Inverse of size n need 3 half multiplies.
        // [Hanrot-Quercia-Zimmerman, section 4]
        let half = p.len() / 2;
        let half_up = p.len() - half;
        Self::_inv_mod_xn(zn, &mut z[..half_up], &p[..half_up], tmp);
        // Get P1 / P0^2 as a middle product of P by Pinv * Pinv
        // The indices are subtle:
        // We have computed half_up coefficients
        // In P we take 1..2*half_up
        let (tmplo, tmphi) = tmp.split_at_mut(half_up);
        let (tmp_p, tmp_mul) = tmphi.split_at_mut(p.len());
        tmp_p[..p.len() - 1].copy_from_slice(&p[1..]);
        tmp_p[p.len() - 1..].fill(MInt::default());
        Self::_middlemul(zn, tmplo, &tmp_p[..2 * half_up - 1], &z[..half_up], tmp_mul);
        // Multiply again by inverse (but we need low words now)
        Self::_longmul(zn, tmp_p, tmplo, &z[..half], tmp_mul);
        // Take low words
        z[half_up..p.len()].fill(MInt::default());
        Self::_sub(zn, &mut z[half_up..p.len()], &tmp_p[0..half]);
    }

    fn _div_mod_xn(zn: &ZmodN, z: &mut [MInt], p: &[MInt], q: &[MInt], tmp: &mut [MInt]) {
        // Quotient is basically like inverse but with an extra multiplication.
        assert!(p.len() == q.len());
        assert!(tmp.len() >= 5 * p.len());
        if p.len() == 1 {
            z[0] = zn.mul(p[0], zn.inv(q[0]).unwrap());
            return;
        }
        // Compute approximate P/Q mod x^half (using 1/Q mod x^half)
        let half = q.len() / 2;
        let half_up = q.len() - half;
        let (alpha, tmp2) = tmp.split_at_mut(half_up);
        let (tmpmul, tmp3) = tmp2.split_at_mut(2 * half_up);
        let (tmparg, tmphi) = tmp3.split_at_mut(half_up);
        // α in HQZ paper.
        Self::_inv_mod_xn(zn, alpha, &q[..half_up], tmphi);
        // β in HQZ paper.
        Self::_longmul(zn, tmpmul, &alpha[..half_up], &p[..half_up], tmphi);
        z[..half_up].copy_from_slice(&tmpmul[..half_up]);
        // Hensel lift mod x^n
        // Get P1 / Q0^2 as a middle product
        // Shift by one like inverse:
        tmpmul[..q.len() - 1].copy_from_slice(&q[1..]);
        tmpmul[q.len() - 1..].fill(MInt::default());
        // γ in HQZ paper.
        Self::_middlemul(zn, tmparg, &tmpmul[..2 * half_up - 1], &z[..half_up], tmphi);
        // Subtract p_hi-γ and multiply (length = half)
        for i in 0..half {
            tmparg[i] = zn.sub(p[half_up + i], tmparg[i]);
        }
        // Multiply again by inverse (but we need low words now)
        Self::_longmul(zn, tmpmul, &alpha[..half], &tmparg, tmphi);
        // Take low words
        z[half_up..p.len()].copy_from_slice(&tmpmul[..half]);
    }

    pub fn div_mod_xn(p: &'a Poly<'a>, q: &Poly<'a>) -> Self {
        let mut z = vec![MInt::default(); p.c.len()];
        let mut tmp = vec![MInt::default(); 5 * p.c.len()];
        Self::_div_mod_xn(&p.zn, &mut z, &p.c, &q.c, &mut tmp);
        Poly { zn: &p.zn, c: z }
    }
}

#[cfg(test)]
use {crate::Uint, std::str::FromStr};

#[cfg(test)]
const MODULUS256: &'static str =
    "107910248100432407082438802565921895527548119627537727229429245116458288637047";

#[test]
fn test_polymul() {
    let n = Uint::from_str(MODULUS256).unwrap();
    let zn = ZmodN::new(n);

    for width in [99, 512] {
        let p1: Vec<MInt> = (1..=width)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 12345 + x * 1234 + 123)))
            .collect();
        let p2: Vec<MInt> = (1..=width)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 56789 + x * 6789 + 789)))
            .collect();
        let pol1 = Poly { zn: &zn, c: p1 };
        let pol2 = Poly { zn: &zn, c: p2 };

        let res = Poly::mul_karatsuba(&pol1, &pol2);
        for i in 0..res.c.len() {
            let mut z = zn.zero();
            for k in 0..=i {
                if k < pol1.c.len() && i - k < pol2.c.len() {
                    z = zn.add(z, zn.mul(pol1.c[k], pol2.c[i - k]))
                }
            }
            assert_eq!(z, res.c[i], "failure for coeff[{i}]");
        }

        // Test FFT
        let res_fft = Poly::mul_fft(&pol1, &pol2);
        for i in 0..res.c.len() {
            if i < res_fft.c.len() {
                assert_eq!(zn.to_int(res.c[i]), zn.to_int(res_fft.c[i]), "index={i}");
            } else {
                assert_eq!(res.c[i], zn.zero());
            }
        }

        // Test polynomial from roots
        let roots = pol1.c.to_vec();
        let pol = Poly::from_roots(&zn, roots.clone());
        for &r in &roots {
            let mut v = zn.zero();
            for i in 0..pol.c.len() {
                let idx = pol.c.len() - 1 - i;
                v = zn.add(zn.mul(v, r), pol.c[idx]);
            }
            assert_eq!(v, zn.zero());
        }
    }
}

#[test]
fn test_poly_middlemul() {
    let n = Uint::from_str(MODULUS256).unwrap();
    let zn = ZmodN::new(n);

    for width in [128_u64, 256, 512_u64, 513, 1024, 2048] {
        let p1: Vec<MInt> = (1..2 * width)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 12345 + x * 1234 + 123)))
            .collect();
        let p2: Vec<MInt> = (1..=width)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 56789 + x * 6789 + 789)))
            .collect();
        let pol1 = Poly { zn: &zn, c: p1 };
        let mut pol2 = Poly { zn: &zn, c: p2 };
        // Compute middle product
        let res = Poly::middlemul(&pol1, &pol2);
        let width = width as usize;
        // Compute full product
        pol2.c.resize(pol1.c.len(), MInt::default());
        let resfull = Poly::mul_karatsuba(&pol1, &pol2);
        for i in 0..res.c.len() {
            // Coefficient i is (pol1*pol2)[n-1+i] where n = len(pol2)
            let z = resfull.c[width - 1 + i];
            assert_eq!(
                z,
                res.c[i],
                "failure for coeff[{i}] {} vs {}",
                zn.to_int(z),
                zn.to_int(res.c[i])
            );
        }
        // FFT middle product
        if width == 513 {
            continue;
        }
        let mut z = vec![MInt::default(); width];
        Poly::_fft_midmul(&zn, &mut z, &pol1.c, &pol2.c[..width]);
        for i in 0..width {
            assert_eq!(
                zn.to_int(z[i]),
                zn.to_int(resfull.c[width - 1 + i]),
                "index {i}"
            );
        }
        eprintln!("Middle product OK for width {width}");
    }
}

#[test]
fn test_polydiv() {
    let n = Uint::from_str(MODULUS256).unwrap();
    let zn = ZmodN::new(n);
    for degree in [57, 94, 135, 200] {
        // Test inverse
        let p: Vec<MInt> = (0..degree)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 12345 + x * 1234 + 123)))
            .collect();
        let mut z = vec![MInt::default(); p.len()];
        let mut z2 = vec![MInt::default(); 2 * p.len()];
        let mut tmp = vec![MInt::default(); 4 * p.len()];
        Poly::_inv_mod_xn(&zn, &mut z[..], &p, &mut tmp[..]);
        // Compute P * P^-1
        Poly::karatsuba(&zn, &mut z2[..], &p, &z, &mut tmp);
        assert_eq!(z2[0], zn.one());
        for i in 1..p.len() {
            assert_eq!(z2[i], zn.zero());
        }
        // Test division
        let mut tmp = vec![MInt::default(); 6 * p.len()];
        let p2: Vec<MInt> = (0..degree)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 56789 + x * 6789 + 789)))
            .collect();
        Poly::_div_mod_xn(&zn, &mut z[..], &p, &p2, &mut tmp[..]);
        // Compute Q * (P/Q)
        Poly::karatsuba(&zn, &mut z2[..], &p2, &z, &mut tmp);
        for i in 0..p.len() {
            assert_eq!(
                &p[i],
                &z2[i],
                "index {i} {} {}",
                zn.to_int(p[i]),
                zn.to_int(z2[i])
            );
        }
    }
}

#[test]
fn test_polyeval() {
    let n = Uint::from_str(MODULUS256).unwrap();
    let zn = ZmodN::new(n);

    // Trivial degree 2 example.
    // P = x^3 + 3x^2 + 5x +7
    // Q = (x-10)(x-100)(x-1000) x
    // P/Q = 1/x + 1113/x + 1124435/x² + 1125579857/x^3
    // P/Q2 = 11435/x + 1144857/x²
    // P/Q1 = 1003005/x + 1003005007/x²
    // P/x-10 = 1357/x
    // The length of P is at most the power of two above len(a).
    let p = Poly {
        zn: &zn,
        c: vec![
            zn.from_int(Uint::from(7_u64)),
            zn.from_int(Uint::from(5_u64)),
            zn.from_int(Uint::from(3_u64)),
            zn.from_int(Uint::from(1_u64)),
        ],
    };

    let vals = p._multi_eval(&[
        zn.from_int(Uint::from(10_u64)),
        zn.from_int(Uint::from(100_u64)),
        zn.from_int(Uint::from(1000_u64)),
        //zn.from_int(Uint::from(10000_u64)),
    ]);
    assert_eq!(zn.to_int(vals[0]).digits()[0], 1357);
    assert_eq!(zn.to_int(vals[1]).digits()[0], 1030507);
    assert_eq!(zn.to_int(vals[2]).digits()[0], 1003005007);
}
