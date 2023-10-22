// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Linear algebra routines for sparse matrices over
//! the ring of integers and GF(p)
//!
//! Computations are based on the ordinary Wiedemann algorithm using
//! iterated multiplications by a single vector. To take advantage
//! of vectorisation, the determinant computation can process a block
//! of vectors using several 64-bit moduli (typical size 4 or 8).
//!
//! Available functions include:
//! - computation of determinants modulo 64-bit primes
//! - computation of integer determinants using CRT
//! - kernel vectors of matrices over a prime field
//!
//! For class group computations, matrix dimension will usually be at most 40000.

use std::cmp::max;

use num_integer::Integer;
use num_traits::Euclid;

use bnum::cast::CastFrom;
use bnum::types::{I256, U256, U512};
use bnum::{BInt, BUint};

use rand::rngs::StdRng;
use rand::{prelude::SliceRandom, Rng, SeedableRng};

use crate::arith;
use crate::arith_gcd;
use crate::arith_montgomery::{mg_2adic_inv, mg_inv, mg_mul, mg_redc};

type IntLarge = BInt<256>;
type UintLarge = BUint<256>;

/// Compute the index of a lattice generated by a basis of sparse
/// vectors.
/// Estimates are given as loose lower/upper bounds.
pub fn compute_lattice_index(
    dim: usize,
    rows: &Vec<Vec<(u32, i32)>>,
    hmin: f64,
    hmax: f64,
) -> U256 {
    // The bounds can be very approximate.
    let prec = (hmax - hmin).abs();
    let hmin = (0.9 * hmin).max(hmin - 3.0 * prec);
    let hmax = (1.1 * hmax).min(hmax + 3.0 * prec);
    if rows.len() == 0 {
        // The determinant of the empty matrix is 1.
        assert!(hmin <= 1.0 && 1.0 <= hmax);
        return U256::ONE;
    }
    assert!(hmin <= hmax);
    assert!(hmax / hmin < 1.5);
    let mut gcd = UintLarge::ZERO;
    // Arbitrary, reasonably deterministic pseudo-random generator.
    let mut rng = StdRng::seed_from_u64(163);
    for _ in 0..100 {
        let selrows = rows
            .choose_multiple(&mut rng, dim)
            .map(|v| &v[..])
            .collect();
        let mat = SparseMat::new(selrows);
        let det = mat.detz();
        //eprintln!("det over Z = {det}");
        if det.is_zero() {
            continue;
        }
        let det = det.unsigned_abs();
        gcd = arith_gcd::big_gcd(&gcd, &det);
        let gcd_f = f64::cast_from(gcd);
        if gcd_f / hmin > 1e4 {
            continue;
        }
        let m1 = (gcd_f / hmax).round() as i64;
        let m2 = (gcd_f / hmin).round() as i64;
        let mut candidates = vec![];
        // GCD is known to be small now.
        let gcd = U256::cast_from(gcd);
        for m in m1..m2 + 1 {
            if m <= 0 {
                continue;
            }
            let m = m as u64;
            let r: u64 = gcd % m;
            if r == 0 {
                let q: U256 = gcd / U256::from(m);
                let qf = f64::cast_from(q);
                if 0.9 * hmin <= qf && qf <= 1.1 * hmax {
                    candidates.push(q)
                }
            }
        }
        if candidates.len() == 1 {
            return candidates[0];
        }
    }
    panic!("failed to determine lattice index");
}

/// A sparse, square matrix in compressed sparse row representation.
///
/// The implementation is optimised to benefit from a large number of ±1
/// coefficients. They are stored separately to avoid unnecessary
/// multiplication instructions.
#[derive(Debug)]
pub struct SparseMat {
    // Compressed sparse row representation
    // Row i is values[indices[i])..indices(i+1]]
    // Coefficients +1
    values_p1: Vec<u32>,
    indices_p1: Vec<u32>,
    // Coefficients -1
    values_m1: Vec<u32>,
    indices_m1: Vec<u32>,
    // Other coefficients
    values_x: Vec<(u32, i32)>,
    indices_x: Vec<u32>,

    // Size
    size: usize,
}

impl SparseMat {
    pub fn new<V: AsRef<[(u32, i32)]>>(rows: Vec<V>) -> Self {
        let size = rows.len();
        let mut vp = vec![];
        let mut vm = vec![];
        let mut vx = vec![];
        let mut ip = vec![0];
        let mut im = vec![0];
        let mut ix = vec![0];
        for r in &rows {
            for &(j, e) in r.as_ref().iter() {
                if e == 1 {
                    vp.push(j);
                } else if e == -1 {
                    vm.push(j);
                } else {
                    vx.push((j, e));
                }
            }
            ip.push(vp.len() as u32);
            im.push(vm.len() as u32);
            ix.push(vx.len() as u32);
        }
        Self {
            values_p1: vp,
            indices_p1: ip,
            values_m1: vm,
            indices_m1: im,
            values_x: vx,
            indices_x: ix,
            size,
        }
    }

    /// Compute product Mv for a tuple of vectors v,
    /// such that v[i] is considered over field GF(p[i]).
    ///
    /// Preferred block sizes for vectorisation are N=4 or N=8.
    fn mulp<const N: usize>(&self, p: [u64; N], v: &[[u64; N]], out: &mut [[u64; N]]) {
        assert!(v.len() == self.size);
        for i in 0..self.size {
            let mut x = [0; N];
            unsafe {
                let idx1 = *self.indices_p1.get_unchecked(i) as usize;
                let idx2 = *self.indices_p1.get_unchecked(i + 1) as usize;
                for idx in idx1..idx2 {
                    let j = *self.values_p1.get_unchecked(idx);
                    let vj = v.get_unchecked(j as usize);
                    for k in 0..N {
                        x[k] += vj[k] as i64;
                    }
                }
                let idx1 = *self.indices_m1.get_unchecked(i) as usize;
                let idx2 = *self.indices_m1.get_unchecked(i + 1) as usize;
                for idx in idx1..idx2 {
                    let j = *self.values_m1.get_unchecked(idx);
                    let vj = v.get_unchecked(j as usize);
                    for k in 0..N {
                        x[k] -= vj[k] as i64;
                    }
                }
                // The last loop is less vectorisation friendly because
                // packed multiplication of u64 is uncommon among CPU architectures.
                let idx1 = *self.indices_x.get_unchecked(i) as usize;
                let idx2 = *self.indices_x.get_unchecked(i + 1) as usize;
                for idx in idx1..idx2 {
                    let (j, mij) = *self.values_x.get_unchecked(idx);
                    let vj = v.get_unchecked(j as usize);
                    for k in 0..N {
                        x[k] += mij as i64 * vj[k] as i64;
                    }
                }
                let wi = out.get_unchecked_mut(i);
                for k in 0..N {
                    wi[k] = x[k].rem_euclid(p[k] as i64) as u64;
                }
            }
        }
    }

    /// Variant of mulp for scalar vectors modulo a (possibly large) prime.
    fn mulpbig<U, I>(&self, p: U, v: &[U], out: &mut [U])
    where
        U: Copy + CastFrom<I>,
        I: Integer + Euclid + CastFrom<U> + CastFrom<i32>,
    {
        assert!(v.len() == self.size);
        assert!(out.len() == self.size);
        for i in 0..self.size {
            let mut x = I::zero();
            unsafe {
                let idx1 = *self.indices_p1.get_unchecked(i) as usize;
                let idx2 = *self.indices_p1.get_unchecked(i + 1) as usize;
                for idx in idx1..idx2 {
                    let j = *self.values_p1.get_unchecked(idx);
                    let vj = v.get_unchecked(j as usize);
                    x = x + I::cast_from(*vj);
                }
                let idx1 = *self.indices_m1.get_unchecked(i) as usize;
                let idx2 = *self.indices_m1.get_unchecked(i + 1) as usize;
                for idx in idx1..idx2 {
                    let j = *self.values_m1.get_unchecked(idx);
                    let vj = v.get_unchecked(j as usize);
                    x = x - I::cast_from(*vj);
                }
                let idx1 = *self.indices_x.get_unchecked(i) as usize;
                let idx2 = *self.indices_x.get_unchecked(i + 1) as usize;
                for idx in idx1..idx2 {
                    let (j, mij) = *self.values_x.get_unchecked(idx);
                    let vj = v.get_unchecked(j as usize);
                    x = x + I::cast_from(mij) * I::cast_from(*vj);
                }
                out[i] = U::cast_from(x.rem_euclid(&I::cast_from(p)));
            }
        }
    }

    /// Matrix determinant over the integers. The computations
    /// uses CRT and determinant modulo word-size primes.
    pub fn detz(&self) -> IntLarge {
        let moduli = self.select_crtprimes();
        let mut modp = vec![];
        let mut det = IntLarge::ZERO;
        for p4 in moduli.chunks_exact(4) {
            let p = [p4[0], p4[1], p4[2], p4[3]];
            let d = self.detp4(p);
            //eprintln!("det {d:?} mod {p:?}");
            for k in 0..4 {
                modp.push(d[k]);
            }
            let det_ = crt(&modp, &moduli);
            if det != det_ {
                det = det_;
            } else {
                //eprintln!("final {det}");
                return det;
            }
        }
        unreachable!()
    }

    /// The norm is of the matrix is an upper bound N
    /// such that for any vector v, |Mv| <= N*|v|
    /// where |v| = max(abs(v[i])).
    ///
    /// It is used to prevent integer overflows. Typical matrices
    /// have norm less than 256.
    fn norm(&self) -> u64 {
        let mut norm: u64 = 0;
        for i in 0..self.size {
            let mut pos: i64 = 0;
            let mut neg: i64 = 0;
            pos += (self.indices_p1[i + 1] - self.indices_p1[i]) as i64;
            neg += (self.indices_m1[i + 1] - self.indices_m1[i]) as i64;
            let start = self.indices_x[i] as usize;
            let end = self.indices_x[i + 1] as usize;
            for &(_, v) in &self.values_x[start..end] {
                if v > 0 {
                    pos += v as i64;
                } else {
                    neg -= v as i64;
                }
            }
            norm = max(norm, max(pos, neg) as u64);
        }
        norm
    }

    fn select_crtprimes(&self) -> Vec<u64> {
        let mut moduli = vec![];
        let norm = self.norm();
        // Any prime less than 2^63 / norm should be fine.
        // About 10% of 56-bit numbers 30k-1 are prime.
        let bound: u64 = (1 << 63) / norm;
        let mut p = 30 * (bound / 30) - 1;
        debug_assert!(p * norm < 1 << 63);
        while moduli.len() < self.size {
            if crate::isprime64(p) {
                moduli.push(p)
            }
            p -= 30;
        }
        //eprintln!("selected moduli {moduli:?}");
        moduli
    }

    /// Matrix determinant over finite field GF(p).
    /// Computed using Wiedemann algorithm.
    pub fn detp4(&self, p: [u64; 4]) -> [u64; 4] {
        let mut v = vec![[0; 4]; self.size];
        let mut w = vec![[0; 4]; self.size];
        // Fill vector pseudorandomly using a Fibonacci sequence.
        let (mut x, mut y) = (0, 1);
        for j in 0..self.size {
            (x, y) = (y, (x + y) % 65537);
            v[j].fill(y);
        }
        // Compute Krylov sequences
        let mut seq = [vec![], vec![], vec![], vec![]];
        for k in 0..4 {
            seq[k] = Vec::with_capacity(2 * self.size);
        }
        loop {
            for k in 0..4 {
                seq[k].push(v[0][k]);
            }
            if seq[0].len() == 2 * self.size {
                break;
            }
            self.mulp(p, &v, &mut w);
            (v, w) = (w, v);
        }
        let mut dets = [0; 4];
        for k in 0..4 {
            let charpoly = berlekamp_massey(p[k], &seq[k]);
            let c0 = charpoly[self.size];
            // FIXME: what if degree != n.
            dets[k] = if self.size & 1 == 1 && c0 != 0 {
                p[k] - c0
            } else {
                c0
            };
        }
        dets
    }

    /// Kernel modulo large p.
    /// This assumes that the rank modulo p is exactly n-1.
    pub fn ker_p256(&self, p: U256) -> Option<Vec<U256>> {
        // FIXME: return an error?
        let norm = self.norm();
        let k = if p.bits() < 56 && norm < 256 {
            self.ker_pbig::<u64, i64, u128>(u64::cast_from(p))?
                .iter()
                .map(|&x| U256::cast_from(x))
                .collect()
        } else if p.bits() < 120 && norm < 256 {
            self.ker_pbig::<u128, i128, U256>(u128::cast_from(p))?
                .iter()
                .map(|&x| U256::cast_from(x))
                .collect()
        } else if p.bits() < 182 && norm < 1024 {
            self.ker_pbig::<BUint<3>, BInt<3>, BUint<6>>(BUint::cast_from(p))?
                .iter()
                .map(|&x| U256::cast_from(x))
                .collect()
        } else {
            self.ker_pbig::<U256, I256, U512>(p)?
        };
        Some(k)
    }

    fn ker_pbig<U, I, UU>(&self, p: U) -> Option<Vec<U>>
    where
        U: std::fmt::Debug
            + Copy
            + Integer
            + CastFrom<I>
            + CastFrom<UU>
            + CastFrom<U256>
            + From<u64>,
        UU: Copy + Integer + CastFrom<U>,
        I: Integer + Euclid + CastFrom<U> + CastFrom<i32>,
        U256: CastFrom<U>,
    {
        let mut v = vec![U::zero(); self.size];
        let mut w = vec![U::zero(); self.size];
        // Fill vector pseudorandomly using a Fibonacci sequence.
        let (mut x, mut y) = (0, 1);
        for j in 0..self.size {
            (x, y) = (y, (x + y) % 65537);
            v[j] = y.into();
        }
        // Compute Krylov sequence
        let mut seq = Vec::with_capacity(2 * self.size);
        loop {
            seq.push(v[0]);
            if seq.len() == 2 * self.size {
                break;
            }
            self.mulpbig::<U, I>(p, &v, &mut w);
            (v, w) = (w, v);
        }
        let charpoly = berlekamp_massey_big::<U, UU>(p, &seq);
        let c0 = charpoly[self.size];
        assert!(c0.is_zero());
        if charpoly[self.size - 1].is_zero() {
            // Characteristic polynomial has a double root
            return None;
        }
        // FIXME: what if degree != n.
        // Now evaluate a kernel vector using Horner rule.
        let mut v0 = vec![U::zero(); self.size];
        let mut rng = StdRng::seed_from_u64(163);
        if U256::cast_from(p).bits() <= 64 {
            let p64 = u64::cast_from(U256::cast_from(p));
            for x in v0.iter_mut() {
                *x = rng.gen_range(1..p64).into();
            }
        } else {
            for x in v0.iter_mut() {
                *x = rng.gen::<u64>().into();
            }
        }
        v.copy_from_slice(&v0);
        for i in 1..self.size {
            // Map V to M V + a V0
            self.mulpbig::<U, I>(p, &v, &mut w);
            let ci = UU::cast_from(charpoly[i]);
            for j in 0..self.size {
                let wj = UU::cast_from(w[j]) + ci * UU::cast_from(v0[j]);
                w[j] = U::cast_from(wj % UU::cast_from(p));
            }
            (v, w) = (w, v);
        }
        // Check that we obtained an actual kernel element.
        self.mulpbig::<U, I>(p, &v, &mut w);
        assert!(w.iter().all(|&x| x.is_zero()));
        assert!(v.iter().any(|&x| !x.is_zero()));
        Some(v)
    }
}

/// Reconstruct a large integer from a list of residues mod p.
fn crt(modp: &[u64], primes: &[u64]) -> IntLarge {
    let mut crt_basis = vec![IntLarge::ONE; modp.len()];
    for i in 0..modp.len() {
        let mut inv: u64 = 1;
        let pi = primes[i];
        for j in 0..modp.len() {
            if i != j {
                crt_basis[i] *= IntLarge::from(primes[j]);
                let inv_j = arith::inv_mod64(primes[j], pi).unwrap();
                inv = ((inv as u128 * inv_j as u128) % (pi as u128)) as u64;
            }
        }
        crt_basis[i] *= IntLarge::from(inv);
    }
    let mut prod = IntLarge::ONE;
    for &p in &primes[..modp.len()] {
        prod *= IntLarge::from(p);
    }
    let mut det = IntLarge::ZERO;
    for (&deti, bi) in modp.iter().zip(&crt_basis) {
        det += IntLarge::from(deti) * bi;
    }
    det %= prod;
    if det > (prod >> 1) {
        IntLarge::cast_from(det) - IntLarge::cast_from(prod)
    } else {
        IntLarge::cast_from(det)
    }
}

/// Returns a characteristic polynomial for sequence
/// seq over GF(p).
///
/// Implementation has quadratic time complexity.
pub fn berlekamp_massey(p: u64, seq: &[u64]) -> Vec<u64> {
    // Prime p is less than 64 bits.
    // We use a naive quadratic algorithm (truncated Euclid)
    // and Montgomery arithmetic.
    let pinv = mg_2adic_inv(p);
    let r = (1u128 << 64).rem_euclid(p as u128) as u64;
    let r2 = (r as u128 * r as u128).rem_euclid(p as u128) as u64;
    let mulp = |a: u64, b: u64| mg_mul(p, pinv, a, b);
    let dotp = |a: u64, b: u64, c: u64, d: u64| {
        let ab_cd: u128 = a as u128 * b as u128 + c as u128 * d as u128;
        mg_redc(p, pinv, ab_cd)
    };
    let invp = |a: u64| mg_inv(p, pinv, r2, a).unwrap();
    debug_assert!(mulp(2, invp(2)) == r);
    let subp = |a: &mut u64, b: u64| {
        if b <= *a {
            *a -= b
        } else {
            *a += p - b;
        }
    };
    // Loop invariants:
    // u * seq = f mod x^N
    // v * seq = g mod x^N
    let n = seq.len();
    let mut u = vec![0; n]; // 1
    u[0] = 1;
    let mut du = 0;
    let mut f = seq.to_vec();
    let mut df = n - 1;
    while f[df] == 0 && df > 0 {
        df -= 1
    }
    if f[df] == 0 {
        return vec![];
    }
    // g = x^d f mod x^n
    let mut v = vec![0; n]; // x
    v[n - df] = 1;
    let mut dv = n - df;
    let mut g = vec![0; n];
    g[n - df..].copy_from_slice(&f[0..df]);
    let mut dg = n - 1;
    while g[dg] == 0 && dg > 0 {
        dg -= 1
    }
    // Not supposed to happen
    if g[dg] == 0 {
        return vec![];
    }
    for _ in 0..2 * n {
        if df > dg {
            (u, v) = (v, u);
            (du, dv) = (dv, du);
            (f, g) = (g, f);
            (df, dg) = (dg, df);
        }
        if df < n / 2 {
            // seq = f / u + o(x^n)
            // Normalize to u = 1 + ...
            // Note that we are in Montgomery arithmetic!
            // but output is expected in natural representation.
            // FIXME: divide by x^v??
            assert!(u[0] != 0);
            let q = mg_redc(p, pinv, invp(u[0]) as u128);
            for i in 0..n {
                u[i] = mulp(u[i], q);
            }
            return u;
        }
        // Divide g by f (dg >= df)
        if dg > df && df > 1 {
            // g -= (q1 x + q0) x^d * f
            // Quotient of power series:
            // 1/(f0 + f1 t) = 1/f0 - f1/f0² t + o(t²)
            // q1 + q0 t = g0/f0 + (g1/f0 - g0f1/f0²) t + o(t²)
            let invf0 = invp(f[df]);
            let invf1 = mulp(f[df - 1], invf0);
            let q1 = mulp(g[dg], invf0);
            let mut q0 = mulp(g[dg - 1], invf0);
            subp(&mut q0, mulp(q1, invf1));
            let d = dg - df - 1;
            subp(&mut g[d], mulp(f[0], q0));
            for i in 1..=df {
                subp(&mut g[i + d], dotp(f[i], q0, f[i - 1], q1));
            }
            subp(&mut g[dg], mulp(f[df], q1));
            assert_eq!(g[dg], 0);
            assert_eq!(g[dg - 1], 0);
            // v -= (q1 x + q0) x^d u
            subp(&mut v[d], mulp(u[0], q0));
            for i in 1..=du {
                subp(&mut v[i + d], dotp(u[i], q0, u[i - 1], q1));
            }
            subp(&mut v[d + du + 1], mulp(u[du], q1));
            for i in dv..=(du + d + 1) {
                if v[i] != 0 {
                    dv = i
                }
            }
        } else {
            // g -= q x^d * f
            let q = mulp(g[dg], invp(f[df]));
            let d = dg - df;
            for i in 0..=df {
                subp(&mut g[i + d], mulp(q, f[i]));
            }
            assert_eq!(g[dg], 0);
            // v -= q x^d u
            for i in 0..=du {
                subp(&mut v[i + d], mulp(q, u[i]));
            }
            for i in dv..=(du + d) {
                if v[i] != 0 {
                    dv = i
                }
            }
        }
        while g[dg] == 0 && dg > 0 {
            dg -= 1
        }
    }
    unreachable!()
}

/// Same as `berlekamp_massey` but for larger integers.
///
/// Type parameter `UU` must be a integer type twice larger than `U`
pub fn berlekamp_massey_big<U, UU>(p: U, seq: &[U]) -> Vec<U>
where
    U: std::fmt::Debug + Integer + Copy + CastFrom<UU> + CastFrom<U256> + From<u64>,
    UU: Integer + CastFrom<U>,
    U256: CastFrom<U>,
{
    let mulp = |a: U, b: U| {
        let ab = UU::cast_from(a) * UU::cast_from(b);
        U::cast_from(ab % UU::cast_from(p))
    };
    let invp = |a: U| -> U {
        U::cast_from(arith_gcd::inv_mod(&U256::cast_from(a), &U256::cast_from(p)).unwrap())
    };
    let subp = |a: &mut U, b: U| {
        if &*a >= &b {
            *a = *a - b
        } else {
            *a = *a + p - b
        }
    };
    // Loop invariants:
    // u * seq = f mod x^N
    // v * seq = g mod x^N
    let n = seq.len();
    let mut u = vec![U::zero(); n]; // 1
    u[0] = U::from(1);
    let mut v = vec![U::zero(); n]; // x
    v[1] = U::from(1);
    let (mut du, mut dv) = (0, 1);
    let mut f = seq.to_vec();
    let mut df = n - 1;
    while f[df].is_zero() && df > 0 {
        df -= 1
    }
    if f[df].is_zero() {
        return vec![];
    }
    let mut g = vec![U::zero(); n];
    // g = x^d f mod x^n
    g[n - df..].copy_from_slice(&f[0..df]);

    let mut dg = n - 1;
    while g[dg].is_zero() && dg > 0 {
        dg -= 1
    }
    // Not supposed to happen
    if g[dg].is_zero() {
        return vec![];
    }
    for _ in 0..2 * n {
        if df > dg {
            (u, v) = (v, u);
            (du, dv) = (dv, du);
            (f, g) = (g, f);
            (df, dg) = (dg, df);
        }
        if df < n / 2 {
            // seq = f / u + o(x^n)
            // Normalize to u = 1 + ...
            // Note that we are in Montgomery arithmetic!
            // but output is expected in natural representation.
            // FIXME: divide by x^v??
            assert!(!u[0].is_zero());
            let q = invp(u[0]);
            for i in 0..n {
                u[i] = mulp(u[i], q);
            }
            return u;
        }
        // Divide g by f (dg >= df)
        // g -= q x^d * f
        let q = mulp(g[dg], invp(f[df]));
        let d = dg - df;
        for i in 0..=df {
            subp(&mut g[i + d], mulp(q, f[i]));
        }
        assert!(g[dg].is_zero());
        // v -= q x^d u
        for i in 0..=du {
            subp(&mut v[i + d], mulp(q, u[i]));
        }
        for i in dv..=(du + d) {
            if !v[i].is_zero() {
                dv = i
            }
        }
        while g[dg].is_zero() && dg > 0 {
            dg -= 1
        }
    }
    unreachable!()
}

#[test]
fn test_berlekamp_massey() {
    let p = 65537;
    let mut v = vec![123, 456, 789, 1, 2, 3, 4];
    let taps = [1234, 2345, 3456, 4567, 5678, 6789, 7890];
    // Construct a recurrent linear sequence
    for _ in 0..7 {
        let mut x = 0;
        for j in 0..7 {
            x += taps[j] * v[v.len() - 1 - j];
        }
        v.push(x % p);
    }
    eprintln!("{v:?}");
    let out = berlekamp_massey(p, &v);
    eprintln!("{out:?}");
    let mut want = vec![1];
    for t in taps {
        want.push(p - t);
    }
    assert_eq!(&out[..want.len()], &want[..]);
    // Also if last element is zero
    let v = vec![
        25063, 456, 789, 1, 2, 3, 4, 18825, 40304, 10105, 63795, 57513, 53235, 0,
    ];
    let out = berlekamp_massey(p, &v);
    assert_eq!(&out[..want.len()], &want[..]);
}

#[test]
fn test_sparse_det() {
    let p: u64 = 1_000_000_000_000_037;
    let mut rows = vec![];
    // Create a sparse matrix
    for i in 0..211 {
        let mut row = vec![];
        let a = (i + 2) % 211;
        let b = (i * i + i + 1) % 211;
        let c = (2 * i * i + 3 * i + 4) % 211;
        row.push((a, 1));
        if b != a {
            row.push((b, 2));
        }
        if c != a && c != b {
            row.push((c, 3));
        }
        rows.push(row);
    }
    let mat = SparseMat::new(rows);
    assert_eq!(mat.detp4([p, p, p, p])[0], 369188203301422);

    // Larger example
    let mut rows = vec![];
    for i in 0..503 {
        let mut row = vec![];
        let a = (i + 2) % 503;
        let b = (i * i + i + 1) % 503;
        let c = (2 * i * i + 3 * i + 4) % 503;
        row.push((a, 1));
        if b != a {
            row.push((b, 2));
        }
        if c != a && c != b {
            row.push((c, 3));
        }
        rows.push(row);
    }
    let mat = SparseMat::new(rows);
    eprintln!("{}", mat.detz());
    let p2 = 100_000_000_000_031;
    assert_eq!(
        mat.detp4([p, p2, p, p2]),
        [
            36_799_940_914_198,
            54_949_682_739_764,
            36_799_940_914_198,
            54_949_682_739_764
        ]
    );
}

#[test]
fn test_kernel() {
    let p: u64 = 11499163612801;
    let mut rows = vec![];
    for i in 0..503 {
        let mut row = vec![];
        let a = (i + 2) % 503;
        let b = (i * i + i + 1) % 503;
        let c = (2 * i * i + 3 * i + 4) % 503;
        row.push((a, 1));
        if b != a {
            row.push((b, 2));
        }
        if c != a && c != b {
            row.push((c, 3));
        }
        rows.push(row);
    }
    let mat = SparseMat::new(rows);
    // Check done by function.
    mat.ker_p256(p.into());
}
