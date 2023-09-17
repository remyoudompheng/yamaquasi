// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Linear algebra routines for matrices over
//! the ring of integers and GF(p)
//! This is used for class group computations, for medium-sized
//! matrices (at most size 1000).

use bnum::cast::CastFrom;
use num_traits::ToPrimitive;

use crate::arith;
use crate::{Int, Uint};

// A context to compute Gram-Schmidt orthogonal bases out of integral
// vectors. This is used to find linearly independent vectors and
// to compute a floating-point estimate for the determinant of a basis.
#[derive(Clone, Default)]
pub struct GramBuilder {
    gram: Vec<Vec<f64>>,
    norms: Vec<f64>,
    pub threshold: Option<f64>,
}

impl GramBuilder {
    pub fn add(&mut self, v: &[i64]) -> bool {
        // Compute floating-point projection
        let mut v: Vec<f64> = v.iter().map(|&x| x as f64).collect();
        for (i, g) in self.gram.iter().enumerate() {
            if self.norms[i] < 1e-9 {
                continue;
            }
            let mu = dotf(&g, &v) / self.norms[i];
            submulf(&mut v, &g, mu);
        }
        let n = normf(&v);
        if let Some(thr) = self.threshold {
            if n < thr {
                return false;
            }
        }
        self.norms.push(n);
        self.gram.push(v);
        true
    }

    pub fn rank(&self) -> usize {
        self.gram.len()
    }

    pub fn det_estimate(&self) -> f64 {
        self.norms.iter().product::<f64>().sqrt()
    }
}

fn normf(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>()
}

fn dotf(v: &[f64], w: &[f64]) -> f64 {
    v.iter().zip(w).map(|(a, b)| a * b).sum::<f64>()
}

fn submulf(v: &mut [f64], w: &[f64], m: f64) {
    for i in 0..v.len() {
        v[i] -= m * w[i];
    }
}

/// Determinant of an integer matrix, with a known estimate
/// of the determinant absolute value (volume) obtained by
/// a Gram-Schmidt computation. The estimate is assumed to have
/// at least 24 correct bits.
pub fn det_matz(mat: Vec<&[i64]>, estimate: f64) -> Int {
    let bits = estimate.abs().log2().round();
    assert!(bits >= 1.0);
    assert!(bits <= 63.0 * 16.0);
    let bits = bits as usize;
    let mut modp = vec![];
    'crtloop: while 60 * modp.len() < bits && modp.len() < 16 {
        let p = PRIMES[modp.len()];
        let mut mp = GFpEchelonBuilder::new(p);
        for v in mat.iter() {
            if !mp.add(v) {
                // Determinant is zero mod p
                modp.push(0);
                continue 'crtloop;
            }
        }
        let dp = mp.det();
        //eprintln!("det mod {p} = {dp}");
        modp.push(dp);
    }
    // Now apply CRT
    let mut crt_basis = vec![Uint::ONE; modp.len()];
    for i in 0..modp.len() {
        let mut inv: u64 = 1;
        let pi = PRIMES[i];
        for j in 0..modp.len() {
            if i != j {
                crt_basis[i] *= Uint::from(PRIMES[j]);
                let inv_j = arith::inv_mod64(PRIMES[j], pi).unwrap();
                inv = ((inv as u128 * inv_j as u128) % (pi as u128)) as u64;
            }
        }
        crt_basis[i] *= Uint::from(inv);
    }
    let mut prod = Uint::ONE;
    for &p in &PRIMES[..modp.len()] {
        prod *= Uint::from(p);
    }
    assert!(prod.bits() as usize >= bits + 2);
    let mut det = Uint::ZERO;
    for (&deti, bi) in modp.iter().zip(&crt_basis) {
        det += Uint::from(deti) * bi;
    }
    det %= prod;
    let det = if det > (prod >> 1) {
        Int::cast_from(det) - Int::cast_from(prod)
    } else {
        Int::cast_from(det)
    };
    // Check result using floating-point estimate.
    let k = det.to_f64().unwrap().abs() / estimate;
    assert!((k - 1.0).abs() < 1e-6, "k={k}");
    det
}

/// Builder for echelonized bases over finite field GF(p).
/// Modulus is usually around 60 bits.
struct GFpEchelonBuilder {
    p: u64,
    basis: Vec<Vec<u64>>,
    indices: Vec<usize>,
    factors: Vec<u64>,
}

// A few large 63-bit primes, enough to compute 1024-bit determinants.
const PRIMES: [u64; 17] = [
    0x7fffffffffffffe7,
    0x7fffffffffffff5b,
    0x7ffffffffffffefd,
    0x7ffffffffffffed3,
    0x7ffffffffffffe89,
    0x7ffffffffffffe7d,
    0x7ffffffffffffe79,
    0x7ffffffffffffe67,
    0x7ffffffffffffe37,
    0x7ffffffffffffe29,
    0x7ffffffffffffdfb,
    0x7ffffffffffffdef,
    0x7ffffffffffffddb,
    0x7ffffffffffffd8d,
    0x7ffffffffffffd77,
    0x7ffffffffffffd63,
    0x7ffffffffffffd39,
];

impl GFpEchelonBuilder {
    fn new(p: u64) -> Self {
        Self {
            p,
            basis: vec![],
            indices: vec![],
            factors: vec![],
        }
    }

    fn add(&mut self, v: &[i64]) -> bool {
        if self.basis.len() > 0 {
            assert_eq!(v.len(), self.basis[0].len());
        }
        let mut vp = vec![0u64; v.len()];
        for i in 0..v.len() {
            vp[i] = v[i].unsigned_abs() % self.p;
            if v[i] < 0 {
                vp[i] = self.p - vp[i];
            }
        }
        // Eliminate
        for (&idx, b) in self.indices.iter().zip(&self.basis) {
            let vi = vp[idx];
            if vi == 0 {
                continue;
            }
            self.submul(&mut vp, b, vi);
        }
        // Find next echelon index
        for i in 0..vp.len() {
            let vi = vp[i];
            if vi != 0 {
                self.div(&mut vp, vi);
                assert_eq!(vp[i], 1);
                self.indices.push(i);
                self.basis.push(vp);
                self.factors.push(vi);
                return true;
            }
        }
        false
    }

    fn div(&self, v: &mut [u64], m: u64) {
        let p = self.p;
        let minv = arith::inv_mod64(m, p).unwrap();
        for vi in v {
            let x = (*vi as u128 * minv as u128).rem_euclid(p as u128);
            *vi = x as u64;
        }
    }

    fn submul(&self, v: &mut [u64], w: &[u64], m: u64) {
        let p = self.p;
        for (i, vi) in v.iter_mut().enumerate() {
            let x = *vi as i128 - m as i128 * w[i] as i128;
            *vi = x.rem_euclid(p as i128) as u64;
        }
    }

    fn det(&self) -> u64 {
        assert!(self.factors.len() == self.basis[0].len());
        // The determinant sign depends on the order of echelon indices.
        let mut ind = self.indices.clone();
        let mut i = 0;
        let mut swaps = 0;
        while i < ind.len() {
            let mut j = ind[i];
            while j != i {
                ind.swap(i, j);
                swaps += 1;
                j = ind[i];
            }
            i += 1;
        }
        let p = self.p;
        let mut det = if swaps % 2 == 0 { 1 } else { p - 1 };
        // Multiply diagonal elements
        for &f in &self.factors {
            det = (det as u128 * f as u128).rem_euclid(p as u128) as u64;
        }
        det
    }
}

#[test]
fn test_matrix_det() {
    use std::str::FromStr;
    // A randomly-generated integer matrix.
    // Its determinant is +14293689752795
    const M: &[&[i64]] = &[
        &[14, 11, 22, 36, 31, 28, 15, 19, 15, 6],
        &[13, 16, 17, 9, 2, 4, 21, 35, 2, 35],
        &[14, 18, 19, 34, 4, 27, 5, 15, 11, 32],
        &[25, 19, 25, 11, 25, 27, 25, 32, 28, 11],
        &[27, 34, 28, 4, 9, 9, 7, 34, 32, 0],
        &[1, 30, 6, 8, 18, 28, 16, 0, 28, 0],
        &[14, 2, 29, 33, 13, 22, 19, 9, 16, 35],
        &[18, 36, 27, 31, 2, 28, 24, 16, 13, 5],
        &[23, 1, 25, 22, 0, 23, 8, 23, 23, 13],
        &[0, 16, 10, 30, 13, 35, 34, 22, 17, 22],
    ];
    let mut g = GramBuilder::default();
    for v in M {
        assert!(g.add(v));
    }
    let det_approx = 14293689752795.0;
    assert!((g.det_estimate() - det_approx).abs() < 0.01);

    assert_eq!(
        det_matz(M.to_vec(), det_approx),
        Int::from(14293689752795_i64)
    );

    // Another random matrix. Its determinant is -1258731415851007568569087128744.
    #[rustfmt::skip]
    const M16: &[&[i64]] = &[
        &[5, 61, 59, 39, 41, 82, 43, 42, 35, 93, 60, 53, 90, 94, 68, 66],
        &[43, 1, 90, 42, 11, 74, 13, 72, 22, 5, 53, 13, 86, 12, 54, 67],
        &[67, 46, 30, 32, 64, 38, 54, 59, 52, 61, 96, 68, 60, 16, 79, 61],
        &[59, 70, 46, 90, 3, 39, 68, 19, 26, 93, 64, 73, 90, 13, 37, 38],
        &[76, 29, 19, 2, 88, 12, 16, 13, 8, 1, 10, 61, 40, 43, 7, 78],
        &[57, 16, 61, 82, 27, 2, 7, 11, 75, 24, 83, 63, 74, 70, 85, 29],
        &[80, 51, 25, 76, 39, 77, 32, 3, 69, 53, 25, 63, 7, 57, 40, 16],
        &[70, 37, 24, 56, 46, 60, 26, 96, 20, 33, 55, 28, 18, 45, 84, 95],
        &[31, 23, 44, 32, 37, 53, 47, 26, 71, 50, 1, 22, 57, 36, 36, 12],
        &[45, 60, 42, 0, 78, 39, 61, 71, 86, 41, 73, 0, 1, 57, 36, 16],
        &[63, 56, 78, 59, 36, 6, 2, 57, 71, 92, 78, 29, 30, 41, 60, 34],
        &[65, 83, 8, 71, 55, 35, 69, 7, 10, 77, 40, 15, 21, 21, 47, 20],
        &[68, 33, 0, 79, 27, 69, 9, 12, 77, 22, 46, 1, 52, 29, 90, 47],
        &[13, 56, 46, 69, 9, 61, 11, 27, 11, 81, 88, 80, 85, 71, 71, 94],
        &[42, 10, 46, 85, 93, 14, 37, 18, 75, 84, 64, 9, 95, 33, 12, 17],
        &[14, 67, 5, 71, 6, 85, 81, 7, 41, 66, 34, 26, 34, 68, 94, 70],
    ];
    let mut g = GramBuilder::default();
    for v in M16 {
        assert!(g.add(v));
    }
    let det_approx = 1.2587314158510076e+30;
    assert!((g.det_estimate() - det_approx).abs() < 1e16);

    assert_eq!(
        det_matz(M16.to_vec(), det_approx),
        Int::from_str("-1258731415851007568569087128744").unwrap()
    );
}
