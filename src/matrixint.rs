// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Linear algebra routines for matrices over
//! the ring of integers and GF(p)
//! This is used for class group computations, for medium-sized
//! matrices (at most size 1000).

use std::collections::BTreeMap;

use bnum::cast::CastFrom;
use bnum::types::{I2048, I256};
use num_integer::Integer;
use num_traits::ToPrimitive;

use crate::arith;
use crate::arith_montgomery::{mg_2adic_inv, mg_mul, mg_redc};

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

    pub fn detlog2_estimate(&self) -> f64 {
        self.norms.iter().map(|&x| x.log2()).sum::<f64>() / 2.0
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
pub fn det_matz(mat: Vec<&[i64]>, log2estimate: f64) -> I2048 {
    let bits = log2estimate.round();
    assert!(bits >= 1.0);
    assert!(
        bits <= 63.0 * PRIMES.len() as f64,
        "determinant is too large {log2estimate:.1} bits"
    );
    let bits = bits as usize;
    let mut modp = vec![];
    'crtloop: while 60 * modp.len() < bits && modp.len() < PRIMES.len() {
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
    let mut crt_basis = vec![I2048::ONE; modp.len()];
    for i in 0..modp.len() {
        let mut inv: u64 = 1;
        let pi = PRIMES[i];
        for j in 0..modp.len() {
            if i != j {
                crt_basis[i] *= I2048::from(PRIMES[j]);
                let inv_j = arith::inv_mod64(PRIMES[j], pi).unwrap();
                inv = ((inv as u128 * inv_j as u128) % (pi as u128)) as u64;
            }
        }
        crt_basis[i] *= I2048::from(inv);
    }
    let mut prod = I2048::ONE;
    for &p in &PRIMES[..modp.len()] {
        prod *= I2048::from(p);
    }
    assert!(prod.bits() as usize >= bits + 2);
    let mut det = I2048::ZERO;
    for (&deti, bi) in modp.iter().zip(&crt_basis) {
        det += I2048::from(deti) * bi;
    }
    det %= prod;
    let det = if det > (prod >> 1) {
        I2048::cast_from(det) - I2048::cast_from(prod)
    } else {
        I2048::cast_from(det)
    };
    // Check result using floating-point estimate.
    if det.to_f64().unwrap().is_finite() {
        let diff = det.to_f64().unwrap().abs().log2() - log2estimate;
        assert!(diff.abs() < 1e-6, "logdiff={diff}");
    }
    det
}

/// Builder for echelonized bases over finite field GF(p).
/// Modulus is usually around 60 bits.
struct GFpEchelonBuilder {
    p: u64,
    pinv: u64,
    r: u64,  // 2^64 % p
    r2: u64, // 2^128 % p
    indices: Vec<usize>,
    // All values are stored in Montgomery form.
    basis: Vec<Vec<u64>>,
    factors: Vec<u64>,
}

// A few large 63-bit primes, enough to compute 1024-bit determinants.
const PRIMES: [u64; 32] = [
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
    0x7ffffffffffffd21,
    0x7ffffffffffffd11,
    0x7ffffffffffffcaf,
    0x7ffffffffffffc99,
    0x7ffffffffffffc85,
    0x7ffffffffffffc6d,
    0x7ffffffffffffc0d,
    0x7ffffffffffffbd3,
    0x7ffffffffffffbb9,
    0x7ffffffffffffb97,
    0x7ffffffffffffb65,
    0x7ffffffffffffb3b,
    0x7ffffffffffffb2b,
    0x7ffffffffffffb1f,
    0x7ffffffffffffaef,
];

impl GFpEchelonBuilder {
    fn new(p: u64) -> Self {
        let pinv = mg_2adic_inv(p);
        let r = (1u128 << 64).rem_euclid(p as u128) as u64;
        let r2 = (r as u128 * r as u128).rem_euclid(p as u128) as u64;
        Self {
            p,
            pinv,
            r,
            r2,
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
        let p = self.p as i64;
        for i in 0..v.len() {
            let mut vi = v[i];
            while vi < 0 {
                vi += p;
            }
            while vi >= p {
                vi -= p;
            }
            vp[i] = mg_mul(self.p, self.pinv, vi as u64, self.r2);
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
                assert_eq!(vp[i], self.r);
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
        // mm = m/R
        let mm = mg_redc(self.p, self.pinv, m as u128);
        let mminv = arith::inv_mod64(mm, p).unwrap();
        // minv = R/mm = R^2/m
        let minv = mg_mul(self.p, self.pinv, mminv, self.r2);
        for vi in v {
            *vi = mg_mul(self.p, self.pinv, *vi, minv);
        }
    }

    fn submul(&self, v: &mut [u64], w: &[u64], m: u64) {
        let p = self.p;
        for (i, vi) in v.iter_mut().enumerate() {
            let mw = mg_mul(self.p, self.pinv, m, w[i]);
            if *vi >= mw {
                *vi -= mw;
            } else {
                *vi += p - mw;
            }
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
        // Multiply diagonal elements
        let mut det = self.factors[0];
        for &f in &self.factors[1..] {
            det = mg_mul(p, self.pinv, det, f);
        }
        det = mg_redc(p, self.pinv, det as u128);
        if swaps % 2 == 1 && det > 0 {
            det = p - det
        }
        det
    }
}

/// Compute the index of a lattice generated by an overdetermined basis.
/// Estimates are given as loose lower/upper bounds.
pub fn compute_lattice_index(rows: &Vec<Vec<i64>>, hmin: f64, hmax: f64) -> u128 {
    assert!(hmin < hmax);
    assert!(hmax / hmin < 1.5);
    assert!(hmax.log2() < 126.0);
    let mut rows: Vec<&[i64]> = rows.iter().map(|v| &v[..]).collect();
    rows.sort_by_cached_key(|x| x.iter().map(|&y| y * y).sum::<i64>());
    let dim = rows[0].len();
    let mut g = GramBuilder::default();
    let mut indices = vec![];
    g.threshold = Some(1.0);
    let mut gcd = I2048::ZERO;
    for idx in 0..rows.len() {
        if g.rank() == dim - 1 {
            let mut gg = g.clone();
            if gg.add(&rows[idx]) {
                indices.push(idx);
                let mat: Vec<&[i64]> = indices.iter().map(|&idx| &rows[idx][..]).collect();
                let det = det_matz(mat, gg.detlog2_estimate());
                eprintln!("det={} estimate={:e}", det, gg.det_estimate());
                //eprintln!("indices = {indices:?}");
                indices.pop();
                // Analyse candidates
                gcd = Integer::gcd(&gcd, &det);
                let gcd_f = gcd.to_f64().unwrap();
                if gcd_f / hmin > 1e6 {
                    continue;
                }
                let m1 = (gcd_f / hmax).round() as i64;
                let m2 = (gcd_f / hmin).round() as i64;
                let mut candidates = vec![];
                for m in m1..m2 + 1 {
                    if m <= 0 {
                        continue;
                    }
                    let (q, r) = gcd.div_rem(&I2048::from(m));
                    if r.is_zero() {
                        candidates.push(q)
                    }
                }
                eprintln!("gcd={gcd} h in {:?}", &candidates);
                if candidates.len() == 1 {
                    return u128::cast_from(candidates[0]);
                }
            }
        } else {
            if g.add(&rows[idx]) {
                indices.push(idx);
            }
        }
    }
    panic!("failed to determine lattice index");
}

/// A context for computation of the Smith normal form
/// for a presentation of a finite abelian group with
/// known order h.
pub struct SmithNormalForm {
    pub rows: Vec<Vec<i128>>,
    pub gens: Vec<u32>,
    pub removed: Vec<(u32, Vec<(u32, i128)>)>,
    pub h: u128,
}

impl SmithNormalForm {
    pub fn new(
        rels: &[Vec<(u32, i32)>],
        removed: Vec<(u32, Vec<(u32, i32)>)>,
        hmin: f64,
        hmax: f64,
    ) -> Self {
        let mut gens = BTreeMap::new();
        for row in rels {
            for &(p, _) in row.iter() {
                gens.insert(p, p);
            }
        }
        let gens: Vec<_> = gens.into_keys().collect();
        let mut rows = vec![];
        for row in rels {
            if row.len() == 0 {
                continue;
            }
            let mut v = vec![0 as i64; gens.len()];
            let mut j = 0;
            for (i, &p) in gens.iter().enumerate() {
                if j >= row.len() {
                    break;
                }
                while j < row.len() && row[j].0 < p {
                    j += 1
                }
                if j < row.len() && row[j].0 == p {
                    v[i] = row[j].1.into();
                }
            }
            rows.push(v);
        }
        let h = compute_lattice_index(&rows, hmin, hmax);
        // Convert to 128-bits.
        let rows = rows
            .into_iter()
            .map(|v| v.into_iter().map(|x| x as i128).collect())
            .collect();
        let removed: Vec<(u32, Vec<(u32, i128)>)> = removed
            .iter()
            .map(|(p, v)| (*p, v.iter().map(|&(l, e)| (l, e as i128)).collect()))
            .collect();
        Self {
            rows,
            gens,
            removed,
            h,
        }
    }

    pub fn reduce(&mut self) {
        // Make matrix triangular
        let n = self.gens.len() - 1;
        for j in 0..self.gens.len() {
            // Invariant
            // for i in 0..j
            // row[i,n-i] = Di
            // row[k>i,n-i] = 0
            for i in j..self.rows.len() {
                for jj in 0..j {
                    self.eliminate(jj, i, n - jj);
                    assert!(self.rows[i][n - jj] == 0);
                }
                if self.rows[i][n - j] != 0 {
                    //eprintln!("using row {i} for step {j}");
                    self.rows.swap(j, i);
                    break;
                }
            }
            //eprintln!("reduction {j} {:?}", self.rows[j]);
            if self.rows[j][n - j] != 0 {
                self.normalize(j, n - j);
            }
            if j < n {
                assert!(self.rows[j][n - j] != 0);
            } else {
                if self.rows[n][0] == 0 {
                    self.rows[n][0] = self.h as i128;
                }
            }
        }
        //let diag: Vec<i128> = (0..=n).map(|j| self.rows[j][n - j]).collect();
        //eprintln!("diag {diag:?}");
        // Add some vectors to reduce until class number is reached.
        for i in n + 1..self.rows.len() {
            for j in 0..=n {
                if self.rows[i][n - j] == 0 {
                    continue;
                }
                self.eliminate(j, i, n - j);
            }
            if self.rows[n][0] == 0 {
                self.rows[n][0] = self.h as i128;
            }
            let diag: Vec<i128> = (0..=n).map(|j| self.rows[j][n - j]).collect();
            //eprintln!("diag {diag:?}");
            let prod = diag.iter().product::<i128>();
            if prod == self.h as i128 {
                eprintln!("Found basis of relation lattice");
                break;
            }
        }
        // Discard unneeded relations
        self.rows.truncate(n + 1);
        // Now the matrix is square and upper triangular.
        // Remove upper coefficients and redundant generators
        // starting from the last column.
        loop {
            let n = self.gens.len() - 1;
            let mut remaining = vec![];
            let mut cols = vec![];
            for j in 0..=n {
                let nj = self.rows[j][n - j];
                for i in 0..j {
                    if nj == 1 || (nj > 0 && self.rows[i][n - j] % nj == 0) {
                        self.eliminate(j, i, n - j);
                    }
                }
            }
            // Extract redundant relations (coefficient=1).
            for j in 0..=n {
                if self.rows[j][n - j] == 1 {
                    let p = self.gens[n - j];
                    let mut rel = vec![];
                    for idx in 0..n - j {
                        let e = self.rows[j][idx];
                        if e == 0 {
                            continue;
                        }
                        if e.unsigned_abs() < self.h / 2 {
                            rel.push((self.gens[idx], -e));
                        } else {
                            rel.push((self.gens[idx], self.h as i128 - e));
                        }
                    }
                    self.removed.push((p, rel));
                } else {
                    remaining.push(self.rows[j].clone());
                    cols.push(n - j);
                }
            }
            // Collect remaining entries, put GCD=1 last
            // then largest diagonal entries.
            let mut order = vec![];
            for i in 0..cols.len() {
                let idx = cols[i];
                let mut gcd: i128 = self.h as i128;
                for r in &remaining {
                    gcd = Integer::gcd(&gcd, &r[idx]);
                }
                order.push((gcd, idx));
            }
            order.sort();
            order.reverse();
            //eprintln!("columns {order:?}");
            let cols: Vec<usize> = order.iter().map(|&(_, idx)| idx).collect();
            self.rows = vec![];
            for r in remaining {
                self.rows.push(cols.iter().map(|&idx| r[idx]).collect());
            }
            self.gens = cols.iter().map(|&idx| self.gens[idx]).collect();
            assert_eq!(self.rows.len(), self.gens.len());
            // Perform reduction again
            let rank = self.gens.len();
            for i in 0..rank {
                for j in i + 1..rank {
                    self.eliminate(i, j, rank - 1 - i);
                }
                self.normalize(i, rank - 1 - i);
            }
            if self.rows[rank - 1][0] == 0 {
                self.rows[rank - 1][0] = self.h as i128;
            }
            if order.last().unwrap().0 > 1 {
                break;
            }
        }
    }

    // Normalize coefficient M[i,k] to divide h.
    fn normalize(&mut self, i: usize, k: usize) {
        // Make sure we have a divisor of h
        let e = Integer::extended_gcd(&self.rows[i][k], &(self.h as i128));
        if self.rows[i][k] == e.gcd {
            return;
        }
        if self.h > 0 && self.h < 1 << 64 {
            for vi in self.rows[i].iter_mut() {
                *vi = (*vi * e.x).rem_euclid(self.h as i128);
            }
        } else {
            for vi in self.rows[i].iter_mut() {
                *vi = i128::cast_from((I256::from(*vi) * I256::from(e.x)) % I256::from(self.h));
            }
        }
    }

    /// Eliminate coefficient M[j,k] using a combination with nonzero M[i,k].
    fn eliminate(&mut self, i: usize, j: usize, k: usize) {
        assert!(i != j);
        let xi = self.rows[i][k];
        let xj = self.rows[j][k];
        if xj == 0 {
            return;
        }
        let (ri, rj) = if i < j {
            let (mi, mj) = self.rows.split_at_mut(j);
            (&mut mi[i][..], &mut mj[0][..])
        } else {
            let (mj, mi) = self.rows.split_at_mut(i);
            (&mut mi[0][..], &mut mj[j][..])
        };
        if xi == 1 {
            // Common situation
            let ri = ri as &[i128];
            for idx in 0..self.gens.len() {
                let (yi, yj) = (ri[idx], rj[idx]);
                if yi == 0 && yj == 0 {
                    continue;
                }
                if self.h > 0 && self.h < 1 << 64 {
                    rj[idx] = (yj - xj * yi).rem_euclid(self.h as i128);
                } else {
                    let x = I256::from(yj) - I256::from(xj) * I256::from(yi);
                    rj[idx] = i128::cast_from(x % I256::from(self.h));
                };
            }
        } else {
            let e = Integer::extended_gcd(&xi, &xj);
            assert!(e.gcd != 0);
            let (a, b, c, d) = (e.x, e.y, -xj / e.gcd, xi / e.gcd);
            for idx in 0..self.gens.len() {
                let (x, y) = (ri[idx], rj[idx]);
                if x == 0 && y == 0 {
                    continue;
                }
                if self.h > 0 && self.h < 1 << 64 {
                    let xx = a * x + b * y;
                    let yy = c * x + d * y;
                    ri[idx] = xx.rem_euclid(self.h as i128);
                    rj[idx] = yy.rem_euclid(self.h as i128);
                } else {
                    let xx = I256::from(a) * I256::from(x) + I256::from(b) * I256::from(y);
                    let yy = I256::from(c) * I256::from(x) + I256::from(d) * I256::from(y);
                    ri[idx] = i128::cast_from(xx % I256::from(self.h));
                    rj[idx] = i128::cast_from(yy % I256::from(self.h));
                };
            }
        }
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
        det_matz(M.to_vec(), det_approx.log2()),
        I2048::from(14293689752795_i64)
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
        det_matz(M16.to_vec(), det_approx.log2()),
        I2048::from_str("-1258731415851007568569087128744").unwrap()
    );
}
