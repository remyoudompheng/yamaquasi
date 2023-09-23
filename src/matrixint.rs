// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Linear algebra routines for matrices over
//! the ring of integers and GF(p)
//! This is used for class group computations, for medium-sized
//! matrices (at most size 1000).

use std::collections::BTreeMap;

use bnum::cast::CastFrom;
use bnum::types::{I256, I4096, U256};
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
pub fn det_matz(mat: Vec<&[i64]>, log2estimate: f64) -> I4096 {
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
    assert!(63 * modp.len() >= bits + 2);
    let det = crt(&modp);
    // Check result using floating-point estimate.
    if det.to_f64().unwrap().is_finite() {
        let diff = det.to_f64().unwrap().abs().log2() - log2estimate;
        assert!(diff.abs() < 1e-6, "logdiff={diff}");
    }
    det
}

fn crt(modp: &[u64]) -> I4096 {
    let mut crt_basis = vec![I4096::ONE; modp.len()];
    for i in 0..modp.len() {
        let mut inv: u64 = 1;
        let pi = PRIMES[i];
        for j in 0..modp.len() {
            if i != j {
                crt_basis[i] *= I4096::from(PRIMES[j]);
                let inv_j = arith::inv_mod64(PRIMES[j], pi).unwrap();
                inv = ((inv as u128 * inv_j as u128) % (pi as u128)) as u64;
            }
        }
        crt_basis[i] *= I4096::from(inv);
    }
    let mut prod = I4096::ONE;
    for &p in &PRIMES[..modp.len()] {
        prod *= I4096::from(p);
    }
    let mut det = I4096::ZERO;
    for (&deti, bi) in modp.iter().zip(&crt_basis) {
        det += I4096::from(deti) * bi;
    }
    det %= prod;
    let det = if det > (prod >> 1) {
        I4096::cast_from(det) - I4096::cast_from(prod)
    } else {
        I4096::cast_from(det)
    };
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
const PRIMES: [u64; 64] = [
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
    0x7ffffffffffffaed,
    0x7ffffffffffffae3,
    0x7ffffffffffffab3,
    0x7ffffffffffffa8d,
    0x7ffffffffffffa45,
    0x7ffffffffffffa2f,
    0x7ffffffffffffa23,
    0x7ffffffffffffa05,
    0x7ffffffffffff9f1,
    0x7ffffffffffff9e7,
    0x7ffffffffffff9d9,
    0x7ffffffffffff9b7,
    0x7ffffffffffff9a3,
    0x7ffffffffffff99d,
    0x7ffffffffffff925,
    0x7ffffffffffff8ef,
    0x7ffffffffffff8d9,
    0x7ffffffffffff8c1,
    0x7ffffffffffff88b,
    0x7ffffffffffff86b,
    0x7ffffffffffff817,
    0x7ffffffffffff787,
    0x7ffffffffffff739,
    0x7ffffffffffff735,
    0x7ffffffffffff70f,
    0x7ffffffffffff703,
    0x7ffffffffffff6f1,
    0x7ffffffffffff6e5,
    0x7ffffffffffff6c3,
    0x7ffffffffffff6b5,
    0x7ffffffffffff69f,
    0x7ffffffffffff669,
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

    fn truncate(&mut self, n: usize) {
        self.basis.truncate(n);
        self.indices.truncate(n);
        self.factors.truncate(n);
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

/// A structure holding an echelonized matrix of size (n-1) x n
/// modulo several primes. This is used to share computations
/// in compute_lattice_index.
///
/// Its memory usage is usually O(n^3).
struct CRTDetBuilder<'a> {
    rows: Vec<&'a [i64]>,
    echelons: Vec<GFpEchelonBuilder>,
}

impl<'a> CRTDetBuilder<'a> {
    fn new(rows: Vec<&'a [i64]>) -> Self {
        Self {
            rows,
            echelons: vec![],
        }
    }

    fn det(&mut self, nth_row: &[i64], log2estimate: f64) -> I4096 {
        let bits = log2estimate.round();
        assert!(bits >= 1.0);
        assert!(
            bits <= 63.0 * PRIMES.len() as f64,
            "determinant is too large {log2estimate:.1} bits"
        );
        let bits = bits as usize;
        let mut modp = vec![];
        'crtloop: while 60 * modp.len() < bits && modp.len() < PRIMES.len() {
            if self.echelons.len() <= modp.len() {
                let p = PRIMES[modp.len()];
                self.echelons.push(GFpEchelonBuilder::new(p));
            }
            let mp = &mut self.echelons[modp.len()];
            if mp.basis.len() > self.rows.len() {
                mp.truncate(self.rows.len());
            }
            for r in &self.rows[mp.basis.len()..] {
                if !mp.add(r) {
                    // Determinant is zero mod p
                    modp.push(0);
                    continue 'crtloop;
                }
            }
            mp.add(nth_row);
            let dp = mp.det();
            modp.push(dp);
        }
        // Now apply CRT
        assert!(63 * modp.len() >= bits + 2);
        let det = crt(&modp);
        // Check result using floating-point estimate.
        if det.to_f64().unwrap().is_finite() {
            let diff = det.to_f64().unwrap().abs().log2() - log2estimate;
            assert!(diff.abs() < 1e-6, "logdiff={diff}");
        }
        det
    }
}

/// Minimal distance for considering a vector independent
/// from an existing subspace.
const LATTICE_MINDIST: f64 = 0.1;

/// Compute the index of a lattice generated by an overdetermined basis.
/// Estimates are given as loose lower/upper bounds.
pub fn compute_lattice_index(rows: &Vec<Vec<i64>>, hmin: f64, hmax: f64) -> u128 {
    // The bounds can be very approximate.
    let hmin = 0.95 * hmin;
    let hmax = 1.05 * hmax;
    assert!(hmin < hmax);
    assert!(hmax / hmin < 1.5);
    assert!(hmax.log2() < 126.0);
    let mut rows: Vec<&[i64]> = rows.iter().map(|v| &v[..]).collect();
    rows.sort_by_cached_key(|x| x.iter().map(|&y| y * y).sum::<i64>());
    let dim = rows[0].len();
    let mut g = GramBuilder::default();
    let mut builder = None;
    let mut indices = vec![];
    g.threshold = Some(LATTICE_MINDIST * LATTICE_MINDIST);
    let mut gcd = I4096::ZERO;
    for idx in 0..rows.len() {
        if g.rank() == dim - 1 {
            if builder.is_none() {
                let mat: Vec<&[i64]> = indices.iter().map(|&idx| rows[idx]).collect();
                builder = Some(CRTDetBuilder::new(mat));
            }
            let b = builder.as_mut().unwrap();
            let mut gg = g.clone();
            if gg.add(&rows[idx]) {
                indices.push(idx);
                let logest = gg.detlog2_estimate();
                let det = b.det(&rows[idx], logest);
                //eprintln!("det={} log2={:.6}", det, logest);
                //eprintln!("indices = {indices:?}");
                indices.pop();
                // Analyse candidates
                gcd = Integer::gcd(&gcd, &det);
                let gcd_f = gcd.to_f64().unwrap();
                if gcd_f / hmin > 1e4 {
                    continue;
                }
                let m1 = (gcd_f / hmax).round() as i64;
                let m2 = (gcd_f / hmin).round() as i64;
                let mut candidates = vec![];
                for m in m1..m2 + 1 {
                    if m <= 0 {
                        continue;
                    }
                    let (q, r) = gcd.div_rem(&I4096::from(m));
                    let qf = q.to_f64().unwrap();
                    if r.is_zero() && hmin <= qf && qf <= hmax {
                        candidates.push(q)
                    }
                }
                //eprintln!("gcd={gcd} h in {:?}", &candidates);
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
///
/// For a relation matrix R, the output is (D, Q) such that
/// D = P R Q^-1 is diagonal for P,Q invertible integer matrices.
pub struct SmithNormalForm {
    pub rows: Vec<Vec<i128>>,
    // A transformation matrix.
    pub q: Vec<Vec<i128>>,
    pub gens: Vec<u32>,
    pub removed: Vec<(u32, Vec<(u32, i128)>)>,
    pub h: u128,
    pub hinv: (u128, i32),
    pub verbose: bool,
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
        let mut gens: Vec<_> = gens.into_keys().collect();
        assert!(rels.len() >= gens.len());
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
            // Put generators are in decreasing order.
            v.reverse();
            rows.push(v);
        }
        gens.reverse();
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
        let hinv = Self::divider(h);
        Self {
            rows,
            q: vec![],
            gens,
            removed,
            h,
            hinv,
            verbose: false,
        }
    }

    pub fn reduce(&mut self) {
        self.reduce_rows();
        self.reduce_cols();
        let mut det: i128 = 1;
        for i in 0..self.rows.len() {
            if self.rows[i][i] == 0 {
                self.rows[i][i] = self.h as i128;
            }
            det *= self.rows[i][i];
        }
        assert_eq!(det, self.h as i128);
    }

    #[allow(unused)]
    fn debug(&self) {
        eprintln!("M =");
        for r in &self.rows {
            eprintln!("{r:?}");
        }
        eprintln!("Q =");
        for r in &self.q {
            eprintln!("{r:?}");
        }
    }

    /// Compute a reduced normal form using row operations only.
    /// Usually this will not produce the actual Smith normal form
    /// with d[i] dividing d[i+1].
    /// The goal is to eliminate as many generators as possible
    /// to keep a very small matrix.
    fn reduce_rows(&mut self) {
        // Make matrix triangular
        let n = self.gens.len();
        for j in 0..n {
            // Invariant
            // for i in 0..j
            // row[i,i] = Di
            // row[k>i,i] = 0
            for i in j..self.rows.len() {
                for jj in 0..j {
                    self.eliminate(jj, i, jj);
                    assert!(self.rows[i][jj] == 0);
                }
                if self.rows[i][j] != 0 {
                    //eprintln!("using row {i} for step {j}");
                    self.rows.swap(j, i);
                    break;
                }
            }
            //eprintln!("reduction {j} {:?}", self.rows[j]);
            if self.rows[j][j] != 0 {
                self.normalize(j, j);
            }
        }
        //let diag: Vec<i128> = (0..n).map(|j| self.rows[j][j]).collect();
        //eprintln!("diag {diag:?}")
        // Add some vectors to reduce until class number is reached.
        for i in n..self.rows.len() {
            for j in 0..n {
                if self.rows[i][j] == 0 {
                    continue;
                }
                self.eliminate(j, i, j);
            }
            if self.rows[n - 1][n - 1] == 0 {
                self.rows[n - 1][n - 1] = self.h as i128;
            }
            let diag: Vec<i128> = (0..n).map(|j| self.rows[j][j]).collect();
            //eprintln!("diag {diag:?}");
            let prod = diag.iter().product::<i128>();
            if prod == self.h as i128 {
                if self.verbose {
                    eprintln!("Found basis of relation lattice");
                }
                break;
            }
        }
        // Discard unneeded relations
        self.rows.truncate(n);
        // Now the matrix is square and upper triangular.
        // Remove upper coefficients and redundant generators
        // starting from the last column.
        let n = self.gens.len();
        let mut remaining = vec![];
        let mut cols = vec![];
        for j in 0..n {
            if self.rows[j][j] != 0 {
                self.normalize(j, j);
            }
            let nj = self.rows[j][j];
            for i in 0..j {
                if nj == 1 || (nj > 0 && self.rows[i][j] % nj == 0) {
                    self.eliminate(j, i, j);
                }
            }
        }

        // Extract redundant relations (coefficient=1).
        for j in 0..n {
            if self.rows[j][j] == 1 {
                let p = self.gens[j];
                let mut rel = vec![];
                for idx in j + 1..n {
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
                cols.push(j);
            }
        }
        // Shrink matrix
        self.rows = vec![];
        for r in remaining {
            self.rows.push(cols.iter().map(|&idx| r[idx]).collect());
        }
        self.gens = cols.iter().map(|&idx| self.gens[idx]).collect();
        assert_eq!(self.rows.len(), self.gens.len());
    }

    /// Reduce columns to obtain the final Smith normal form
    /// The input matrix is small and upper triangular.
    fn reduce_cols(&mut self) {
        // If input matrix is U such that U gens = 0, the output is:
        // (P U Q, Q) where P,Q are invertible
        // and P U Q is diagonal.
        //
        // Then Q^-1 gens are the basis of the abelian group
        // meaning that the columns of Q are their coordinates.
        // We don't need the value of matrix P for group calculations.
        //
        // At each step, reduce columns so that:
        // for all i < j M[i,j] < M[i,i]
        // Then repeat row reduction.
        let n = self.rows.len();
        // Initialize q
        self.q.clear();
        for i in 0..n {
            let mut v = vec![0; n];
            v[i] = 1;
            self.q.push(v);
        }
        // is it necessary to loop here??
        for _ in 0..10 {
            let mut changes = 0;
            for i in 0..n {
                for j in i + 1..n {
                    while self.rows[i][j].abs() > self.rows[i][i].abs() {
                        let k = self.rows[i][j].div_euclid(self.rows[i][i]);
                        self.colsub(j, i, k);
                        changes += 1;
                        if self.rows[i][j] != 0 {
                            // The GCD M[i,j] M[j,j] must be less than M[i,i]
                            // Swap columns i and j, reduce.
                            self.colswap(i, j);
                        }
                    }
                    if self.rows[i][i] != 0 && self.rows[j][j] % self.rows[i][i] != 0 {
                        // Enforce divisibility relation.
                        self.colsub(j, i, 1);
                        self.colswap(i, j);
                    }
                }
            }
            if changes == 0 {
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
            for k in 0..self.gens.len() {
                let vi = self.rows[i][k];
                self.rows[i][k] = self.modh256(I256::from(vi) * I256::from(e.x));
            }
        }
    }

    /// Subtract k times column j from column i
    fn colsub(&mut self, i: usize, j: usize, k: i128) {
        if k == 0 {
            return;
        }
        for idx in 0..self.gens.len() {
            let (yi, yj) = (self.rows[idx][i], self.rows[idx][j]);
            let (qi, qj) = (self.q[idx][i], self.q[idx][j]);
            if self.h > 0 && self.h < 1 << 64 {
                self.rows[idx][i] = self.modh128(yi - k * yj);
                self.q[idx][i] = self.modh128(qi - k * qj);
            } else {
                let x = I256::from(yi) - I256::from(k) * I256::from(yj);
                self.rows[idx][i] = self.modh256(x);
                let x = I256::from(qi) - I256::from(k) * I256::from(qj);
                self.q[idx][i] = self.modh256(x);
            };
        }
    }

    /// Swap columns and make triangular again.
    fn colswap(&mut self, i: usize, j: usize) {
        for r in self.rows.iter_mut() {
            r.swap(i, j);
        }
        for rq in self.q.iter_mut() {
            rq.swap(i, j);
        }
        // Reduce rows in block i..=j
        // This has no effect on matrix Q.
        for k in i..j {
            for l in k + 1..=j {
                self.eliminate(k, l, k);
            }
            self.normalize(k, k);
            if self.rows[k][k] == 0 {
                self.rows[k][k] = self.h as i128;
            }
        }
        self.normalize(j, j);
        if self.rows[j][j] == 0 {
            self.rows[j][j] = self.h as i128;
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
        if xi == 1 {
            // Common situation
            for idx in 0..self.gens.len() {
                let (yi, yj) = (self.rows[i][idx], self.rows[j][idx]);
                if yi == 0 && yj == 0 {
                    continue;
                }
                if self.h > 0 && self.h < 1 << 64 {
                    self.rows[j][idx] = self.modh128(yj - xj * yi);
                } else {
                    let x = I256::from(yj) - I256::from(xj) * I256::from(yi);
                    self.rows[j][idx] = self.modh256(x);
                };
            }
        } else {
            let e = Integer::extended_gcd(&xi, &xj);
            assert!(e.gcd != 0);
            let (a, b, c, d) = (e.x, e.y, -xj / e.gcd, xi / e.gcd);
            for idx in 0..self.gens.len() {
                let (x, y) = (self.rows[i][idx], self.rows[j][idx]);
                if x == 0 && y == 0 {
                    continue;
                }
                if self.h > 0 && self.h < 1 << 64 {
                    let xx = a * x + b * y;
                    let yy = c * x + d * y;
                    self.rows[i][idx] = xx.rem_euclid(self.h as i128);
                    self.rows[j][idx] = yy.rem_euclid(self.h as i128);
                } else {
                    let xx = I256::from(a) * I256::from(x) + I256::from(b) * I256::from(y);
                    let yy = I256::from(c) * I256::from(x) + I256::from(d) * I256::from(y);
                    self.rows[i][idx] = self.modh256(xx);
                    self.rows[j][idx] = self.modh256(yy);
                };
            }
        }
    }

    fn divider(h: u128) -> (u128, i32) {
        assert!(h < 1 << 125);
        // Compute a rounded value for 2^255/h
        let q: U256 = (U256::ONE << 255) / U256::from(h);
        // Keep top 127 bits
        let qlen = q.bits();
        if qlen <= 127 {
            (u128::cast_from(q), -255)
        } else {
            let shift = qlen - 127;
            let round = u128::from(q.bit(shift - 1));
            (u128::cast_from(q >> shift) + round, shift as i32 - 255)
        }
    }

    // Reduction modulo h using precomputed divider.
    fn modh128(&self, x: i128) -> i128 {
        let h = self.h as i128;
        let (qm, qe) = self.hinv;
        let q = (I256::from(x) * I256::from(qm)) >> (-qe);
        let q = i128::cast_from(q);
        let mut rem = x - q * h;
        while rem >= h {
            rem -= h;
        }
        while rem < 0 {
            rem += h;
        }
        rem
    }

    // Reduction modulo h using precomputed divider.
    fn modh256(&self, x: I256) -> i128 {
        if x.is_negative() {
            self.h as i128 - self.modh256u(x.unsigned_abs())
        } else {
            self.modh256u(x.unsigned_abs())
        }
    }

    fn modh256u(&self, x: U256) -> i128 {
        let h = self.h as i128;
        let (qm, qe) = self.hinv;
        // We only care about top 128 bits.
        let xw: &[u64; 4] = x.digits();
        let q = if xw[2] == 0 && xw[3] == 0 {
            u128::cast_from((x * U256::cast_from(qm)) >> (-qe))
        } else {
            let shift = x.bits() as i32 - 128;
            u128::cast_from(((x >> shift) * U256::cast_from(qm)) >> (-qe - shift))
        };
        let mut rem = i128::cast_from(I256::cast_from(x) - I256::from(q) * I256::from(h));
        while rem >= h {
            rem -= h;
        }
        while rem < 0 {
            rem += h;
        }
        rem
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
        I4096::from(14293689752795_i64)
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
        I4096::from_str("-1258731415851007568569087128744").unwrap()
    );
}
