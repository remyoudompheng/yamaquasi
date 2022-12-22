// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Relations describe an equation:
//! x^2 = product(pi^ki) mod n
//!
//! where pi = -1 or a prime in the factor base

use std::collections::HashMap;
use std::default::Default;

use bitvec_simd::BitVec;
use bnum::cast::CastFrom;
use num_integer::Integer;
use num_traits::One;

use crate::arith::{pow_mod, U512};
use crate::matrix;
use crate::{Int, Uint};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relation {
    pub x: Uint,
    pub cofactor: u64,
    // Is this relation from a double-large prime relation?
    pub pp: bool,
    pub factors: Vec<(i64, u64)>, // -1 for the sign
}

impl Relation {
    pub fn verify(&self, n: &Uint) -> bool {
        let mut prod = Uint::from(self.cofactor);
        for &(p, k) in self.factors.iter() {
            if p == -1 {
                if k % 2 == 1 {
                    prod = n - prod;
                }
            } else {
                assert!(p > 0);
                prod = (prod * pow_mod(Uint::from(p as u64), Uint::from(k), *n)) % n;
            }
        }
        (self.x * self.x) % n == prod
    }

    fn pack(self) -> PackedRelation {
        PackedRelation::pack(self)
    }
}
/// A RelationSet collects all relations x^2 = product(small primes) * cofactor
/// encountered during sieve.
///
/// We don't implement cycle finding to obtain smooth relations from a cycle
/// of "double large relations".
/// Instead, a partial relation ("p") and a double large relation ("pp")
/// can be combined to a partial relation (cofactor is a large prime).
/// A pp-relation can be combined with 2 matching p-relations
/// to produce a complete relation.
/// This is equivalent to finding cycles only in the connected component
/// of single prime relations.
///
/// Empirical studies show that non-trivial cycles never appear: all cycles
/// intersect the (huge) connected component of p-relations.
///
#[derive(Default)]
pub struct RelationSet {
    pub n: Uint,
    pub maxlarge: u64,
    pub complete: Vec<Relation>,
    // p => relation with cofactor p
    partial: HashMap<u64, PackedRelation>,
    // p => relation with cofactor pq (p < q)
    // No key is common with partial map
    doubles: HashMap<(u64, u64), PackedRelation>,
    pub n_smooths: usize,
    pub n_partials: usize,
    pub n_doubles: usize,
    pub n_combined: usize,
    pub n_combined2: usize,
    pub n_combined12: usize,
}

impl RelationSet {
    pub fn new(n: Uint, maxlarge: u64) -> Self {
        RelationSet {
            n,
            maxlarge,
            ..Default::default()
        }
    }

    // Consumes the set and returns the inner vector.
    pub fn into_inner(self) -> Vec<Relation> {
        self.complete
    }

    pub fn n_complete(&self) -> usize {
        self.complete.len()
    }

    pub fn truncate(&mut self, l: usize) {
        self.complete.truncate(l)
    }

    pub fn len(&self) -> usize {
        self.complete.len()
    }

    pub fn gap(&self) -> usize {
        relation_gap(&self.complete)
    }

    pub fn log_progress<S: AsRef<str>>(&self, prefix: S) {
        eprintln!(
            "{} found {} smooths (c0={} c1={} c2={} p={} p12={} pp={})",
            prefix.as_ref(),
            self.len(),
            self.n_smooths,
            self.n_combined,
            self.n_combined2,
            self.n_partials,
            self.n_combined12,
            self.n_doubles,
        )
    }

    /// Combine 2 relations sharing a cofactor.
    pub fn combine(&self, r1: &Relation, r2: &Relation) -> Relation {
        // Combine factors
        let mut exps = HashMap::<i64, u64>::new();
        for (p, k) in &r1.factors {
            let e = exps.get(&p).unwrap_or(&0);
            exps.insert(*p, e + k);
        }
        for (p, k) in &r2.factors {
            let e = exps.get(&p).unwrap_or(&0);
            exps.insert(*p, e + k);
        }
        let mut factors: Vec<_> = exps.into_iter().collect();
        // Combine cofactors
        if r1.cofactor % r2.cofactor == 0 {
            factors.push((r2.cofactor as i64, 2));
            Relation {
                x: (r1.x * r2.x) % self.n,
                cofactor: r1.cofactor / r2.cofactor,
                pp: r1.pp || r2.pp,
                factors,
            }
        } else {
            assert!(r2.cofactor % r1.cofactor == 0);
            factors.push((r1.cofactor as i64, 2));
            Relation {
                x: (r1.x * r2.x) % self.n,
                cofactor: r2.cofactor / r1.cofactor,
                pp: r1.pp || r2.pp,
                factors,
            }
        }
    }

    pub fn add(&mut self, r: Relation, pq: Option<(u64, u64)>) {
        if r.cofactor == 1 {
            self.n_smooths += 1;
            self.complete.push(r);
        } else if r.cofactor < self.maxlarge {
            // Factor base elements have at most 24 bits
            self.n_partials += 1;
            if !self.combine_single(&r) {
                self.partial.insert(r.cofactor, r.pack());
            }
        } else {
            // Cofactor is above 32 bits: is it a double prime?
            let Some((p, q)) = pq
                else { return; };
            self.n_doubles += 1;
            if self.combine_double(&r, p, q) {
                // nothing to do
            } else {
                // No combination available.
                let key = if p < q { (p, q) } else { (q, p) };
                self.doubles.insert(key, PackedRelation::pack(r));
            }
            // Every 5000 double relations, scan the array.
            if self.n_doubles % 5000 == 0 {
                self.gc_doubles()
            }
        }
    }

    fn gc_doubles(&mut self) {
        let mut delete = vec![];
        for &(p, q) in self.doubles.keys() {
            if self.partial.contains_key(&p) || self.partial.contains_key(&q) {
                delete.push((p, q));
            }
        }
        let deleted = delete.len();
        if deleted == 0 {
            return;
        }
        for (p, q) in delete {
            let r = self.doubles.remove(&(p, q)).unwrap();
            let ok = self.combine_double(&r.unpack(), p, q);
            assert!(ok);
        }
        eprintln!("[RelationSet] Combined {} double-large relations", deleted);
        // FIXME: don't even attempt to find any relation cycle.
        //self.find_cycles();
    }

    /// Debugging method to count cycles in a set of pp-relations.
    /// It uses the union find algorithm to determine connected components.
    #[allow(dead_code)]
    fn find_cycles(&self) {
        // prime => (parent, depth)
        let mut forest = HashMap::<i32, (i32, u32)>::new();
        fn root(forest: &HashMap<i32, (i32, u32)>, n: i32) -> Option<(i32, u32)> {
            let mut node = n;
            while let Some(&(r, l)) = forest.get(&node) {
                if r == node {
                    return Some((r, l));
                }
                node = r;
            }
            None
        }
        // Insert all keys in the forest
        for &(p, q) in self.doubles.keys() {
            if p == q {
                panic!("impossible")
            }
            let (p, q) = (p as i32, q as i32);
            let (rp, rq) = (root(&forest, p), root(&forest, q));
            match (rp, rq) {
                (None, None) => {
                    // insert q with parent p
                    forest.insert(p, (p, 1));
                    forest.insert(q, (p, 0));
                }
                (Some((r1, _)), None) => {
                    forest.insert(q, (r1, 0));
                }
                (None, Some((r2, _))) => {
                    forest.insert(p, (r2, 0));
                }
                (Some((r1, l1)), Some((r2, l2))) => {
                    if r1 != r2 {
                        if l1 <= l2 {
                            forest.insert(r1, (r1, l1 + 1));
                            forest.insert(r2, (r1, 0));
                        } else {
                            forest.insert(r1, (r2, 0));
                            forest.insert(r2, (r2, l2 + 1));
                        }
                    }
                }
            }
        }
        // Count connected components
        let mut h0 = 0;
        for (k, (v, _)) in forest.iter() {
            if k == v {
                h0 += 1;
            }
        }
        // Check Euler characteristic
        let e0 = forest.len();
        let e1 = self.doubles.len();
        let h1: i64 = h0 as i64 - (e0 as i64 - e1 as i64);
        eprintln!(
            "[RelationSet] V={} E={} components={} cycles={}",
            e0, e1, h0, h1
        );
    }

    // Tries to combine a relation with an existing one.
    fn combine_single(&mut self, r: &Relation) -> bool {
        if let Some(r0) = self.partial.get(&r.cofactor) {
            let rr = self.combine(&r, &r0.unpack());
            if rr.factors.iter().all(|(_, exp)| exp % 2 == 0) {
                // FIXME: Poor choice of A's can lead to duplicate relations.
                eprintln!("FIXME: ignoring trivial relation");
                return false;
            }
            debug_assert!(
                rr.verify(&self.n),
                "INTERNAL ERROR: invalid combined relation\nr1={:?}\nr2={:?}\nr1*r2={:?}",
                r,
                r0.unpack(),
                rr
            );
            self.complete.push(rr);
            if r.pp || r0.pp {
                self.n_combined2 += 1;
            } else {
                self.n_combined += 1;
            }
            true
        } else {
            false
        }
    }

    fn combine_double(&mut self, r: &Relation, p: u64, q: u64) -> bool {
        if p == q {
            // Very unlikely: a perfect square
            let mut f = r.factors.clone();
            f.push((p as i64, 2));
            let rr = Relation {
                cofactor: 1,
                factors: f,
                ..r.clone()
            };
            self.complete.push(rr);
            self.n_combined2 += 1;
            true
        } else if self.partial.contains_key(&p) && self.partial.contains_key(&q) {
            // Ideal case, both primes already available.
            let rp = self.partial.get(&p).unwrap().unpack();
            let rq = self.partial.get(&q).unwrap().unpack();
            let r2 = self.combine(&self.combine(&r, &rp), &rq);
            assert_eq!(r2.cofactor, 1);
            self.complete.push(r2);
            self.n_combined2 += 1;
            true
        } else if self.partial.contains_key(&p) {
            let rp = self.partial.get(&p).unwrap();
            let mut rq = self.combine(&r, &rp.unpack());
            rq.pp = true;
            assert_eq!(rq.cofactor, q);
            self.n_combined12 += 1;
            self.partial.insert(q, rq.pack());
            true
        } else if self.partial.contains_key(&q) {
            let rq = self.partial.get(&q).unwrap();
            let mut rp = self.combine(&r, &rq.unpack());
            rp.pp = true;
            assert_eq!(rp.cofactor, p);
            self.n_combined12 += 1;
            self.partial.insert(p, rp.pack());
            true
        } else {
            false
        }
    }
}

pub fn relation_gap(rels: &[Relation]) -> usize {
    if rels.len() == 0 {
        return 1000; // infinity
    }
    let mut occs = HashMap::<i64, u64>::new();
    for r in rels {
        for (f, k) in r.factors.iter() {
            if k % 2 == 1 {
                let c = occs.get(&f).unwrap_or(&0);
                occs.insert(*f, c + 1);
            }
        }
    }
    if occs.len() > rels.len() {
        occs.len() - rels.len()
    } else {
        0
    }
}

pub fn final_step(n: &Uint, rels: &[Relation], verbose: bool) -> Option<(Uint, Uint)> {
    for r in rels {
        debug_assert!(r.verify(n));
    }
    // Collect occurrences
    let mut occs = HashMap::<i64, u64>::new();
    for r in rels {
        for (f, k) in r.factors.iter() {
            if k % 2 == 1 {
                let c = occs.get(&f).unwrap_or(&0);
                occs.insert(*f, c + 1);
            }
        }
    }
    if verbose {
        eprintln!("Input {} relations {} factors", rels.len(), occs.len());
    }
    // Sort factors by increasing occurrences
    let mut occs: Vec<(i64, u64)> = occs.into_iter().filter(|&(_, k)| k > 1).collect();
    occs.sort_by_key(|&(_, k)| k);
    let nfactors = occs.len();
    let mut idxs = HashMap::<i64, usize>::new();
    for (idx, (f, _)) in occs.into_iter().enumerate() {
        idxs.insert(f, idx);
    }
    // Build vectors
    // ridx[i] = j if rels[j] is the i-th vector in the matrix
    let mut filt_rels: Vec<&Relation> = vec![];
    let mut matrix = vec![];
    let size = idxs.len();
    let mut coeffs = 0;
    'skiprel: for r in rels.iter() {
        let mut v = vec![];
        for (f, k) in r.factors.iter() {
            if k % 2 == 0 {
                continue;
            }
            if let Some(&idx) = idxs.get(&f) {
                v.push(idx);
                coeffs += 1;
            } else {
                continue 'skiprel;
            }
        }
        filt_rels.push(r);
        matrix.push(v);
    }
    if verbose {
        eprintln!(
            "Filtered {} relations {} factors",
            filt_rels.len(),
            nfactors
        );
        eprintln!(
            "Build matrix {}x{} ({:.1} entries/col)",
            size,
            matrix.len(),
            coeffs as f64 / size as f64
        );
    }
    let k = if size > 5000 {
        // Block Lanczos
        let mat = matrix::SparseMat {
            k: size,
            cols: matrix,
        };
        matrix::kernel_lanczos(&mat, true)
    } else {
        let mut dense = vec![];
        for col in matrix {
            let mut v = BitVec::zeros(size);
            for idx in col {
                v.set(idx, true);
            }
            dense.push(v);
        }
        matrix::kernel_gauss(dense)
    };
    if verbose {
        eprintln!("Found kernel of dimension {}", k.len());
    }
    for eq in k {
        let mut rs = vec![];
        for i in eq.into_usizes().into_iter() {
            rs.push(filt_rels[i].clone());
        }
        if verbose {
            eprintln!("Combine {} relations...", rs.len());
        }
        let (a, b) = combine(n, &rs);
        if verbose {
            eprintln!("Same square mod N: {} {}", a, b);
        }
        if let Some((p, q)) = try_factor(n, a, b) {
            if verbose {
                eprintln!("Found factors!");
                println!("{}", p);
                println!("{}", q);
            }
            return Some((p, q));
        }
    }
    None
}

/// Combine relations into an identity a^2 = b^2
pub fn combine(n: &Uint, rels: &[Relation]) -> (Uint, Uint) {
    // Check that the product is a square
    let mut a = Uint::one();
    for r in rels {
        a = (a * r.x) % n;
    }
    // Collect exponents
    let mut exps = HashMap::<i64, u64>::new();
    for r in rels {
        for (p, k) in &r.factors {
            let e = exps.get(&p).unwrap_or(&0);
            exps.insert(*p, e + k);
        }
    }
    let mut b = Uint::one();
    for (p, k) in exps.into_iter() {
        assert_eq!(k % 2, 0);
        if p == -1 {
            continue;
        }
        b = (b * pow_mod(Uint::from(p as u64), Uint::from(k / 2), *n)) % n;
    }
    assert_eq!((a * a) % n, (b * b) % n);
    (a, b)
}

/// Using a^2 = b^2 mod n, try to factor n
pub fn try_factor(n: &Uint, a: Uint, b: Uint) -> Option<(Uint, Uint)> {
    if a == b || a + b == *n {
        // Trivial square relation
        return None;
    }
    let e = Integer::extended_gcd(&Int::from_bits(*n), &Int::from_bits(a + b));
    if e.gcd > Int::one() {
        let p = e.gcd.to_bits();
        let q = n / p;
        assert!(p * q == *n);
        assert!(p.bits() > 1 && q.bits() > 1);
        return Some((p, q));
    }
    let e = Integer::extended_gcd(&Int::from_bits(*n), &Int::from_bits(n + a - b));
    if e.gcd > Int::one() {
        let p = e.gcd.to_bits();
        let q = n / p;
        assert!(p * q == *n);
        assert!(p.bits() > 1 && q.bits() > 1);
        return Some((p, q));
    }
    None
}

// A packed version of Relation.
// Factors are encoded as ULEB128 with optional 1-byte exponent.
// Typical memory usage is about 2-3 bytes per factor instead of 16.

struct PackedRelation {
    x: U512,
    cofactor: u64,
    pp: bool,
    // Usually less than 64 bytes (200 bytes for combined relations).
    factors_blob: Vec<u8>,
}

impl PackedRelation {
    fn pack(r: Relation) -> PackedRelation {
        let mut ints = vec![];
        for &(p, k) in &r.factors {
            // encode each factor as a sequence of integers
            // -1 encodes to 0
            // 2 encodes to 1
            // p encodes to p
            // (p, k > 1) encodes to (2p, k)
            match (p, k) {
                (-1, k) if k % 2 == 0 => continue,
                (-1, k) if k % 2 == 1 => ints.push(0),
                (p, k) => {
                    assert!(p > 0 && p < (1 << 31));
                    let p = if p == 2 { 1 } else { p };
                    assert!(p % 2 == 1);
                    if k > 1 {
                        ints.push(2 * p as u32);
                        ints.push(k as u32);
                    } else {
                        ints.push(p as u32);
                    }
                }
            }
        }
        let mut factors_blob = vec![];
        for n in ints {
            // encode as leb128
            let length = std::cmp::max(1, (32 - u32::leading_zeros(n) + 6) / 7);
            for j in 0..length {
                let mut byte = (n >> (7 * (length - 1 - j))) & 0x7f;
                if j > 0 {
                    byte |= 0x80;
                }
                factors_blob.push(byte as u8);
            }
        }
        PackedRelation {
            x: U512::cast_from(r.x),
            cofactor: r.cofactor,
            pp: r.pp,
            factors_blob,
        }
    }

    fn unpack(&self) -> Relation {
        // Decode ULEB128
        let mut ints = vec![];
        let mut n = 0;
        for (i, &byte) in self.factors_blob.iter().enumerate() {
            if byte < 0x80 {
                if i > 0 {
                    ints.push(n)
                }
                n = byte as u32;
            } else {
                n = (n << 7) | (byte as u32 & 0x7f);
            }
        }
        ints.push(n);
        let mut factors = vec![];
        let mut idx = 0;
        while idx < ints.len() {
            let n = ints[idx];
            if n == 0 {
                factors.push((-1i64, 1u64));
                idx += 1;
            } else if n % 2 == 1 {
                // factor without exponent
                let p = if n == 1 { 2 } else { n } as i64;
                factors.push((p, 1));
                idx += 1;
            } else {
                let p = if n == 2 { 2 } else { n >> 1 };
                let k = ints[idx + 1] as u64;
                factors.push((p as i64, k));
                idx += 2;
            }
        }
        Relation {
            x: Uint::cast_from(self.x),
            cofactor: self.cofactor,
            pp: self.pp,
            factors,
        }
    }
}

#[test]
fn test_pack_relation() {
    use std::str::FromStr;

    let r = Relation {
        x: Uint::from_str("135487168713871387841578923567").unwrap(),
        cofactor: 7915738421,
        pp: true,
        factors: vec![
            (-1, 1),
            (2, 17),
            (3, 5),
            (5, 1),
            (9109, 1),
            (9173, 2),
            (9241, 3),
            (9349, 1),
            (19349, 1),
            (39349, 1),
            (289349, 1),
            (3879645, 1),
        ],
    };
    assert_eq!(PackedRelation::pack(r.clone()).unpack(), r);
}
