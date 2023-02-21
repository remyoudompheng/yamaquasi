// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Relations describe an equation:
//! x^2 = product(pi^ki) mod n
//!
//! where pi = -1 or a prime in the factor base

use std::cmp::{max, min};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::default::Default;

use bitvec_simd::BitVec;
use bnum::cast::CastFrom;
use num_integer::Integer;
use num_traits::One;

use crate::arith::{pow_mod, Num, U512};
use crate::arith_montgomery::ZmodN;
use crate::matrix;
use crate::{Int, Uint, Verbosity};

/// Number of extra relations that must be collected for factoring.
/// Block Lanczos cannot find more kernel vectors than its block size.
///
/// We will find random elements of the order 2 subgroup of Z/nZ
/// hoping that they will generate a large rank subgroup.
pub const MIN_KERNEL_SIZE: usize = 48;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relation {
    pub x: Uint,
    pub cofactor: u64,
    pub cyclelen: u64,
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
                assert!(p > 0 || p == 0 && self.factors.len() == 1);
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
/// intersect the (huge) connected component of p-relations. Similarly, there
/// is negligible benefit (usually less than 1%) in trying to build minimal
/// cycles: using a greedy approach to build cycles is acceptable.
///
/// A cycle of length 1 is a complete relation.
/// A cycle of length 2 is a combination of 2 p-relations.
/// A cycle of length >= 3 involves at least a pp-relation.
#[derive(Default)]
pub struct RelationSet {
    pub n: Uint,
    pub maxlarge: u64,
    pub cycles: Vec<Relation>,
    // p => relation with cofactor p
    partial: HashMap<u64, PackedRelation>,
    // p => relation with cofactor pq (p < q)
    // No key is common with partial map
    doubles: BTreeMap<(u32, u32), PackedRelation>,
    // Set of keys (q > p) for pp-relations for easy reverse lookup
    doubles_rev: BTreeSet<(u32, u32)>,
    pub n_partials: usize,
    pub n_doubles: usize,
    pub n_combined12: usize,
    pub n_cycles: [usize; 8],
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
        self.cycles
    }

    pub fn truncate(&mut self, l: usize) {
        self.cycles.truncate(l)
    }

    pub fn len(&self) -> usize {
        self.cycles.len()
    }

    pub fn gap(&self) -> usize {
        relation_gap(&self.cycles)
    }

    pub fn log_progress<S: AsRef<str>>(&self, prefix: S) {
        eprintln!(
            "{} found {} smooths (p={} p12={} pp={} cycles={:?})",
            prefix.as_ref(),
            self.len(),
            self.n_partials,
            self.n_combined12,
            self.n_doubles,
            self.n_cycles,
        )
    }

    /// Combine 2 relations sharing a cofactor.
    pub fn combine(&self, r1: &Relation, r2: &Relation) -> Relation {
        // Combine factors
        let mut factors = r1.factors.clone();
        'iter2: for f2 @ &(p, k) in &r2.factors {
            for f in factors.iter_mut() {
                if f.0 == p {
                    f.1 += k;
                    continue 'iter2;
                }
            }
            factors.push(f2.clone());
        }
        // Combine cofactors
        if r1.cofactor % r2.cofactor == 0 {
            factors.push((r2.cofactor as i64, 2));
            Relation {
                x: (r1.x * r2.x) % self.n,
                cofactor: r1.cofactor / r2.cofactor,
                cyclelen: r1.cyclelen + r2.cyclelen,
                factors,
            }
        } else {
            assert!(r2.cofactor % r1.cofactor == 0);
            factors.push((r1.cofactor as i64, 2));
            Relation {
                x: (r1.x * r2.x) % self.n,
                cofactor: r2.cofactor / r1.cofactor,
                cyclelen: r1.cyclelen + r2.cyclelen,
                factors,
            }
        }
    }

    pub fn add(&mut self, r: Relation, pq: Option<(u64, u64)>) {
        debug_assert!(&r.x < &self.n);
        if r.cofactor == 1 {
            self.add_cycle(r);
        } else if r.cofactor < self.maxlarge {
            // Factor base elements have at most 24 bits
            self.n_partials += 1;
            if self.combine_single(&r) {
                return;
            }
            let p = r.cofactor;
            self.partial.insert(r.cofactor, r.pack());
            // Combine with existing pp-relations
            assert!(p >> 32 == 0);
            self.walk_doubles(p as u32);
        } else {
            // Cofactor is above 32 bits: is it a double prime?
            let Some((p, q)) = pq
                else { return; };
            assert!(p >> 32 == 0 && q >> 32 == 0);
            self.n_doubles += 1;
            // Hold pp-relations for a while
            if self.combine_double(&r, p, q) {
                // nothing to do
            } else {
                // No combination available.
                let (p, q) = (p as u32, q as u32);
                let key = if p < q { (p, q) } else { (q, p) };
                self.doubles.insert(key, PackedRelation::pack(r));
                self.doubles_rev.insert((key.1, key.0));
            }
        }
    }

    pub fn add_cycle(&mut self, r: Relation) {
        assert_eq!(r.cofactor, 1);
        self.n_cycles[min(self.n_cycles.len(), r.cyclelen as usize) - 1] += 1;
        self.cycles.push(r);
    }

    fn walk_doubles(&mut self, root: u32) {
        let pqs: Vec<(u32, u32)> = self
            .doubles
            .range((root, 0)..(root + 1, 0))
            .map(|(&k, _)| k)
            .collect();
        let qps: Vec<(u32, u32)> = self
            .doubles_rev
            .range((root, 0)..(root + 1, 0))
            .map(|&k| k)
            .collect();
        for key @ &(p, q) in &pqs {
            let Some(r) = self.doubles.remove(key) else { continue };
            self.doubles_rev.remove(&(q, p));
            let ok = self.combine_double(&r.unpack(), p as u64, q as u64);
            assert!(ok);
        }
        for key @ &(q, p) in &qps {
            let Some(r) = self.doubles.remove(&(p, q)) else { continue };
            self.doubles_rev.remove(key);
            let ok = self.combine_double(&r.unpack(), p as u64, q as u64);
            assert!(ok);
        }
        for (p, q) in pqs {
            assert_eq!(p, root);
            self.walk_doubles(q);
        }
        for (q, p) in qps {
            assert_eq!(q, root);
            self.walk_doubles(p);
        }
    }

    #[allow(dead_code)]
    fn gc_doubles(&mut self) {
        let mut delete = vec![];
        for &(p, q) in self.doubles.keys() {
            if self.partial.contains_key(&(p as u64)) || self.partial.contains_key(&(q as u64)) {
                delete.push((p, q));
            }
        }
        let deleted = delete.len();
        if deleted == 0 {
            return;
        }
        for (p, q) in delete {
            let Some(r) = self.doubles.remove(&(p, q)) else { continue };
            self.doubles_rev.remove(&(q, p));
            let ok = self.combine_double(&r.unpack(), p as u64, q as u64);
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
        if let Some(r0_h) = self.partial.get(&r.cofactor) {
            let r0 = r0_h.unpack();
            let rr = self.combine(r, &r0);
            if rr.factors.iter().all(|(_, exp)| exp % 2 == 0) {
                // FIXME: Poor choice of A's can lead to duplicate relations.
                if crate::DEBUG {
                    eprintln!("FIXME: ignoring trivial relation");
                }
                return false;
            }
            debug_assert!(
                rr.verify(&self.n),
                "INTERNAL ERROR: invalid combined relation\nr1={:?}\nr2={:?}\nr1*r2={:?}",
                r,
                r0,
                rr
            );
            self.add_cycle(rr);
            if r.cyclelen < r0.cyclelen {
                self.partial.insert(r.cofactor, r.clone().pack());
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
            self.add_cycle(rr);
            true
        } else if self.partial.contains_key(&p) && self.partial.contains_key(&q) {
            // Ideal case, both primes already available.
            //   1    3 relations are involved.
            //  / \
            // p---q
            let rp = self.partial.get(&p).unwrap().unpack();
            let rq = self.partial.get(&q).unwrap().unpack();
            let r2 = self.combine(&self.combine(r, &rp), &rq);
            self.add_cycle(r2);
            // Minimize spanning tree size by replacing paths.
            if rp.cyclelen + r.cyclelen < rq.cyclelen {
                let rpq = self.combine(r, &rp);
                assert_eq!(rpq.cofactor, q);
                self.partial.insert(q, rpq.pack());
            } else if rq.cyclelen + r.cyclelen < rp.cyclelen {
                let rqp = self.combine(r, &rq);
                assert_eq!(rqp.cofactor, p);
                self.partial.insert(p, rqp.pack());
            }
            true
        } else if self.partial.contains_key(&p) {
            let rp = self.partial.get(&p).unwrap();
            let rq = self.combine(r, &rp.unpack());
            assert_eq!(rq.cofactor, q);
            self.n_combined12 += 1;
            self.partial.insert(q, rq.pack());
            self.walk_doubles(q as u32);
            true
        } else if self.partial.contains_key(&q) {
            let rq = self.partial.get(&q).unwrap();
            let rp = self.combine(r, &rq.unpack());
            assert_eq!(rp.cofactor, p);
            self.n_combined12 += 1;
            self.partial.insert(p, rp.pack());
            self.walk_doubles(p as u32);
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
                let c = occs.get(f).unwrap_or(&0);
                occs.insert(*f, c + 1);
            }
        }
    }
    // We require additional relations compared to the number of factors.
    // This is because relations may accidentally be trivial
    // (x^2=y^2 where n divides x-y or x+y).
    if occs.len() + MIN_KERNEL_SIZE > rels.len() {
        occs.len() + MIN_KERNEL_SIZE - rels.len()
    } else {
        0
    }
}

/// Finds non trivial square roots of 1 modulo n and returns
/// a list of non-trivial divisors of n.
///
/// Note that n may be different from the original sieve modulus
/// which includes the multiplier.
pub fn final_step(n: &Uint, rels: &[Relation], verbose: Verbosity) -> Vec<Uint> {
    for r in rels {
        debug_assert!(r.verify(n));
    }
    // Collect occurrences
    let mut occs = HashMap::<i64, u64>::new();
    for r in rels {
        for (f, k) in r.factors.iter() {
            let c = occs.get(f);
            if k % 2 == 1 {
                occs.insert(*f, c.unwrap_or(&0) + 1);
            } else if c.is_none() {
                // Make sure to register all factors.
                // They will be required when computing products below.
                occs.insert(*f, 0);
            }
        }
    }
    if verbose >= Verbosity::Info {
        eprintln!("Input {} relations {} factors", rels.len(), occs.len());
    }
    // Sort factors by decreasing occurrences
    // Gauss elimination is much more efficient if it starts by eliminating
    // the (few) densest rows: the remaining rows will remain relatively sparse
    // during the rest of the elimination.
    let mut occs: Vec<(i64, u64)> = occs.into_iter().collect();
    occs.sort_by_key(|&(_, k)| -(k as i64));
    // We keep only factors with > 1 occurrences for the matrix.
    let nfactors = occs.iter().position(|&(_, k)| k <= 1).unwrap_or(occs.len());
    // Map primes to their index (even filtered ones).
    let mut idxs = HashMap::<i64, usize>::new();
    for (idx, &(f, _)) in occs.iter().enumerate() {
        idxs.insert(f, idx);
    }
    // Build vectors
    // ridx[i] = j if rels[j] is the i-th vector in the matrix
    let mut filt_rels: Vec<Relation> = vec![];
    let mut matrix = vec![];
    let size = nfactors;
    let mut coeffs = 0;
    'skiprel: for r in rels.iter() {
        let mut v = vec![];
        for (f, k) in r.factors.iter() {
            if k % 2 == 0 {
                continue;
            }
            let &idx = idxs.get(f).unwrap();
            if idx < nfactors {
                v.push(idx);
                coeffs += 1;
            } else {
                continue 'skiprel;
            }
        }
        // Make sure relation element is smaller than N.
        let mut r = r.clone();
        if &r.x > n {
            r.x %= n;
        }
        filt_rels.push(r);
        matrix.push(v);
    }
    if verbose >= Verbosity::Info {
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
    let start = std::time::Instant::now();
    let k = if size > 5000 {
        // Block Lanczos
        let mat = matrix::SparseMat {
            k: size,
            cols: matrix,
        };
        matrix::kernel_lanczos(&mat, verbose)
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
    if verbose >= Verbosity::Info {
        let dt = start.elapsed();
        eprintln!(
            "Found kernel of dimension {} in {:.3}s",
            k.len(),
            dt.as_secs_f64()
        );
    }
    let mut divisors = vec![];
    let mut nontrivial = 0;
    let zn = ZmodN::new(*n);
    for eq in k {
        // Collect relations for this vector.
        let mut xs = vec![];
        let mut exps = vec![0u64; occs.len()];
        for i in eq.into_usizes().into_iter() {
            xs.push(filt_rels[i].x);
            for (f, k) in &filt_rels[i].factors {
                let idx = idxs.get(f).unwrap();
                exps[*idx] += k;
            }
        }
        let factors: Vec<(i64, u64)> = exps
            .into_iter()
            .enumerate()
            .filter(|&(_, exp)| exp > 0)
            .map(|(idx, exp)| (occs[idx].0, exp))
            .collect();
        if verbose >= Verbosity::Debug {
            eprintln!("Combine {} relations...", xs.len());
        }
        let (a, b) = combine(&zn, &xs, &factors);
        assert_eq!((a * a) % n, (b * b) % n);
        if verbose >= Verbosity::Debug {
            eprintln!("Same square mod N: {} {}", a, b);
        }
        let Some((p, q)) = try_factor(n, a, b) else { continue };
        divisors.push(p);
        divisors.push(q);
        nontrivial += 1;
        // Stop combining relations in the common case where there
        // are only 2 prime factors. If there are more factors, it means they
        // were not found before, meaning that the number is quite large
        // so this additional cost is acceptable, because we will avoid redundant
        // computations.
        if crate::pseudoprime(p) && crate::pseudoprime(q) {
            break;
        }
    }
    divisors.sort_unstable();
    divisors.dedup();
    if verbose >= Verbosity::Info {
        eprintln!(
            "{} divisors from {} successful factorizations",
            divisors.len(),
            nontrivial
        );
    }
    divisors
}

/// Combine relations into an identity a^2 = b^2
/// Instead of handling an array of relations,
/// we provide xs such that a = product(xs)
/// and [(p, k)] such that b^2 = product(p^k)
pub fn combine(zn: &ZmodN, xs: &[Uint], factors: &[(i64, u64)]) -> (Uint, Uint) {
    // Avoid too many (x % n) operations especially when factors are small.
    // All factors are less than 32 bits.

    // Product of x (they are less than 512 bits wide).
    let mut a = zn.one();
    for &x in xs {
        a = zn.mul(&a, &zn.from_int(x));
    }
    // Product of factors: they are smaller than 32 bits.
    // Accumulate product in a u64 before performing long multiplications.
    let mut b = zn.one();
    let mut chunk = 1_u64;
    let maxchunk = if zn.n.bits() <= 64 {
        min(1 << 32, zn.n.low_u64())
    } else {
        1 << 32
    };
    for &(p, k) in factors {
        if p == -1 {
            continue;
        }
        assert_eq!(k % 2, 0);
        for _ in 0..k / 2 {
            let c = chunk * p as u64;
            if c >= maxchunk {
                b = zn.mul(&b, &zn.from_int(chunk.into()));
                chunk = p as u64;
            } else {
                chunk = c;
            }
        }
    }
    b = zn.mul(&b, &zn.from_int(chunk.into()));
    (zn.to_int(a), zn.to_int(b))
}

/// Using a^2 = b^2 mod n, try to factor n
pub fn try_factor(n: &Uint, a: Uint, b: Uint) -> Option<(Uint, Uint)> {
    // Note that when a = ±b we can still obtain a factor
    // if a and b actually share a factor with n.
    if a + b != *n {
        let gcd = Integer::gcd(&Int::from_bits(*n), &Int::from_bits(a + b));
        if gcd > Int::one() {
            let p = gcd.to_bits();
            let q = n / p;
            assert!(p * q == *n);
            assert!(p.bits() > 1 && q.bits() > 1);
            return Some((p, q));
        }
    }
    if a != b {
        let gcd = Integer::gcd(&Int::from_bits(*n), &Int::from_bits(n + a - b));
        if gcd > Int::one() {
            let p = gcd.to_bits();
            let q = n / p;
            assert!(p * q == *n);
            assert!(p.bits() > 1 && q.bits() > 1);
            return Some((p, q));
        }
    }
    None
}

// A packed version of Relation.
// Integers are encoded as ULEB128 with optional 1-byte exponent.
// Typical memory usage is about 2-3 bytes per factor instead of 16.

struct PackedRelation {
    // Usually less than 128 bytes (200 bytes for combined relations).
    blob: Box<[u8]>,
}

impl PackedRelation {
    fn pack(r: Relation) -> PackedRelation {
        // Capacity assumes that no more than 4 factors have exponent > 1.
        let mut ints = Vec::with_capacity(8 + 2 + r.factors.len() + 4);
        // Append 8 words from r.x (assumed to be less than 512 bits)
        // and cofactor, cyclelen.
        ints.extend_from_slice(&r.x.digits()[..8]);
        ints.push(r.cofactor);
        ints.push(r.cyclelen);
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
                    assert!(p > 0 && p < (1 << 31) && k > 0);
                    let p = if p == 2 { 1 } else { p };
                    assert!(p % 2 == 1);
                    if k > 1 {
                        ints.push(2 * p as u64);
                        ints.push(k);
                    } else {
                        ints.push(p as u64);
                    }
                }
            }
        }
        let mut blob = Vec::with_capacity(max(64, ints.len() * 2));
        for n in ints {
            // encode as leb128
            let length = max(1, (64 - u64::leading_zeros(n) + 6) / 7);
            for j in 0..length {
                let mut byte = (n >> (7 * (length - 1 - j))) & 0x7f;
                if j > 0 {
                    byte |= 0x80;
                }
                blob.push(byte as u8);
            }
        }
        PackedRelation {
            blob: blob.into_boxed_slice(),
        }
    }

    fn unpack(&self) -> Relation {
        // Decode ULEB128
        let mut ints = Vec::with_capacity(max(32, self.blob.len() / 2));
        let mut n = 0;
        for (i, &byte) in self.blob.iter().enumerate() {
            if byte < 0x80 {
                if i > 0 {
                    ints.push(n)
                }
                n = byte as u64;
            } else {
                n = (n << 7) | (byte as u64 & 0x7f);
            }
        }
        ints.push(n);
        let [x0,x1,x2,x3,x4,x5,x6,x7,cofactor,clen] = ints[..10]
            else { unreachable!("corrupted relation data") };
        let x = U512::from_digits([x0, x1, x2, x3, x4, x5, x6, x7]);
        let mut factors = vec![];
        let mut idx = 10;
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
                let k = ints[idx + 1];
                factors.push((p as i64, k));
                idx += 2;
            }
        }
        Relation {
            x: Uint::cast_from(x),
            cofactor: cofactor,
            cyclelen: clen,
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
        cyclelen: 4,
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
