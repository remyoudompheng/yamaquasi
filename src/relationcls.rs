// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! CRelations in the class group of a quadratic imaginary field.
//! They are represented as sequences (p,k) where k is a possibly
//! negative integer, and (p,1) (p,-1) represent a conjugate
//! pair of ideals of norm p, in normalized convention.
//!
//! In class group computations, we need to perform linear algebra
//! over Z, so we want to keep the most precise set of relations
//! to perform matrix filtering.
//! We still keep track of the largest connected component of the
//! large prime graph, but emit relations instead of combining them.

use std::cmp::min;
use std::collections::{BTreeMap, BTreeSet};
use std::default::Default;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use crate::Int;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CRelation {
    // All factors belong to the original factor base.
    pub factors: Vec<(u32, i32)>,
    // Large primes are stored separately and may be larger.
    pub large1: Option<(u32, i32)>,
    pub large2: Option<(u32, i32)>,
}

/// Combine 2 relations sharing a large prime.
pub fn combine(r1: &CRelation, r2: &CRelation) -> CRelation {
    // Which large prime is common?
    let (_p, e1, e2, q1, q2) = match (r1.large1, r1.large2, r2.large1, r2.large2) {
        (Some((p1, e1)), q1, Some((p2, e2)), q2) if p1 == p2 => (p1, e1, e2, q1, q2),
        (Some((p1, e1)), q1, q2, Some((p2, e2))) if p1 == p2 => (p1, e1, e2, q1, q2),
        (q1, Some((p1, e1)), Some((p2, e2)), q2) if p1 == p2 => (p1, e1, e2, q1, q2),
        (q1, Some((p1, e1)), q2, Some((p2, e2))) if p1 == p2 => (p1, e1, e2, q1, q2),
        _ => {
            panic!("impossible: relations cannot be combined")
        }
    };
    let (q1, q2) = if q1.is_none() { (q2, q1) } else { (q1, q2) };
    // Eliminate like Gauss elimination e2 * factors1 - e1 * factors2
    // The exponents are always ±1 and ±1 in practice, because it's
    // highly unlikely that a binary form value is divisible by a large square.
    let mut factors: Vec<(u32, i32)> = vec![];
    'iter1: for &(p, k) in &r1.factors {
        for f in factors.iter_mut() {
            if f.0 == p {
                f.1 += e2 * k;
                continue 'iter1;
            }
        }
        factors.push((p, e2 * k));
    }
    'iter2: for &(p, k) in &r2.factors {
        for f in factors.iter_mut() {
            if f.0 == p {
                f.1 -= e1 * k;
                continue 'iter2;
            }
        }
        factors.push((p, -e1 * k));
    }
    // Also multiply large primes
    let q1 = q1.map(|(p, k)| (p, e2 * k));
    let q2 = q2.map(|(p, k)| (p, -e1 * k));
    CRelation {
        factors,
        large1: q1,
        large2: q2,
    }
}

#[derive(Default)]
pub struct CRelationSet {
    pub d: Int,
    // Target number of relations
    pub target: usize,
    pub maxlarge: u32,
    pub emitted: Vec<CRelation>,
    // A map from large prime to distance
    // It is expected to be a spanning tree.
    paths: BTreeMap<u32, Vec<u32>>,
    // p,q => relation with cofactor pq (1 <= p < q)
    doubles: BTreeMap<(u32, u32), CRelation>,
    // Set of keys (q > p) for easy reverse lookup
    doubles_rev: BTreeSet<(u32, u32)>,
    pub n_partials: usize,
    pub n_doubles: usize,
    // Number of spanning tree vertices from pp-relations
    pub n_combined12: usize,
    pub n_cycles: [usize; 8],
    pub print_cycles: bool,
    // Output
    file: Option<fs::File>,
}

impl CRelationSet {
    pub fn new(d: Int, target: usize, maxlarge: u32, out: PathBuf) -> Self {
        let mut set = CRelationSet {
            d,
            target,
            maxlarge,
            print_cycles: true,
            file: fs::File::create(out).ok(), // FIXME: errors
            ..Default::default()
        };
        set.paths.insert(1, vec![1]);
        set
    }

    pub fn len(&self) -> usize {
        self.n_cycles.iter().sum()
    }

    pub fn done(&self) -> bool {
        self.len() > self.target
    }

    pub fn log_progress<S: AsRef<str>>(&self, prefix: S) {
        // Similar to [relations::RelationSet]
        let prefix = prefix.as_ref();
        // Compute K >= 1 such that finding K times more relations would end the sieve.
        let n0 = self.n_cycles[0] as f64;
        if n0 == 0.0 {
            return; // Cannot say anything interesting.
        }
        if self.n_partials == 0 {
            // No large primes, easy estimate.
            let progress = n0 / self.target as f64 * 100.0;
            eprintln!(
                "{prefix} found {} relations (~{progress:.1}% done)",
                self.len()
            );
            return;
        }
        let mut n1 = self.n_cycles[1] as f64;
        if n1 <= 2.0 {
            n1 /= 2.0; // maybe we were lucky, use a conservative estimate
        }
        if self.n_doubles == 0 {
            // Single large primes: solve K such that n0 K + n1 K^2 == target
            let k = if n1 == 0.0 {
                // Cannot estimate, assume that n0 = target/2 is the target.
                self.target as f64 / (2.0 * n0)
            } else {
                let mut kmin = 1.0;
                let mut kmax = self.target as f64 / n0;
                while kmax / kmin > 1.001 {
                    let k = (kmin + kmax) / 2.0;
                    let est = n0 * k + n1 * k * k;
                    if est > self.target as f64 {
                        kmax = k;
                    } else {
                        kmin = k;
                    }
                }
                (kmin + kmax) / 2.0
            };
            let progress = 100.0 / k;
            eprintln!(
                "{prefix} found {} relations (~{progress:.1}% done, p={} cycles={:?})",
                self.len(),
                self.n_partials,
                &self.n_cycles[..2],
            );
            return;
        }
        // Double large primes: solve K such that n0 K + n1 K^2 + n2 K^3.5 = target
        let mut n2 = self.n_cycles[2..].iter().sum::<usize>() as f64;
        if n2 <= 2.0 {
            n2 /= 2.0; // maybe we were lucky, use a conservative estimate
        }
        let k = if n2 == 0.0 {
            // Cannot estimate, assume that n0 = target/4 is the target.
            self.target as f64 / (4.0 * n0)
        } else {
            let mut kmin = 1.0;
            let mut kmax = self.target as f64 / n0;
            while kmax / kmin > 1.001 {
                let k = (kmin + kmax) / 2.0;
                let est = n0 * k + n1 * k * k + n2 * k.powf(3.5);
                if est > self.target as f64 {
                    kmax = k;
                } else {
                    kmin = k;
                }
            }
            (kmin + kmax) / 2.0
        };
        let progress = 100.0 / k;
        eprintln!(
            "{prefix} found {} relations (~{progress:.1}% done, p={} p12={} pp={} cycles={:?})",
            self.len(),
            self.n_partials,
            self.n_combined12,
            self.n_doubles,
            self.n_cycles,
        )
    }

    pub fn add(&mut self, r: CRelation) {
        match (r.large1, r.large2) {
            (None, None) => self.emit(r, 1),
            (Some((p, _)), None) if p < self.maxlarge => {
                // Factor base elements have at most 24 bits
                self.n_partials += 1;
                self.add_path(1, p, r);
            }
            (Some((p, _)), Some((q, _))) => {
                assert!(p != q);
                self.n_doubles += 1;
                self.add_path(p, q, r);
            }
            _ => return,
        }
    }

    pub fn add_path(&mut self, p: u32, q: u32, r: CRelation) {
        // Adding path can either:
        // - enlarge the spanning tree by 1 edge
        // - create a cycle
        // - add a new orphan edge
        let (p, q) = if p < q { (p, q) } else { (q, p) };
        let hasp = self.paths.contains_key(&p);
        let hasq = self.paths.contains_key(&q);
        if hasp && hasq {
            // Create a cycle
            let vp = self.paths.get(&p).unwrap().clone();
            let vq = self.paths.get(&q).unwrap().clone();
            self.emit_path(&vp);
            self.emit_path(&vq);
            self.emit(r, vp.len() + vq.len() - 1);
        } else {
            // Update spanning tree
            self.doubles.insert((p, q), r);
            self.doubles_rev.insert((q, p));
            if hasp {
                self.update_tree(p, q);
            }
            if hasq {
                self.update_tree(q, p);
            }
        }
    }

    pub fn update_tree(&mut self, p: u32, q: u32) {
        // q is the new vertex
        if !self.paths.contains_key(&q) {
            let mut v = self.paths.get(&p).unwrap().clone();
            v.push(q);
            if v.len() > 2 {
                self.n_combined12 += 1;
            }
            self.paths.insert(q, v);
            // Recurse on q2 > q
            let qgt: Vec<u32> = self
                .doubles
                .range((q, 0)..(q + 1, 0))
                .map(|(&(_, q2), _)| q2)
                .collect();
            for q2 in qgt {
                self.update_tree(q, q2);
            }
            // Recurse on q2 < q
            let qlt: Vec<u32> = self
                .doubles_rev
                .range((q, 0)..(q + 1, 0))
                .map(|&(_, q2)| q2)
                .collect();
            for q2 in qlt {
                self.update_tree(q, q2);
            }
        }
    }

    pub fn emit_path(&mut self, path: &[u32]) {
        for i in 1..path.len() {
            let p = path[i - 1];
            let q = path[i];
            let (p, q) = if p < q { (p, q) } else { (q, p) };
            if let Some(r) = self.doubles.remove(&(p, q)) {
                self.emit(r, 0);
            }
        }
    }

    pub fn emit(&mut self, r: CRelation, clen: usize) {
        if self.print_cycles {
            // Display factors
            let mut line = vec![];
            for &(p, e) in &r.factors {
                let (rp, re) = if e > 0 {
                    (p as i64, e)
                } else {
                    (-(p as i64), -e)
                };
                for _ in 0..re {
                    if line.len() > 0 {
                        line.push(b' ');
                    }
                    write!(&mut line, "{rp}").unwrap();
                }
            }
            if let Some((p, e)) = r.large1 {
                for _ in 0..e.unsigned_abs() {
                    line.push(b' ');
                    if e < 0 {
                        line.push(b'-');
                    }
                    write!(&mut line, "{p}").unwrap();
                }
            }
            if let Some((p, e)) = r.large2 {
                for _ in 0..e.unsigned_abs() {
                    line.push(b' ');
                    if e < 0 {
                        line.push(b'-');
                    }
                    write!(&mut line, "{p}").unwrap();
                }
            }
            line.push(b'\n');
            // FIXME: handle errors?
            self.file
                .as_mut()
                .expect("Output file not open")
                .write(&line)
                .unwrap();
        }
        if clen > 0 {
            self.n_cycles[min(self.n_cycles.len(), clen) - 1] += 1;
        }
        self.emitted.push(r);
    }
}
