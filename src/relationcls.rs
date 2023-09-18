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

use std::cmp::{max, min};
use std::collections::{BTreeMap, BTreeSet};
use std::default::Default;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use bnum::cast::CastFrom;

use crate::matrixint;
use crate::Int;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CRelation {
    // All factors belong to the original factor base.
    pub factors: Vec<(u32, i32)>,
    // Large primes are stored separately and may be larger.
    pub large1: Option<(u32, i32)>,
    pub large2: Option<(u32, i32)>,
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
    pub fn new(d: Int, target: usize, maxlarge: u32, out: Option<PathBuf>) -> Self {
        let mut set = CRelationSet {
            d,
            target,
            maxlarge,
            print_cycles: out.is_some(),
            // FIXME: errors
            file: out.and_then(|out| fs::File::create(out).ok()),
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

    pub fn to_vec(self) -> Vec<CRelation> {
        self.emitted
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
            let mut push = |p: u32, e: i32| {
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
            };
            for &(p, e) in &r.factors {
                push(p, e);
            }
            if let Some((p, e)) = r.large1 {
                push(p, e);
            }
            if let Some((p, e)) = r.large2 {
                push(p, e);
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

pub fn group_structure(rels: Vec<CRelation>, hest: (f64, f64), outdir: Option<PathBuf>) {
    let t0 = Instant::now();
    let mut r = RelFilterSparse::new(rels);
    while let Some(_) = r.pivot_one() {
        if r.weight.len() % 16 == 0 {
            let nrows = r.rows.iter().filter(|r| r.len() > 0).count();
            eprintln!(
                "{} columns {} rows {} coefs",
                r.weight.len(),
                nrows,
                r.nonzero.len()
            );
        }
    }
    if let Some(outdir) = outdir.as_ref() {
        r.write_files(outdir);
    }
    let mut r = matrixint::SmithNormalForm::new(&r.rows, r.removed, hest.0, hest.1);
    eprintln!("{} columns {} rows", r.gens.len(), r.rows.len());
    eprintln!("Class number is {}", r.h);
    if let Some(outdir) = outdir.as_ref() {
        let mut w = fs::File::create(outdir.join("classnumber")).unwrap();
        writeln!(w, "{}", r.h).unwrap();
    }
    r.reduce();
    // Removed relations, in reverse order
    if let Some(outdir) = outdir.as_ref() {
        write_relations(&r, outdir);
        write_group_structure(&r, outdir);
        let qual = if r.gens.len() == 1 { "Full" } else { "Partial" };
        eprintln!(
            "{} group structure computed in {:.3}s",
            qual,
            t0.elapsed().as_secs_f64()
        );
    }
}

/// A structure to filter relations when they are sparse.
/// It implements a simple form of structured Gauss elimination.
struct RelFilterSparse {
    // Array of relations represented by sparse, sorted
    // vectors of (prime, exponent).
    rows: Vec<Vec<(u32, i32)>>,
    // How many times each given prime appears.
    weight: BTreeMap<u32, u32>,
    // Set of ±1 coefficients
    ones: BTreeSet<(u32, u32)>,
    // Set of nonzero indices (prime, row index)
    nonzero: BTreeSet<(u32, u32)>,
    // An array of removed relations, arranged as a triangular matrix.
    // Each relation involves primes appearing afterwards.
    removed: Vec<(u32, Vec<(u32, i32)>)>,
    // Min weight
    wmin: usize,
    dense: bool,
}

impl RelFilterSparse {
    fn new(rels: Vec<CRelation>) -> Self {
        let mut rows = vec![];
        for r in rels {
            let mut row: Vec<_> = r.factors.clone();
            if let Some((p, e)) = r.large1 {
                row.push((p, e));
            }
            if let Some((p, e)) = r.large2 {
                row.push((p, e));
            }
            row.sort();
            //eprintln!("got {row:?}");
            rows.push(row);
        }
        let mut weight = BTreeMap::new();
        let mut ones = BTreeSet::new();
        let mut nonzero = BTreeSet::new();
        for i in 0..rows.len() {
            for &(p, e) in &rows[i] {
                let w: &mut u32 = weight.entry(p).or_default();
                *w += 1;
                nonzero.insert((p, i as u32));
                if e.unsigned_abs() == 1 {
                    ones.insert((p, i as u32));
                }
            }
        }
        Self {
            rows,
            weight,
            ones,
            nonzero,
            removed: vec![],
            wmin: 0,
            dense: false,
        }
    }

    fn write_files(&self, outdir: &PathBuf) {
        let mut buf = vec![];
        for row in &self.rows {
            if row.len() == 0 {
                continue;
            }
            let mut first = true;
            for &(p, e) in row {
                if e == 0 {
                    continue;
                }
                if !first {
                    buf.push(b' ');
                }
                first = false;
                write!(&mut buf, "{p}^{e}").unwrap();
            }
            buf.push(b'\n');
        }
        let mut w = fs::File::create(outdir.join("relations.filtered")).unwrap();
        w.write(&buf[..]).unwrap();
    }

    fn coeff(&self, idx: usize, p: u32) -> i32 {
        let row = &self.rows[idx];
        if let Ok(j) = row.binary_search_by_key(&p, |&(_p, _)| _p) {
            row[j].1
        } else {
            0
        }
    }

    fn pivot_one(&mut self) -> Option<()> {
        let mut p = 0;
        'mainloop: while self.wmin < self.rows.len() {
            for (&q, &wq) in self.weight.iter() {
                if wq as usize <= self.wmin {
                    for &(q, _) in self.ones.range((q, 0)..(q + 1, 0)) {
                        p = q;
                        break 'mainloop;
                    }
                }
            }
            self.wmin += 1;
        }
        if p == 0 {
            return None;
        }
        self.pivot(p)
    }

    /// Eliminate p from all relations.
    fn pivot(&mut self, p: u32) -> Option<()> {
        //eprintln!("pivot {p}");
        let idx: Option<u32> = {
            let mut idx = None;
            for &(_, i) in self.ones.range((p, 0)..(p + 1, 0)) {
                idx = match idx {
                    None => Some(i),
                    Some(j) if self.rows[i as usize].len() < self.rows[j as usize].len() => Some(i),
                    _ => idx,
                };
            }
            idx
        };
        let Some(idx) = idx else {
            return None;
        };
        if !self.dense && self.nonzero.len() > self.weight.len() * self.rows.len() / 10 {
            self.dense = true;
            self.nonzero.clear();
            // stop updating weights.
        }
        let ci = self.coeff(idx as usize, p);
        debug_assert!(ci == 1 || ci == -1);
        if self.dense {
            // Dense mode.
            for j in 0..self.rows.len() {
                if j == idx as usize {
                    continue;
                }
                let cj = self.coeff(j, p);
                if cj != 0 {
                    self.rowsub(j, idx as usize, cj * ci)?;
                }
            }
            self.weight.remove(&p);
        } else {
            // Sparse mode.
            let indices: Vec<usize> = self
                .nonzero
                .range((p, 0)..(p + 1, 0))
                .map(|&(_, i)| i as usize)
                .collect();
            for j in indices {
                if j == idx as usize {
                    continue;
                }
                let cj = self.coeff(j, p);
                // Stop now in case of overflow.
                self.rowsub(j, idx as usize, cj * ci)?;
            }
        }
        let ri = self.rows[idx as usize].clone();
        // update indices
        self.del_row(idx as usize);
        // pop p
        let mut rel = ri;
        for i in 0..rel.len() {
            if rel[i].0 == p {
                rel.remove(i);
                break;
            }
        }
        if ci == 1 {
            // p + sum(ei [li]) == 0
            // => p = sum(-ei [li])
            for i in 0..rel.len() {
                rel[i].1 = -rel[i].1;
            }
        }
        self.removed.push((p, rel));
        Some(())
    }

    fn update_index(&mut self, idx: usize, p: u32, old: Option<i32>, new: Option<i32>) {
        match (old, new) {
            (None, None) => {}
            (None, Some(x)) => {
                if x.unsigned_abs() == 1 {
                    self.ones.insert((p, idx as u32));
                }
                if !self.dense {
                    self.weight.entry(p).and_modify(|w| *w += 1);
                    self.nonzero.insert((p, idx as u32));
                }
            }
            (Some(x), None) => {
                if x.unsigned_abs() == 1 {
                    self.ones.remove(&(p, idx as u32));
                }
                if !self.dense {
                    self.weight.entry(p).and_modify(|w| *w -= 1);
                    if self.weight.get(&p) == Some(&0) {
                        self.weight.remove(&p);
                    }
                    self.nonzero.remove(&(p, idx as u32));
                }
            }
            (Some(x), Some(y)) => {
                if x.unsigned_abs() == 1 && y.unsigned_abs() != 1 {
                    self.ones.remove(&(p, idx as u32));
                } else if x.unsigned_abs() != 1 && y.unsigned_abs() == 1 {
                    self.ones.insert((p, idx as u32));
                }
            }
        }
    }

    fn del_row(&mut self, idx: usize) {
        for &(p, e) in &self.rows[idx] {
            if !self.dense {
                self.weight.entry(p).and_modify(|w| *w -= 1);
                if self.weight.get(&p) == Some(&0) {
                    self.weight.remove(&p);
                }
            }
            self.nonzero.remove(&(p, idx as u32));
            if e.unsigned_abs() == 1 {
                self.ones.remove(&(p, idx as u32));
            }
        }
        self.rows[idx as usize].clear();
    }

    // Apply operation row[i]-=c*row[j]
    // Returns false in case of overflow.
    #[must_use]
    fn rowsub(&mut self, i: usize, j: usize, c: i32) -> Option<()> {
        debug_assert!(c != 0 && i != j);
        let li = self.rows[i].len();
        let lj = self.rows[j].len();
        let mut res: Vec<(u32, i32)> = Vec::with_capacity(max(li, lj));
        let mut ii = 0;
        let mut jj = 0;
        while ii < li || jj < lj {
            let pei = self.rows[i].get(ii);
            let pej = self.rows[j].get(jj);
            match (pei, pej) {
                (Some(&(p, e)), None) => {
                    // No change
                    res.push((p, e));
                    ii += 1;
                }
                (None, Some(&(p, e))) => {
                    // New entry
                    let pe = (p, e.checked_mul(-c)?);
                    self.update_index(i, p, None, Some(pe.1));
                    res.push(pe);
                    jj += 1;
                }
                (Some(&(pi, ei)), Some(&(pj, _))) if pi < pj => {
                    // No change
                    res.push((pi, ei));
                    ii += 1;
                }
                (Some(&(pi, _)), Some(&(pj, ej))) if pi > pj => {
                    let pe = (pj, ej.checked_mul(-c)?);
                    self.update_index(i, pj, None, Some(pe.1));
                    res.push(pe);
                    jj += 1;
                }
                (Some(&(pi, ei)), Some(&(pj, ej))) if pi == pj => {
                    let e = ei.checked_add(ej.checked_mul(-c)?)?;
                    if e != 0 {
                        res.push((pi, e));
                    }
                    self.update_index(i, pi, Some(ei), if e == 0 { None } else { Some(e) });
                    ii += 1;
                    jj += 1;
                }
                _ => unreachable!(),
            }
        }
        self.rows[i] = res;
        Some(())
    }
}

fn write_relations(snf: &matrixint::SmithNormalForm, outdir: &PathBuf) {
    let mut removed = snf.removed.clone();
    removed.reverse();
    let mut buf = vec![];
    for (p, rel) in &removed {
        write!(&mut buf, "{p} =").unwrap();
        for &(l, e) in rel {
            write!(&mut buf, " {l}^{e}").unwrap();
        }
        buf.push(b'\n');
    }
    {
        let mut w = fs::File::create(outdir.join("relations.removed")).unwrap();
        w.write(&buf[..]).unwrap();
    }
    let mut buf = vec![];
    for row in &snf.rows {
        for (&p, &e) in snf.gens.iter().zip(row) {
            if e == 0 {
                continue;
            }
            write!(&mut buf, "{p}^{e} ").unwrap();
        }
        buf.push(b'\n');
    }
    {
        // Overwrite filtered relations.
        let mut w = fs::File::create(outdir.join("relations.filtered")).unwrap();
        w.write(&buf[..]).unwrap();
    }
}

fn write_group_structure(snf: &matrixint::SmithNormalForm, outdir: &PathBuf) {
    if snf.gens.len() > 1 {
        return; // not yet implemented
    }
    // Cyclic group, we know how to do this.
    let mut w = fs::File::create(outdir.join("group.structure")).unwrap();
    let mut buf = vec![];
    writeln!(&mut buf, "G {}", snf.h).unwrap();
    writeln!(&mut buf, "{} 1", snf.gens[0]).unwrap();
    w.write(&buf[..]).unwrap();

    let mut w = fs::File::create(outdir.join("group.structure.extra")).unwrap();
    let mut buf = vec![];
    let mut coords = BTreeMap::<u32, i128>::new();
    coords.insert(snf.gens[0], 1);
    let mut removed = snf.removed.clone();
    removed.reverse();
    'extra: for (p, rel) in removed.iter() {
        let mut dlog = Int::ZERO;
        for (l, e) in rel {
            if !coords.contains_key(l) {
                eprintln!("Missing relations for {l}");
                continue 'extra;
            }
            dlog += Int::from(*e) * Int::from(*coords.get(l).unwrap());
        }
        let dl = i128::cast_from(dlog.rem_euclid(Int::from(snf.h)));
        coords.insert(*p, dl);
        writeln!(&mut buf, "{} {}", p, dl).unwrap();
    }
    w.write(&buf[..]).unwrap();
}
