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
use crate::{Int, Verbosity};

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

pub fn group_structure(
    rels: Vec<CRelation>,
    hest: (f64, f64),
    v: Verbosity,
    outdir: Option<PathBuf>,
) {
    let t0 = Instant::now();
    let mut r = RelFilterSparse::new(rels);
    while let Some(_) = r.pivot_one() {
        if r.weight.len() % 64 == 0 {
            let nrows = r.rows.iter().filter(|r| r.len() > 0).count();
            if v >= Verbosity::Debug {
                eprintln!(
                    "{} columns {} rows {} coefs",
                    r.weight.len(),
                    nrows,
                    r.nonzero.len()
                );
            }
            let nc = r.weight.len();
            let want = nc + nc / 2 + 128;
            if nrows > want {
                let trimmed = r.trim(nrows - want);
                if v >= Verbosity::Debug && trimmed > 0 {
                    eprintln!("{trimmed} extra relations trimmed");
                }
            }
        }
    }
    if let Some(outdir) = outdir.as_ref() {
        r.write_files(outdir);
    }
    let nrows = r.rows.iter().filter(|r| r.len() > 0).count();
    eprintln!(
        "Filtered matrix has {} columns {} rows (elapsed: {:.3}s)",
        r.weight.len(),
        nrows,
        t0.elapsed().as_secs_f64()
    );
    let mut r = matrixint::SmithNormalForm::new(&r.rows, r.removed, hest.0, hest.1);
    eprintln!("Class number is {}", r.h);
    if let Some(outdir) = outdir.as_ref() {
        let mut w = fs::File::create(outdir.join("classnumber")).unwrap();
        writeln!(w, "{}", r.h).unwrap();
    }
    r.reduce();
    // Removed relations, in reverse order
    if let Some(outdir) = outdir.as_ref() {
        write_relations(&r, outdir);
        let group = write_group_structure(&r, outdir);
        eprintln!(
            "Group structure computed in {:.3}s",
            t0.elapsed().as_secs_f64()
        );
        std::io::stdout().write_all(&group).unwrap();
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
    // Set of nonzero indices (prime, row index)
    // We never remove elements, except when a prime is eliminated.
    nonzero: BTreeSet<(u32, u32)>,
    // An array of removed relations, arranged as a triangular matrix.
    // Each relation involves primes appearing afterwards.
    removed: Vec<(u32, Vec<(u32, i32)>)>,
    // A set of primes that cannot be handled (they have no coefficient 1).
    skip: BTreeSet<u32>,
    // Min weight
    wmin: usize,
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
            nonzero,
            removed: vec![],
            skip: BTreeSet::new(),
            wmin: 0,
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
                if wq as usize <= self.wmin && !self.skip.contains(&q) {
                    for &(q, idx) in self.nonzero.range((q, 0)..(q + 1, 0)) {
                        if self.coeff(idx as usize, q).unsigned_abs() == 1 {
                            p = q;
                            break 'mainloop;
                        }
                    }
                    self.skip.insert(q);
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
        let idx: Option<u32> = {
            let mut idx = None;
            for &(_, i) in self.nonzero.range((p, 0)..(p + 1, 0)) {
                if self.coeff(i as usize, p).unsigned_abs() != 1 {
                    continue;
                }
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

        let ci = self.coeff(idx as usize, p);
        debug_assert!(ci == 1 || ci == -1);
        // A superset of interesting indices.
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
            if cj == 0 {
                continue;
            }
            // Stop now in case of overflow.
            self.rowsub(j, idx as usize, cj * ci)?;
        }
        let ri = self.rows[idx as usize].clone();
        self.rows[idx as usize].clear();
        // update indices
        self.weight.remove(&p);
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

    fn trim(&mut self, count: usize) -> usize {
        let mut counts = vec![0; self.weight.len() + 1];
        for r in &self.rows {
            counts[r.len()] += 1;
        }
        let mut trimmed = 0;
        let mut threshold = self.weight.len() + 1;
        while threshold > 0 {
            let n = counts[threshold - 1];
            if trimmed + n < count {
                trimmed += n;
                threshold -= 1;
            } else {
                break;
            }
        }
        for r in self.rows.iter_mut() {
            if r.len() >= threshold {
                r.clear();
            }
        }
        trimmed
    }

    fn add_index(&mut self, idx: usize, p: u32) {
        if self.nonzero.insert((p, idx as u32)) {
            self.weight.entry(p).and_modify(|w| *w += 1);
        }
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
            let (pi, ei) = if ii < li {
                self.rows[i][ii]
            } else {
                (u32::MAX, 0)
            };
            let (pj, ej) = if jj < lj {
                self.rows[j][jj]
            } else {
                (u32::MAX, 0)
            };
            if pi < pj {
                // No change
                res.push((pi, ei));
                ii += 1;
            } else if pi > pj {
                // New entry
                let pe = (pj, ej.checked_mul(-c)?);
                self.add_index(i, pj);
                res.push(pe);
                jj += 1;
            } else {
                let e = ei.checked_add(ej.checked_mul(-c)?)?;
                if e != 0 {
                    res.push((pi, e));
                }
                ii += 1;
                jj += 1;
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
}

fn write_group_structure(snf: &matrixint::SmithNormalForm, outdir: &PathBuf) -> Vec<u8> {
    let n = snf.gens.len();
    // Cyclic group, we know how to do this.
    let mut buf = vec![];
    // Group invariants
    write!(&mut buf, "G").unwrap();
    let mut ds = vec![];
    for i in 0..n {
        let d = snf.rows[i][i];
        if d != 1 {
            write!(&mut buf, " {d}").unwrap();
        }
        ds.push(d);
    }
    buf.push(b'\n');
    // Coordinates
    // If D = PUQ is diagonal, it means that Q^-1 gens are cyclic
    let mut coords = BTreeMap::<u32, Vec<i128>>::new();
    for i in 0..n {
        let p = snf.gens[i];
        let mut c = snf.q[i].clone();
        write!(&mut buf, "{p}").unwrap();
        for j in 0..n {
            if ds[j] != 1 {
                c[j] %= ds[j];
                write!(&mut buf, " {}", c[j]).unwrap();
            }
        }
        buf.push(b'\n');
        coords.insert(p, c);
    }
    let mut w = fs::File::create(outdir.join("group.structure")).unwrap();
    w.write(&buf[..]).unwrap();
    let group = buf;
    // Compute coordinates for eliminated relations
    let mut w = fs::File::create(outdir.join("group.structure.extra")).unwrap();
    let mut buf = vec![];
    let mut removed = snf.removed.clone();
    removed.reverse();
    let mut skipped = 0;
    'extra: for (p, rel) in removed.iter() {
        let mut dlog = vec![Int::ZERO; ds.len()];
        for (l, e) in rel {
            if !coords.contains_key(l) {
                skipped += 1;
                continue 'extra;
            }
            let v = coords.get(l).unwrap();
            for idx in 0..ds.len() {
                dlog[idx] += Int::from(*e) * Int::from(v[idx]);
            }
        }
        for idx in 0..ds.len() {
            dlog[idx] = dlog[idx].rem_euclid(Int::from(ds[idx]));
        }
        let dl: Vec<i128> = dlog.into_iter().map(i128::cast_from).collect();
        write!(&mut buf, "{p}").unwrap();
        for (x, &d) in dl.iter().zip(&ds) {
            if d != 1 {
                write!(&mut buf, " {x}").unwrap();
            }
        }
        buf.push(b'\n');
        coords.insert(*p, dl);
    }
    if skipped > 0 {
        eprintln!("{skipped} discrete logs skipped due to missing relations");
    }
    w.write(&buf[..]).unwrap();
    group
}
