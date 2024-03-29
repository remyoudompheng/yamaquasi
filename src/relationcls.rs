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
//!
//! This module also implements the computation of the class group
//! structure from the set of relations. It uses both dense and sparse
//! linear algebra, and assumes that the class number is always less
//! than 256 bits (which is true if the input number does not exceed
//! 500 bits).

use std::cmp::min;
use std::collections::{BTreeMap, BTreeSet};
use std::default::Default;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bnum::cast::CastFrom;
use bnum::types::I256;

use crate::matrix::intdense as matdense;
use crate::matrix::intsparse as matsparse;
use crate::{Int, Uint, Verbosity};

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
            _ => (),
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
                .write_all(&line)
                .unwrap();
        }
        if clen > 0 {
            self.n_cycles[min(self.n_cycles.len(), clen) - 1] += 1;
        }
        self.emitted.push(r);
    }
}

pub struct ClassGroup {
    pub h: Uint,
    pub invariants: Vec<u128>,
    /// A list of (p, vs) where vs are the coordinates of [p]
    /// in the class group.
    pub gens: Vec<(u32, Vec<u128>)>,
}

/// Compute class group structure from a set of relations obtained by sieving.
///
/// A class number estimate is provided through approximate bounds `hest`.
/// The bounds are allowed to be slightly incorrect.
///
/// The choice of sparse/dense linear algebra can be overridden by `use_sparse`.
pub fn group_structure(
    rels: Vec<CRelation>,
    use_sparse: Option<bool>,
    hest: (f64, f64),
    v: Verbosity,
    outdir: Option<PathBuf>,
    tpool: Option<&rayon::ThreadPool>,
) -> Option<ClassGroup> {
    // Sparse linear algebra is used for discriminants above 200 bits.
    // The caller can adjust this using the knowledge of actual factor base.
    let use_sparse = use_sparse.unwrap_or(hest.1.log2() > 100.0);
    if use_sparse {
        group_structure_sparse(rels, hest, v, outdir, tpool)
    } else {
        group_structure_dense(rels, hest, v, outdir)
    }
}

fn group_structure_dense(
    rels: Vec<CRelation>,
    hest: (f64, f64),
    v: Verbosity,
    outdir: Option<PathBuf>,
) -> Option<ClassGroup> {
    let t0 = Instant::now();
    let mut r = RelFilterSparse::new(rels);
    while let Some(_) = r.pivot_one() {
        if r.weight.len() % 64 == 0 {
            let nrows = r.nonzero_rows;
            if v >= Verbosity::Debug {
                eprintln!(
                    "{} columns {} rows {} coefs",
                    r.weight.len(),
                    nrows,
                    r.nonzero_coeffs
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
    let dups = r.remove_duplicates();
    if v >= Verbosity::Info {
        eprintln!("Removed {dups} duplicate rows");
    }
    if let Some(outdir) = outdir.as_ref() {
        r.write_files(outdir);
    }
    let nrows = r.rows.iter().filter(|r| r.len() > 0).count();
    if v >= Verbosity::Info {
        eprintln!(
            "Filtered matrix has {} columns {} rows (elapsed: {:.3}s)",
            r.weight.len(),
            nrows,
            t0.elapsed().as_secs_f64()
        );
    }
    let mut r = matdense::SmithNormalForm::new(&r.rows, r.removed, hest.0, hest.1);
    r.verbose = v >= Verbosity::Info;
    if v >= Verbosity::Info {
        eprintln!(
            "Class number is {} elapsed {:.3}s",
            r.h,
            t0.elapsed().as_secs_f64()
        );
    }
    if let Some(outdir) = outdir.as_ref() {
        let mut w = fs::File::create(outdir.join("classnumber")).unwrap();
        writeln!(w, "{}", r.h).unwrap();
    }
    r.reduce();
    // Removed relations, in reverse order
    if let Some(outdir) = outdir.as_ref() {
        write_relations(&r, outdir);
    }
    let mut g = ClassGroup {
        h: Uint::cast_from(r.h),
        invariants: vec![],
        gens: vec![],
    };
    let n = r.gens.len();
    for i in 0..n {
        let d = r.rows[i][i];
        if d != 1 {
            g.invariants.push(d as u128);
        }
    }
    for i in 0..n {
        let p = r.gens[i];
        let mut c = vec![];
        for j in 0..n {
            let d = r.rows[j][j];
            if d != 1 {
                c.push((r.q[i][j] % d) as u128);
            }
        }
        g.gens.push((p, c));
    }
    if v >= Verbosity::Info {
        eprintln!(
            "Group structure computed in {:.4}s",
            t0.elapsed().as_secs_f64()
        );
    }
    if let Some(outdir) = outdir.as_ref() {
        write_extra_group_structure(&r, &g, outdir);
    }
    Some(g)
}

pub fn group_structure_sparse(
    rels: Vec<CRelation>,
    hest: (f64, f64),
    v: Verbosity,
    outdir: Option<PathBuf>,
    tpool: Option<&rayon::ThreadPool>,
) -> Option<ClassGroup> {
    let t0 = Instant::now();
    let mut r = RelFilterSparse::new(rels);
    let dsize = 2 * hest.1.log2().round() as usize + 1;
    let maxweight = dsize / 2;
    let mut lasttrim = (2, r.weight.len());
    while let Some(_) = r.pivot_one() {
        let nrows = r.nonzero_rows;
        let nonzero = r.nonzero_coeffs;
        if nonzero > nrows * maxweight {
            break;
        }
        let avg = nonzero as f64 / nrows as f64;
        if r.wmin > lasttrim.0 && r.weight.len() + 64 < lasttrim.1 {
            lasttrim = (r.wmin, r.weight.len());
            if v >= Verbosity::Debug {
                eprintln!("{} columns {} rows avg {:.1}", r.weight.len(), nrows, avg);
            }
            // During merge keep a large number of rows.
            let nc = r.weight.len();
            let want = 3 * nc + 128;
            if nrows > want {
                let trimmed = r.trim(nrows - want);
                if v >= Verbosity::Debug && trimmed > 0 {
                    eprintln!("{trimmed} extra relations trimmed");
                }
            }
        }
    }
    let dups = r.remove_duplicates();
    if v >= Verbosity::Info {
        eprintln!("Removed {dups} duplicate rows");
    }
    // Trim again to keep a small row excess.
    let nc = r.weight.len();
    let want = nc + 2 * dsize + 128;
    if r.rows.len() > want {
        let trimmed = r.trim(r.rows.len() - want);
        if v >= Verbosity::Debug && trimmed > 0 {
            eprintln!("{trimmed} extra relations trimmed");
        }
    }
    for i in 0..r.rows.len() {
        while i < r.rows.len() && r.rows[i].is_empty() {
            r.rows.swap_remove(i);
        }
    }
    if let Some(outdir) = outdir.as_ref() {
        r.write_files(outdir);
    }
    let nrows = r.rows.len();
    let n_coeffs = r.rows.iter().map(|r| r.len()).sum::<usize>();
    let avgw = n_coeffs as f64 / nrows as f64;
    if v >= Verbosity::Info {
        eprintln!(
            "Filtered matrix has {} columns {} rows average weight {avgw:.1} (elapsed: {:.3}s)",
            r.weight.len(),
            nrows,
            t0.elapsed().as_secs_f64()
        );
    }

    let mut gens = BTreeSet::new();
    for r in &r.rows {
        for &(p, e) in r {
            if e != 0 {
                gens.insert(p);
            }
        }
    }
    let gens: Vec<u32> = gens.into_iter().collect();
    //eprintln!("{gens:?}");
    let rows2: Vec<Vec<(u32, i32)>> = r
        .rows
        .iter()
        .map(|r| {
            let mut row = Vec::with_capacity(r.len());
            for (p, e) in r {
                let pidx = gens.binary_search(p).unwrap();
                row.push((pidx as u32, *e));
            }
            row
        })
        .collect();
    let h = matsparse::compute_lattice_index(gens.len(), &rows2, hest.0, hest.1, tpool);
    if v >= Verbosity::Info {
        eprintln!(
            "Class number {h} elapsed {:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }
    if let Some(outdir) = outdir.as_ref() {
        let mut w = fs::File::create(outdir.join("classnumber")).unwrap();
        writeln!(w, "{}", h).unwrap();
    }
    // Now factor class number and compute class group structure.
    // The class number is small.
    let factors = {
        let mut prefs = crate::Preferences::default();
        prefs.verbosity = Verbosity::Silent;
        crate::factor(Uint::cast_from(h), crate::Algo::Auto, &prefs)
    }
    .unwrap();
    if v >= Verbosity::Info {
        eprintln!("Class number factors {:?}", factors);
    }
    // FIXME: structure is incomplete.
    let g = ClassGroup {
        h: Uint::cast_from(h),
        invariants: vec![],
        gens: vec![],
    };
    Some(g)
}

/// A structure to filter relations when they are sparse.
/// It implements a simple form of structured Gauss elimination.
struct RelFilterSparse {
    // Array of relations represented by sparse, sorted
    // vectors of (prime, exponent).
    rows: Vec<Vec<(u32, i32)>>,
    // How many times each given prime appears.
    // It may be larger than the actual number, but never smaller.
    weight: BTreeMap<u32, u32>,
    // Set of nonzero indices prime => [row index]
    // We never remove elements, except when a prime is eliminated.
    nonzero: BTreeMap<u32, Vec<u32>>,
    // An array of removed relations, arranged as a triangular matrix.
    // Each relation involves primes appearing afterwards.
    removed: Vec<(u32, Vec<(u32, i32)>)>,
    // A set of primes that cannot be handled (they have no coefficient 1).
    skip: BTreeSet<u32>,
    // Min weight
    wmin: usize,
    // List of candidates for next elimination
    nextelims: Vec<u32>,
    // Stats
    nonzero_rows: usize,
    nonzero_coeffs: usize,
}

impl RelFilterSparse {
    fn new(rels: Vec<CRelation>) -> Self {
        let mut rows = Vec::with_capacity(rels.len());
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
            if !row.is_empty() {
                rows.push(row);
            }
        }
        let mut weight = BTreeMap::new();
        let nonzero_rows = rows.len();
        let mut nonzero_total: usize = 0;
        for r in &rows {
            for &(p, _) in r {
                let w: &mut u32 = weight.entry(p).or_default();
                *w += 1;
            }
            nonzero_total += r.len();
        }
        let mut nonzero = BTreeMap::new();
        for (&p, &w) in &weight {
            nonzero.insert(p, Vec::with_capacity(w as usize));
        }
        for i in 0..rows.len() {
            for &(p, _) in &rows[i] {
                nonzero.get_mut(&p).unwrap().push(i as u32);
            }
        }
        Self {
            rows,
            weight,
            nonzero,
            removed: vec![],
            skip: BTreeSet::new(),
            wmin: 0,
            nextelims: vec![],
            nonzero_rows,
            nonzero_coeffs: nonzero_total,
        }
    }

    fn write_files(&self, outdir: &Path) {
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
        w.write_all(&buf[..]).unwrap();
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
        while self.wmin < self.rows.len() {
            if self.nextelims.is_empty() {
                // Fetch next low weight candidates
                if self.wmin < 50 {
                    self.wmin += 1;
                } else {
                    self.wmin += 2;
                }
                for (&q, &wq) in self.weight.iter() {
                    if wq as usize <= self.wmin && !self.skip.contains(&q) {
                        self.nextelims.push(q);
                    }
                }
            }
            'qloop: while let Some(q) = self.nextelims.pop() {
                // Largest primes are eliminated first.
                if let Some(nz) = self.nonzero.get(&q) {
                    let mut can_pivot = false;
                    let mut weight = 0;
                    for &idx in nz {
                        let c = self.coeff(idx as usize, q);
                        if c.unsigned_abs() == 1 {
                            can_pivot = true;
                        }
                        if c != 0 {
                            weight += 1;
                        }
                    }
                    if weight > self.wmin {
                        // Too many rows containing this prime.
                        continue 'qloop;
                    }
                    if can_pivot {
                        return self.pivot(q);
                    }
                }
                self.skip.insert(q);
            }
        }
        None
    }

    /// Eliminate p from all relations.
    fn pivot(&mut self, p: u32) -> Option<()> {
        let idx: Option<u32> = {
            let mut idx = None;
            let nz = self.nonzero.get(&p).unwrap();
            for &i in nz {
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
        let indices = self.nonzero.get(&p).unwrap().clone();
        for j in indices {
            if j == idx {
                continue;
            }
            let j = j as usize;
            let cj = self.coeff(j, p);
            if cj == 0 {
                continue;
            }
            // Stop now in case of overflow.
            self.rowsub(j, idx as usize, cj * ci)?;
        }
        let ri = self.rows[idx as usize].clone();
        self.remove_row(idx as usize);
        // update indices
        self.weight.remove(&p);
        self.nonzero.remove(&p);
        self.save_removed(p, ri);
        Some(())
    }

    fn save_removed(&mut self, p: u32, rel: Vec<(u32, i32)>) {
        let mut rel = rel;
        let mut ci = 0;
        for i in 0..rel.len() {
            if rel[i].0 == p {
                ci = rel.remove(i).1;
                break;
            }
        }
        assert!(ci.unsigned_abs() == 1);
        if ci == 1 {
            // p + sum(ei [li]) == 0
            // => p = sum(-ei [li])
            for i in 0..rel.len() {
                rel[i].1 = -rel[i].1;
            }
        }
        self.removed.push((p, rel));
    }

    // Eliminate duplicate relations where r1=±r2.
    // Indexes are no longer valid after this operation.
    fn remove_duplicates(&mut self) -> usize {
        // Remove empty rows.
        for i in 0..self.rows.len() {
            while i < self.rows.len() && self.rows[i].is_empty() {
                self.rows.swap_remove(i);
            }
        }
        // Now eliminate duplicates
        // The sign is normalized so that pairs (r,-r) can also be eliminated.
        let l = self.rows.len();
        for row in self.rows.iter_mut() {
            if matches!(row.first(), Some(&(_, e)) if e < 0) {
                let neg: Vec<_> = row.iter().map(|&(p, e)| (p, -e)).collect();
                assert_eq!(row.len(), neg.len());
                *row = neg;
            }
        }
        self.rows.sort();
        self.rows.dedup();
        l - self.rows.len()
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
        for idx in 0..self.rows.len() {
            if self.rows[idx].len() >= threshold {
                self.remove_row(idx);
            }
        }
        trimmed
    }

    fn add_index(&mut self, idx: usize, p: u32) {
        self.nonzero.entry(p).and_modify(|v| v.push(idx as u32));
        self.weight.entry(p).and_modify(|w| *w += 1);
    }

    fn remove_row(&mut self, idx: usize) {
        if self.rows[idx].is_empty() {
            return;
        }
        for &(p, _) in &self.rows[idx] {
            self.weight
                .entry(p)
                .and_modify(|w| *w = w.checked_sub(1).unwrap());
        }
        self.nonzero_rows -= 1;
        self.nonzero_coeffs -= self.rows[idx].len();
        self.rows[idx].clear();
    }

    // Apply operation row[i]-=c*row[j]
    // Returns false in case of overflow.
    #[must_use]
    fn rowsub(&mut self, i: usize, j: usize, c: i32) -> Option<()> {
        debug_assert!(c != 0 && i != j);
        let li = self.rows[i].len();
        let lj = self.rows[j].len();
        let mut res: Vec<(u32, i32)> = Vec::with_capacity(min(self.weight.len(), li + lj));
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
        self.nonzero_coeffs += res.len();
        self.nonzero_coeffs -= li;
        if res.len() == 0 {
            self.nonzero_rows -= 1;
        }
        self.rows[i] = res;
        Some(())
    }
}

fn write_relations(snf: &matdense::SmithNormalForm, outdir: &Path) {
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
        w.write_all(&buf[..]).unwrap();
    }
}

fn write_extra_group_structure(snf: &matdense::SmithNormalForm, g: &ClassGroup, outdir: &Path) {
    let mut coords = BTreeMap::<u32, Vec<u128>>::new();
    for (p, v) in &g.gens {
        coords.insert(*p, v.clone());
    }
    // Compute coordinates for eliminated relations
    let mut w = fs::File::create(outdir.join("group.structure.extra")).unwrap();
    let mut buf = vec![];
    let mut removed = snf.removed.clone();
    removed.reverse();
    let mut skipped = 0;
    let n = g.invariants.len();
    'extra: for (p, rel) in removed.iter() {
        let mut dlog = vec![I256::ZERO; n];
        for (l, e) in rel {
            if !coords.contains_key(l) {
                skipped += 1;
                continue 'extra;
            }
            let v = coords.get(l).unwrap();
            for idx in 0..n {
                dlog[idx] += I256::from(*e).checked_mul(I256::from(v[idx])).unwrap();
            }
        }
        for idx in 0..n {
            let d = g.invariants[idx];
            dlog[idx] = dlog[idx].rem_euclid(I256::from(d));
        }
        let dl: Vec<u128> = dlog.into_iter().map(u128::cast_from).collect();
        write!(&mut buf, "{p}").unwrap();
        for x in &dl {
            write!(&mut buf, " {x}").unwrap();
        }
        buf.push(b'\n');
        coords.insert(*p, dl);
    }
    if skipped > 0 {
        eprintln!("{skipped} discrete logs skipped due to missing relations");
    }
    w.write_all(&buf[..]).unwrap();
}
