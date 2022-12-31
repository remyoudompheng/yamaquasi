// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Shared common implementation of sieve.
//!
//! Sieving using a quadratic polynomial can be reduced to handling
//! a list of primes p and 1 or 2 roots of the polynomial modulo p.
//!
//! This requires approximately sum(1/p) operations per element
//! of the interval to be sieved, since about half of all primes
//! will have 2 roots each.
//!
//! For each element such that Ax^2+Bx+C is smooth we need to determine
//! the list of primes such that Ax^2+Bx+C=0 mod p. For a given prime,
//! such elements are given by x0 + kp where x0 is some root of the polynomial.
//!
//! Due to intensive operations on vectors, this file uses
//! many unsafe indexing operations.
//!
//! # Small primes
//!
//! Since the number of operations is O(M log log maxprime)
//! most operations are caused by small primes:
//!
//! In the case of primes where 1e6+3 is a quadratic residue
//! the distribution of sum(#roots/p) is:
//!
//! For 5k primes (largest ~100e3, sum~=3.14):
//! > 50% for 5 smallest primes
//! ~ 25% for larger primes < 1024
//! ~ 5% for primes in (1024, 4096)
//! ~ 7% for primes in (4096, 32768)
//! ~ 3% for primes above 32768
//! The product of 5 smallest primes (2*3*7*11*13) is 13 bits large.
//!
//! For 39k primes (largest ~1e6, sum~=3.32):
//! > 60% for 10 smallest primes
//! ~ 15% for larger primes < 1024
//! ~ 5% for primes in (1024, 4096)
//! ~ 8% for primes in (4096, 32768)
//! ~ 8% for primes above 32768
//! The product of 10 smallest primes (2*...*43) is 38 bits large.
//!
//! In the case of MPQS/SIQS
//! ignoring these small primes may add 20 additional candidates.
//!
//! # Large primes
//!
//! In constrast, large primes sieve hits are very sparse: they will
//! typically divide less than half of interval elements. We can use
//! loosely sorted buckets to quickly find which primes divide a given
//! element. This is similar to a hashmap-based strategy in msieve.
//!
//! Since sieve hits are distributed randomly, we need to ensure memory
//! locality: this is done by "layering" tables according to prime size.
//! The sorting key is log(p) (bit length of p) which avoids storing
//! that value and is also naturally sorted during iteration over the factor
//! base.
//!
//! For a bucket of size B, the expected number of sieve hits for a range
//! of primes is sum(B/p for large p) (each p has 2 roots but the factor base
//! contains half of primes), and the variance is almost equal when p is large,
//! as a sum of Bernoulli variables with parameter B/p.
//!
//! For each size class (bit length), the sum(1/p) is:
//!
//! class   sum(1/p) #primes  #fbase/256
//! size 16  0.0644    3030     5
//! size 17  0.0606    5709    11
//! size 18  0.0570   10749    20
//! size 19  0.0541   20390    39
//! size 20  0.0512   38635    75
//! size 21  0.0488   73586   143
//! size 22  0.0465  140336   274
//! size 23  0.0444  268216   523
//! size 24  0.0426  513708  1003
//!
//! For an interval of size 256, 32 slots are enough to accomodate the average
//! number of hits + 3.82σ (for size 16) or >4σ (size 17+) meaning that overflow
//! will happen with probability < 1e-4.
//!
//! It is fine to simply ignore overflow (we will lose a few relations but keep
//! algorithm simple). Each slot contains an offset (1 byte) and an approximage
//! prime index (1 byte) so the size of each table is interval_size/4.
//!
//! TODO: Bibliography

use std::cmp::{max, min};
use wide;

use crate::arith::{self, Num};
use crate::fbase::FBase;
use crate::Int;

pub const BLOCK_SIZE: usize = 32 * 1024;
const OFFSET_NONE: u16 = 0xffff;

const LARGE_PRIME_LOG: usize = 16;
const PRIME_BUCKET_SIZES: &[usize] = &[7, 13, 23, 43, 79, 147, 277, 527, 1009];
const PRIME_BUCKET_DIVIDERS: &[arith::Divider31] = &[
    arith::Divider31::new(7),
    arith::Divider31::new(13),
    arith::Divider31::new(23),
    arith::Divider31::new(43),
    arith::Divider31::new(79),
    arith::Divider31::new(147),
    arith::Divider31::new(277),
    arith::Divider31::new(527),
    arith::Divider31::new(1009),
];

// Map prime indices to 0..256
fn prime_bucket(pidx: u32, logp: usize) -> u32 {
    PRIME_BUCKET_DIVIDERS[logp - 16].divu31(pidx)
}

// The sieve makes an array of offsets progress through the
// sieve interval. All offsets are assumed to fit in a u16.
#[derive(Clone)]
pub struct Sieve<'a> {
    pub offset: i64,
    pub nblocks: usize,
    pub blk_no: usize,
    // First offset of prime actually used in sieve
    pub idxskip: usize,
    // Factor base
    pub fbase: &'a FBase,
    pub pfunc: &'a (dyn Fn(usize) -> SievePrime + Sync),
    // Small primes
    pub lo: Vec<u16>,      // cursor offsets
    pub lo_prev: Vec<u16>, // clone of lo before sieving block.
    // Result of sieve.
    pub blk: [u8; BLOCK_SIZE],
    // Cache for large prime hits
    pub tables: Vec<SieveTable>,
}

pub struct SievePrime {
    pub p: u32,
    pub offsets: [Option<u32>; 2],
}

impl<'a> Sieve<'a> {
    // Initialize sieve starting at offset = -M
    // Function f determines starting offsets for a given prime
    // and returns log(p). This allows reading log(p) from an
    // alterante memory location when relevant.
    pub fn new<F>(offset: i64, nblocks: usize, fbase: &'a FBase, f: &'a F) -> Self
    where
        F: Fn(usize) -> SievePrime + Sync,
    {
        // Skip enough primes to skip ~half of operations
        let pskip = match fbase.len() {
            0..=1999 => 3,
            2000..=4999 => 5,
            5000..=9999 => 7,
            10000..=19999 => 11,
            20000..=49999 => 13,
            _ => 17,
        };

        // Primes duplicated for each cursor offset.
        let n_small = fbase.idx_by_log[LARGE_PRIME_LOG];
        let mut offs = Vec::with_capacity(2 * n_small);
        let maxprime = fbase.bound();
        // Need tables for logp in 16..=log(maxprime)
        let maxlog = 32 - u32::leading_zeros(maxprime as u32) as usize;
        let mut tables: Vec<_> = (LARGE_PRIME_LOG..=maxlog)
            .map(|_| SieveTable::new(nblocks))
            .collect();

        fbase.len();
        for log in 0..=maxlog {
            let idx1 = fbase.idx_by_log[log];
            let idx2 = fbase.idx_by_log[log + 1];
            if log < LARGE_PRIME_LOG {
                for idx in idx1..idx2 {
                    let SievePrime {
                        p: _,
                        offsets: [o1, o2],
                    } = f(idx);
                    // Small prime
                    offs.push(if let Some(o1) = o1 {
                        o1 as u16
                    } else {
                        OFFSET_NONE
                    });
                    offs.push(if let Some(o2) = o2 {
                        o2 as u16
                    } else {
                        OFFSET_NONE
                    });
                    debug_assert!(offs.len() == 2 * idx + 2);
                }
            } else {
                assert!(prime_bucket((idx2 - idx1) as u32, log) < 256);
                let table = &mut tables[log - LARGE_PRIME_LOG];
                let pbsize = PRIME_BUCKET_SIZES[log - LARGE_PRIME_LOG];
                let mut pbidx = 0;
                let mut pbucket = 0;
                let interval_size = (nblocks * BLOCK_SIZE) as isize;
                for idx in idx1..idx2 {
                    // Large prime: register them in hashmap
                    let SievePrime {
                        p,
                        offsets: [o1, o2],
                    } = f(idx);
                    let mut kp: isize = 0;
                    let [Some(o1), Some(o2)] = [o1, o2]
                        else { unreachable!("large primes must have 2 roots") };
                    let (o1, o2) = (o1 as isize, o2 as isize);
                    let p = p as isize;
                    let rmax = max(o1, o2) as isize;
                    let m = interval_size - p - rmax;
                    while kp < m {
                        table.add((kp + o1) as usize, pbucket);
                        table.add((kp + o2) as usize, pbucket);
                        table.add((kp + p + o1) as usize, pbucket);
                        table.add((kp + p + o2) as usize, pbucket);
                        kp += 2 * p;
                    }
                    let mut off = o1 as isize + kp;
                    while off < interval_size {
                        table.add(off as usize, pbucket);
                        off += p;
                    }
                    let mut off = o2 as isize + kp;
                    while off < interval_size {
                        table.add(off as usize, pbucket);
                        off += p;
                    }
                    pbidx += 1;
                    if pbidx == pbsize {
                        pbidx = 0;
                        pbucket += 1;
                    }
                }
            }
        }
        let overflows: usize = tables.iter().map(|t| t.n_overflows).sum();
        if overflows > 0 {
            // FIXME
            //eprintln!("bucket overflows {}", overflows);
        }
        let idxskip = 2 * fbase
            .primes
            .iter()
            .position(|&p| p > pskip)
            .unwrap_or(fbase.len());
        let len = offs.len();
        Sieve {
            offset,
            nblocks,
            blk_no: 0,
            idxskip,
            fbase,
            pfunc: f,
            lo: offs,
            lo_prev: vec![0u16; len],
            blk: [0u8; BLOCK_SIZE],
            tables,
        }
    }

    // Recompute largehits/largeoffs for next blocks.
    // This is onyl for classic quadratic sieve where a single
    // polynomial is sieved over a huge interval.
    pub fn rehash(&mut self) {
        self.blk_no = 0;
        if self.nblocks == 0 {
            return;
        }
        for t in &mut self.tables {
            t.reset();
        }
        let mut table_pidx = 0;
        let mut prevlog = 0;
        for (idx, &p) in self.fbase.primes.iter().enumerate() {
            if (p as usize) < BLOCK_SIZE {
                continue; // Small prime
            }
            let l = 32 - u32::leading_zeros(p) as usize;
            if l != prevlog {
                table_pidx = 0;
                prevlog = l;
            }
            let pbucket = prime_bucket(table_pidx, l);
            let table = &mut self.tables[l - LARGE_PRIME_LOG];
            for o in (self.pfunc)(idx).offsets {
                let Some(o) = o else { continue };
                let mut off = o as usize;
                loop {
                    if off >= self.nblocks * BLOCK_SIZE {
                        break;
                    }
                    table.add(off, pbucket);
                    off += p as usize;
                }
            }
            table_pidx += 1;
        }
    }

    pub fn sieve_block(&mut self) {
        let len: usize = BLOCK_SIZE;
        self.blk.fill(0u8);
        let blk = &mut self.blk;
        std::mem::swap(&mut self.lo, &mut self.lo_prev);
        let primes: &[_] = &self.fbase.primes;
        let lo: &mut [_] = &mut self.lo;
        let lo_prev: &[_] = &self.lo_prev;
        unsafe {
            // Smallest primes: we don't update self.blk at all,
            // and simply update offsets for next block.
            for i in 0..self.idxskip {
                let pidx = i / 2;
                let pp = *primes.get_unchecked(pidx) as usize;
                let pdiv = self.fbase.divs.get_unchecked(pidx);
                // off -> off + kp - BLOCK_SIZE
                let mut off: usize = *lo_prev.get_unchecked(i) as usize;
                off = off + pp - pdiv.modu16(BLOCK_SIZE as u16) as usize;
                if off >= pp {
                    off -= pp;
                }
                lo[i] = off as u16;
            }
            // Small primes: perform ordinary sieving.
            // They are guaranteed to have >= 16 hits inside the block.
            for log in 2..=12 {
                // Interval of primes such that bit length == log.
                let i_start = max(self.idxskip, 2 * self.fbase.idx_by_log[log] as usize);
                let i_end = if log < 15 {
                    2 * self.fbase.idx_by_log[log + 1] as usize
                } else {
                    lo.len()
                };
                let log = log as u8;
                for i in i_start / 2..i_end / 2 {
                    let p = *primes.get_unchecked(i);
                    let mut off1: usize = *lo_prev.get_unchecked(2 * i) as usize;
                    let mut off2: usize = *lo_prev.get_unchecked(2 * i + 1) as usize;
                    if off1 != OFFSET_NONE as usize && off2 != OFFSET_NONE as usize {
                        // Almost always true
                        let m = max(off1, off2);
                        let p = p as usize;
                        let mut kp = 0 as usize;
                        let ll = len - p - m;
                        while kp < ll {
                            // both kp+off1 and kp+off2 are in range
                            *blk.get_unchecked_mut(kp + off1) += log;
                            *blk.get_unchecked_mut(kp + off2) += log;
                            *blk.get_unchecked_mut(kp + p + off1) += log;
                            *blk.get_unchecked_mut(kp + p + off2) += log;
                            kp += 2 * p as usize;
                        }
                        off1 += kp;
                        off2 += kp
                    }
                    // Update state.
                    if off1 != OFFSET_NONE as usize {
                        while off1 < len {
                            *blk.get_unchecked_mut(off1) += log;
                            off1 += p as usize;
                        }
                        *lo.get_unchecked_mut(2 * i) = (off1 % BLOCK_SIZE) as u16;
                    }
                    if off2 != OFFSET_NONE as usize {
                        while off2 < len {
                            *blk.get_unchecked_mut(off2) += log;
                            off2 += p as usize;
                        }
                        *lo.get_unchecked_mut(2 * i + 1) = (off2 % BLOCK_SIZE) as u16;
                    }
                }
            }
            // Not so small primes, no need to process both roots at the same time.
            for log in 13..=15 {
                // Interval of primes such that bit length == log.
                let i_start = max(self.idxskip, 2 * self.fbase.idx_by_log[log] as usize);
                let i_end = if log < 15 {
                    2 * self.fbase.idx_by_log[log + 1] as usize
                } else {
                    lo.len()
                };
                for i in i_start..i_end {
                    let pidx = i / 2;
                    let p = *primes.get_unchecked(pidx);
                    let mut off: usize = *lo_prev.get_unchecked(i) as usize;
                    if off == OFFSET_NONE as usize {
                        continue;
                    }
                    let log = log as u8;
                    while off < len {
                        *blk.get_unchecked_mut(off) += log;
                        off += p as usize;
                    }
                    *lo.get_unchecked_mut(i) = (off % BLOCK_SIZE) as u16;
                }
            }
        }
        // Handle large primes
        if self.tables.len() == 0 {
            return;
        }
        let ventries: Vec<(&_, &_)> = unsafe {
            self.tables
                .iter()
                .map(|t| {
                    (
                        t.entries.get_unchecked(
                            self.blk_no * SieveTable::N_ENTRIES
                                ..(self.blk_no + 1) * SieveTable::N_ENTRIES,
                        ),
                        t.blens.get_unchecked(
                            self.blk_no * SieveTable::N_BUCKETS
                                ..(self.blk_no + 1) * SieveTable::N_BUCKETS,
                        ),
                    )
                })
                .collect()
        };
        for bidx in 0..BLOCK_SIZE / BUCKET_WIDTH {
            let base_off = bidx * BUCKET_WIDTH;
            for (tidx, (t, blens)) in ventries.iter().enumerate() {
                let logp = (LARGE_PRIME_LOG + tidx) as u8;
                unsafe {
                    let blen = *blens.get_unchecked(bidx) as usize;
                    for &(boff, _) in t.get_unchecked(bidx * BUCKET_SIZE..bidx * BUCKET_SIZE + blen)
                    {
                        let off = base_off + boff as usize;
                        *blk.get_unchecked_mut(off) += logp;
                    }
                }
            }
        }
    }

    // Debugging method to inspect sizes of smooth factors over the interval.
    pub fn stats(&self) -> [usize; 256] {
        let mut s = self.clone();
        s.sieve_block();
        let mut res = [0; 256];
        for b in s.blk {
            res[b as usize] += 1
        }
        res
    }

    pub fn next_block(&mut self) {
        self.offset += BLOCK_SIZE as i64;
        self.blk_no += 1;
    }

    fn skipbits(&self) -> usize {
        // Extra amount to subtract from threshold if we skipped sieving
        // small primes.
        // Beware that primes typically appear twice.
        let mut skipped = 0usize;
        for i in 0..self.idxskip / 2 {
            let p = self.fbase.primes[i];
            let log = 32 - u32::leading_zeros(p) as usize;
            skipped += log;
        }
        skipped
    }

    fn prime_bucket_bounds(&self, pbucket: usize, logp: usize) -> (usize, usize) {
        let idxs = &self.fbase.idx_by_log;
        let base = idxs[logp] as usize;
        let next = idxs[logp + 1] as usize;
        let bsz = PRIME_BUCKET_SIZES[logp - LARGE_PRIME_LOG];
        (base + pbucket * bsz, min(next, base + (pbucket + 1) * bsz))
    }

    // Slow method to determine whether a prime divides the sieve element
    // at given offset.
    fn is_factor(&self, offset: usize, pidx: usize) -> bool {
        let sp = (self.pfunc)(pidx);
        let div = self.fbase.div(pidx);
        let o = div
            .divmod64(self.blk_no as u64 * BLOCK_SIZE as u64 + offset as u64)
            .1;
        sp.offsets.contains(&Some(o as u32))
    }

    // Returns a list of block offsets and factors (as indices into the factor base).
    // The threshold is given for the "ends" of the interval.
    //
    pub fn smooths(&self, threshold: u8, root: Option<u32>) -> (Vec<u16>, Vec<Vec<usize>>) {
        // As smallest primes have been skipped, values in self.blk
        // are smaller than they should be: subtract an upper bound for
        // the missing part from the threshold.
        //
        // Also subtract log(BLOCK_SIZE) to handle root.
        let skipbits = self.skipbits() + (if root.is_some() { 15 } else { 0 });
        let threshold2 = threshold - min(skipbits, threshold as usize / 2) as u8;
        // How many zeros in half interval size M ?
        let mzeros = u32::leading_zeros(self.nblocks as u32 * BLOCK_SIZE as u32 / 2);

        let mut res: Vec<u16> = vec![];
        let thr16x = wide::u8x16::splat(threshold2 - 1);
        let mut i = 0;
        while i < BLOCK_SIZE {
            unsafe {
                // Cast as [u8;16] to avoid assuming alignment.
                let blk16 = (&self.blk[i] as *const u8) as *const [u8; 16];
                let blk16w = wide::u8x16::new(*blk16);
                if thr16x != blk16w.max(thr16x) {
                    // Some element is > threshold2-1
                    for j in 0..16 {
                        let mut t = (*blk16)[j];
                        if t <= threshold2 {
                            continue;
                        }
                        let ij = (i + j) as u16;
                        // Now add missing log(p) for smallest primes.
                        for i in 0..self.idxskip / 2 {
                            let pp = *self.fbase.primes.get_unchecked(i);
                            let pdiv = self.fbase.divs.get_unchecked(i);
                            let off1 = *self.lo_prev.get_unchecked(2 * i);
                            let off2 = *self.lo_prev.get_unchecked(2 * i + 1);
                            let imod = pdiv.modu16(ij);
                            if imod == off1 || imod == off2 {
                                let log = 32 - u32::leading_zeros(pp) as u8;
                                t += log;
                            }
                        }
                        // Compensate for distance to root.
                        if let Some(r) = root {
                            let x = (ij as i32)
                                + (self.blk_no * BLOCK_SIZE - self.nblocks * BLOCK_SIZE / 2) as i32;
                            let dist = (x.abs() - r as i32).abs();
                            let zeros = u32::leading_zeros(dist as u32);
                            if zeros > mzeros {
                                t += (zeros - mzeros) as u8;
                            }
                        }
                        // Now apply requested threshold.
                        if t >= threshold {
                            res.push(ij)
                        }
                    }
                }
            }
            i += 16;
        }
        if res.len() == 0 {
            return (vec![], vec![]);
        }
        let rlen = res.len();
        // Now find factors.
        let mut facs = vec![vec![]; res.len()];
        for i in 0..self.fbase.idx_by_log[15] {
            // Prime less than BLOCK_SIZE/2
            let i = i as usize;
            unsafe {
                let pdiv = self.fbase.divs.get_unchecked(i);
                let off1 = self.lo_prev[2 * i];
                let off2 = self.lo_prev[2 * i + 1];
                // OFFSET_NONE will never match.
                for idx in 0..rlen {
                    let rmod = pdiv.modu16(res[idx]);
                    if rmod == off1 || rmod == off2 {
                        facs[idx].push(i)
                    }
                }
            }
        }
        // Prime more than BLOCK_SIZE/2
        for i in 2 * self.fbase.idx_by_log[15]..self.lo_prev.len() {
            let pidx = i / 2;
            let off = self.lo_prev[i];
            let p = self.fbase.primes[pidx] as u16;
            for idx in 0..rlen {
                let r = res[idx];
                if r == off || r == off + p {
                    facs[idx].push(pidx)
                }
            }
        }
        // Prime more than BLOCK_SIZE
        // Use reverse lookup table.
        if self.tables.len() == 0 {
            return (res, facs);
        }
        let base_off = self.blk_no * BLOCK_SIZE;
        for j in 0..rlen {
            let r = res[j];
            let (b, boff) = SieveTable::bucket(base_off + r as usize);
            for (tidx, t) in self.tables.iter().enumerate() {
                let logp = tidx + LARGE_PRIME_LOG;
                let blen = t.blens[b] as usize;
                for &(off, pbucket) in &t.entries[b * BUCKET_SIZE..b * BUCKET_SIZE + blen] {
                    if boff as u8 == off {
                        // Walk candidate primes
                        let (pmin, pmax) = self.prime_bucket_bounds(pbucket as usize, logp);
                        for pidx in pmin..pmax {
                            if self.is_factor(r as usize, pidx) {
                                facs[j].push(pidx)
                            }
                        }
                    }
                }
                // Look for offset in overflows (extremely uncommon)
                for &(ooff, pbucket) in &t.overflows[..min(t.overflows.len(), t.n_overflows)] {
                    if ooff == r {
                        let (pmin, pmax) = self.prime_bucket_bounds(pbucket as usize, logp);
                        for pidx in pmin..pmax {
                            if self.is_factor(r as usize, pidx) {
                                facs[j].push(pidx)
                            }
                        }
                    }
                }
            }
        }
        (res, facs)
    }

    // Returns the quotient of x by prime divisors determined
    // by the sieve at index i.
    #[inline]
    pub fn cofactor(&self, i: usize, facs: &[usize], x: &Int) -> (u64, Vec<(i64, u64)>) {
        let mut factors: Vec<(i64, u64)> = Vec::with_capacity(20);
        if x.is_negative() {
            factors.push((-1, 1));
        }
        let xabs = x.abs().to_bits();
        let mut cofactor = xabs;
        for &pidx in facs {
            let pp = self.fbase.p(pidx);
            let pdiv = self.fbase.div(pidx);
            let mut exp = 0;
            loop {
                let (q, r) = pdiv.divmod_uint(&cofactor);
                if r == 0 {
                    cofactor = q;
                    exp += 1;
                } else {
                    break;
                }
            }
            factors.push((pp as i64, exp));
        }
        let cofactor = cofactor.to_u64().unwrap();
        (cofactor, factors)
    }
}

const BUCKET_WIDTH: usize = 256;
const BUCKET_SIZE: usize = 32;
// Overflows happen in less than 1/10000th of buckets
// but MPQS params use huge intervals.
const MAX_OVERFLOWS: usize = 32;

// A SieveTable holds sieve offsets for a size class of primes
// from the factor base.
// A size class is an interval [2^k, 2^(k+1)] for k in 15..24
#[derive(Clone)]
pub struct SieveTable {
    entries: Vec<(u8, u8)>,
    blens: Vec<u8>,
    overflows: [(u16, u8); 16],
    n_overflows: usize,
}

impl SieveTable {
    const N_ENTRIES: usize = BLOCK_SIZE / BUCKET_WIDTH * BUCKET_SIZE;
    const N_BUCKETS: usize = BLOCK_SIZE / BUCKET_WIDTH;

    fn new(nblocks: usize) -> Self {
        SieveTable {
            entries: vec![(0u8, 0u8); Self::N_ENTRIES * nblocks],
            blens: vec![0u8; Self::N_BUCKETS * nblocks],
            overflows: [(0u16, 0u8); 16],
            n_overflows: 0,
        }
    }

    fn reset(&mut self) {
        // no need to reset entries, overflows
        // as long as the counters are reset
        self.blens.fill(0u8);
        self.n_overflows = 0;
    }

    fn bucket(off: usize) -> (usize, usize) {
        (off / BUCKET_WIDTH, off % BUCKET_WIDTH)
    }

    #[inline]
    fn add(&mut self, offset: usize, pidx: u32) {
        debug_assert!(pidx < 256);
        debug_assert!(offset < self.entries.len() * BLOCK_SIZE / Self::N_ENTRIES);
        let (b, bucket_off) = Self::bucket(offset);
        unsafe {
            let blen_p = self.blens.get_unchecked_mut(b);
            let mut blen = *blen_p;
            if blen < BUCKET_SIZE as u8 {
                *self
                    .entries
                    .get_unchecked_mut(b * BUCKET_SIZE + blen as usize) =
                    (bucket_off as u8, pidx as u8);
                blen += 1;
                *blen_p = blen;
            } else {
                self.add_overflow(offset, pidx);
            }
        }
    }

    #[inline(never)]
    fn add_overflow(&mut self, offset: usize, pidx: u32) {
        if self.n_overflows < self.overflows.len() {
            self.overflows[self.n_overflows] = ((offset % BLOCK_SIZE) as u16, pidx as u8);
        }
        self.n_overflows += 1;
        if self.n_overflows >= MAX_OVERFLOWS {
            // Extremely unlikely to ever happen.
            eprintln!("WARNING: max overflows reached {}", self.n_overflows);
        }
    }
}

#[test]
fn test_sieve_block() {
    // Unit test based on quadratic sieve for a 128-bit integer.
    // sage: n = 176056248311966088405511077755578022771
    // sage: fb = [p for p in primes(50000) if Zmod(p)(n).is_square()]
    // sage: len(fb)
    // sage: r = isqrt(n)
    // sage: def cofactor(x):
    // ....:     for p in fb:
    // ....:         while True:
    // ....:             q, r = x.quo_rem(p)
    // ....:             if r == 0:
    // ....:                 x = q
    // ....:             else:
    // ....:                 break
    // ....:     return x
    // sage: cofs = [cofactor((r+i)**2 - n) for i in range(32768)]
    // sage: [i for i, x in enumerate(cofs) if x == 1]
    // [314, 957, 1779, 2587, 5882, 7121, 13468, 16323, 22144, 23176, 32407]
    // sage: [((r+i)**2-n).bit_length() for i, x in enumerate(cofs) if x == 1]
    // [73, 75, 76, 76, 78, 78, 79, 79, 79, 80, 80]
    use crate::fbase;
    use crate::qsieve;
    use crate::Uint;
    use std::str::FromStr;

    let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
    let fb = fbase::FBase::new(n, 2566);
    let nsqrt = crate::arith::isqrt(n);
    let qs = qsieve::SieveQS::new(n, &fb, 1 << 30, false);
    let mut s = qs.init_sieve_for_test();
    s.sieve_block();
    let expect: &[u16] = &[
        314, 957, 1779, 2587, 5882, 7121, 13468, 16323, 22144, 23176, 32407,
    ];
    let (idxs, facss) = s.smooths(70, None);
    eprintln!("sieve > 70 {:?}", idxs);
    let mut res = vec![];
    for (i, facs) in idxs.into_iter().zip(facss) {
        let ii = Uint::from(i as u64);
        let x = Int::from_bits((nsqrt + ii) * (nsqrt + ii) - n);
        let (cof, _) = s.cofactor(i as usize, &facs[..], &x);
        if cof == 1 {
            res.push(i);
        }
    }
    eprintln!("smooth {:?}", res);
    assert_eq!(res, expect);
}
