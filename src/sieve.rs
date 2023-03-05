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
//! most operations are caused by small primes, especially
//! for a properly optimized multiplier.
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
//! # Large primes
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
//! # Very large primes
//!
//! Very large primes (above 512k, size class 20) appear for factor
//! base sizes above ~20k. In that case, it is not so important to quickly
//! find prime factors (because there are very few smooth numbers).
//!
//! Using the same memory footprint, we can use large buckets (size 16384),
//! so that the hit count is at most 840 on average (less than 1024 up to 6
//! standard deviations). We can store a 16-bit offset and a 16-bit prime
//! offset using 4096 bytes.
//!
//! By applying this strategy for size classes >= 20, the number of sieve reports
//! should be very low and the impact of searching linearly through the large
//! buckets is less important.
//!
//! TODO: Bibliography

use std::cmp::{max, min};
use wide;

use crate::fbase::FBase;

pub const BLOCK_SIZE: usize = 32 * 1024;
const OFFSET_NONE: u16 = 0xffff;

const LARGE_PRIME_LOG: usize = 16;
const VERY_LARGE_PRIME_LOG: usize = 19;

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
    // Small primes
    pub lo: Vec<u16>,      // cursor offsets
    pub lo_prev: Vec<u16>, // clone of lo before sieving block.
    // Result of sieve.
    pub blk: [u8; BLOCK_SIZE],
    // Cache for large prime hits
    pub tables: Vec<SieveTable>,
    pub ltables: Vec<SieveTableLarge>,
}

/// A dummy structure used to recycle allocated resources between sieves.
pub struct SieveRecycle {
    tables: Vec<SieveTable>,
    ltables: Vec<SieveTableLarge>,
}

impl<'a> Sieve<'a> {
    // Initialize sieve starting at offset = -M
    // Function f determines starting offsets for a given prime
    // and returns log(p). This allows reading log(p) from an
    // alterante memory location when relevant.
    pub fn new(
        offset: i64,
        nblocks: usize,
        fbase: &'a FBase,
        roots: [&[u32]; 2],
        recycled: Option<SieveRecycle>,
    ) -> Self {
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
        let maxlog = 32 - u32::leading_zeros(maxprime) as usize;
        let (mut tables, mut ltables) = if let Some(rec) = recycled {
            // Use recycled memory: it should originate from a sieve
            // with the same factor base and number of blocks.
            let mut ts = rec.tables;
            let expected_len =
                (maxlog + 1).clamp(LARGE_PRIME_LOG, VERY_LARGE_PRIME_LOG) - LARGE_PRIME_LOG;
            assert_eq!(ts.len(), expected_len);
            for t in &mut ts {
                assert_eq!(t.entries.len(), SieveTable::N_ENTRIES * nblocks);
                t.reset();
            }
            let mut lts = rec.ltables;
            let expected_len = max(maxlog + 1, VERY_LARGE_PRIME_LOG) - VERY_LARGE_PRIME_LOG;
            assert_eq!(lts.len(), expected_len);
            for t in &mut lts {
                t.reset();
            }
            (ts, lts)
        } else {
            (
                (LARGE_PRIME_LOG..=min(VERY_LARGE_PRIME_LOG - 1, maxlog))
                    .map(|_| SieveTable::new(nblocks))
                    .collect(),
                (VERY_LARGE_PRIME_LOG..=maxlog)
                    .map(|_| SieveTableLarge::new(nblocks))
                    .collect(),
            )
        };

        fbase.len();
        let interval_size = (nblocks * BLOCK_SIZE) as isize;
        let [roots1, roots2] = roots;
        for log in 0..=maxlog {
            let idx1 = fbase.idx_by_log[log];
            let idx2 = fbase.idx_by_log[log + 1];
            let primes = &fbase.primes;
            assert!(idx2 <= roots1.len() && idx2 <= roots2.len() && idx2 <= primes.len());
            if log < LARGE_PRIME_LOG {
                // First size class: primes smaller than block size.
                for idx in idx1..idx2 {
                    let (o1, o2) = (roots1[idx], roots2[idx]);
                    offs.push(o1 as u16);
                    offs.push(if o1 != o2 { o2 as u16 } else { OFFSET_NONE });
                    debug_assert!(offs.len() == 2 * idx + 2);
                }
            } else if log < VERY_LARGE_PRIME_LOG {
                // Second class: primes larger than block size but (roughly)
                // smaller than interval size.
                let table = &mut tables[log - LARGE_PRIME_LOG];
                for pidx in idx1..idx2 {
                    // Large prime: register them in hashmap
                    // They cannot be factors of A in SIQS, so they must have 2 roots.
                    let (o1, o2, p) = unsafe {
                        (
                            *roots1.get_unchecked(pidx) as isize,
                            *roots2.get_unchecked(pidx) as isize,
                            *primes.get_unchecked(pidx) as isize,
                        )
                    };
                    debug_assert!(
                        o1 != o2,
                        "large primes must have 2 roots p={p} o1={o1} o2={o2}"
                    );
                    let pidx = pidx as u32;
                    let mut kp: isize = 0;
                    let rmax = max(o1, o2);
                    let m = interval_size - p - rmax;
                    while kp < m {
                        table.add((kp + o1) as usize, pidx);
                        table.add((kp + o2) as usize, pidx);
                        table.add((kp + p + o1) as usize, pidx);
                        table.add((kp + p + o2) as usize, pidx);
                        kp += 2 * p;
                    }
                    let mut off = o1 + kp;
                    while off < interval_size {
                        table.add(off as usize, pidx);
                        off += p;
                    }
                    let mut off = o2 + kp;
                    while off < interval_size {
                        table.add(off as usize, pidx);
                        off += p;
                    }
                }
            } else {
                // Very large primes (above 500k).
                // We use large hash buckets that need fewer active cache
                // lines. Many primes will not even generate a hit.
                let table = &mut ltables[log - VERY_LARGE_PRIME_LOG];
                for pidx in idx1..idx2 {
                    // Large prime: register them in hashmap
                    let (mut o1, mut o2, p) = unsafe {
                        (
                            *roots1.get_unchecked(pidx) as isize,
                            *roots2.get_unchecked(pidx) as isize,
                            *primes.get_unchecked(pidx) as isize,
                        )
                    };
                    debug_assert!(
                        o1 != o2,
                        "large primes must have 2 roots p={p} o1={o1} o2={o2}"
                    );
                    while (o1 as isize) < interval_size {
                        table.add(o1 as usize, pidx);
                        o1 += p;
                    }
                    while (o2 as isize) < interval_size {
                        table.add(o2 as usize, pidx);
                        o2 += p;
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
            lo: offs,
            lo_prev: vec![0u16; len],
            blk: [0u8; BLOCK_SIZE],
            tables,
            ltables,
        }
    }

    // Recompute largehits/largeoffs for next blocks.
    // This is only for classic quadratic sieve where a single
    // polynomial is sieved over a huge interval.
    pub fn rehash(&mut self, roots: [&[u32]; 2]) {
        self.blk_no = 0;
        if self.nblocks == 0 {
            return;
        }
        for t in &mut self.tables {
            t.reset();
        }
        for t in &mut self.ltables {
            t.reset();
        }
        let [roots1, roots2] = roots;
        for (pidx, &p) in self.fbase.primes.iter().enumerate() {
            if (p as usize) < BLOCK_SIZE {
                continue; // Small prime
            }
            let l = 32 - u32::leading_zeros(p) as usize;
            if l < VERY_LARGE_PRIME_LOG {
                let table = &mut self.tables[l - LARGE_PRIME_LOG];
                let (o1, o2) = (roots1[pidx], roots2[pidx]);
                for o in [o1, o2] {
                    let mut off = o as usize;
                    loop {
                        if off >= self.nblocks * BLOCK_SIZE {
                            break;
                        }
                        table.add(off, pidx as u32);
                        off += p as usize;
                    }
                }
            } else {
                let ltable = &mut self.ltables[l - VERY_LARGE_PRIME_LOG];
                let (o1, o2) = (roots1[pidx], roots2[pidx]);
                for o in [o1, o2] {
                    let mut off = o as usize;
                    loop {
                        if off >= self.nblocks * BLOCK_SIZE {
                            break;
                        }
                        ltable.add(off, pidx);
                        off += p as usize;
                    }
                }
            }
        }
    }

    /// Consume the sieve and recycle allocated hash tables.
    pub fn recycle(self) -> SieveRecycle {
        SieveRecycle {
            tables: self.tables,
            ltables: self.ltables,
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
                    2 * self.fbase.idx_by_log[log + 1]
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
                        let mut kp = 0;
                        let ll = len - p - m;
                        while kp < ll {
                            // both kp+off1 and kp+off2 are in range
                            *blk.get_unchecked_mut(kp + off1) += log;
                            *blk.get_unchecked_mut(kp + off2) += log;
                            *blk.get_unchecked_mut(kp + p + off1) += log;
                            *blk.get_unchecked_mut(kp + p + off2) += log;
                            kp += 2 * p;
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
                let i_start = max(self.idxskip, 2 * self.fbase.idx_by_log[log]);
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
                    for &entry in t.get_unchecked(bidx * BUCKET_SIZE..bidx * BUCKET_SIZE + blen) {
                        let (boff, _) = std::mem::transmute::<u16, (u8, u8)>(entry);
                        let off = base_off + boff as usize;
                        *blk.get_unchecked_mut(off) += logp;
                    }
                }
            }
        }
        // Handle very large primes: each 32k block is 2 buckets.
        // First half must be handled completely before the second half.
        for bucket in 0..SieveTableLarge::BUCKETS_PER_BLOCK {
            let bno = self.blk_no * SieveTableLarge::BUCKETS_PER_BLOCK + bucket;
            for (tidx, t) in self.ltables.iter().enumerate() {
                let logp = (VERY_LARGE_PRIME_LOG + tidx) as u8;
                unsafe {
                    for &entry in t.bucket_offsets(bno) {
                        let (boff, _) = std::mem::transmute::<u32, (u16, u16)>(entry);
                        *blk.get_unchecked_mut(boff as usize) += logp;
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

    // Returns a list of block offsets and factors (as indices into the factor base).
    // The threshold is given for the "ends" of the interval.
    //
    pub fn smooths(
        &self,
        threshold: u8,
        root: Option<u32>,
        polyroots: [&[u32]; 2],
    ) -> (Vec<u16>, Vec<Vec<usize>>) {
        let [roots1, roots2] = polyroots;
        let is_factor = |offset: usize, pidx: usize| -> bool {
            let div = self.fbase.div(pidx);
            let blkbase = (self.blk_no as u32).checked_mul(BLOCK_SIZE as u32).unwrap();
            let o = div.modi64(blkbase as i64 + offset as i64) as u32;
            o == roots1[pidx] || o == roots2[pidx]
        };

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
                            let x = (ij as i32) + (self.blk_no * BLOCK_SIZE) as i32
                                - (self.nblocks * BLOCK_SIZE / 2) as i32;
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
                // Bounds of this size class.
                let idx1 = self.fbase.idx_by_log[logp];
                let idx2 = self.fbase.idx_by_log[logp + 1];
                let blen = t.blens[b] as usize;
                for eidx in b * BUCKET_SIZE..b * BUCKET_SIZE + blen {
                    let (off, pidx8) = unsafe { t.unchecked_entry(eidx) };
                    if boff as u8 == off {
                        // Walk candidate primes
                        for msb in idx1 >> 8..(idx2 >> 8) + 1 {
                            let pidx = (msb << 8) + pidx8 as usize;
                            if idx1 <= pidx && pidx < idx2 {
                                // We may exceptionally have duplicates:
                                // that's fine.
                                if is_factor(r as usize, pidx) {
                                    facs[j].push(pidx)
                                }
                            }
                        }
                    }
                }
                // Look for offset in overflows (extremely uncommon)
                for &(ooff, pidx8) in &t.overflows[..min(t.overflows.len(), t.n_overflows)] {
                    if ooff == r {
                        // Walk candidate primes
                        for msb in idx1 >> 8..(idx2 >> 8) + 1 {
                            let pidx = (msb << 8) + pidx8 as usize;
                            if idx1 <= pidx && pidx < idx2 {
                                // We may exceptionally have duplicates:
                                // that's fine.
                                if is_factor(r as usize, pidx) {
                                    facs[j].push(pidx)
                                }
                            }
                        }
                    }
                }
            }
            // Look for very large primes
            for t in &self.ltables {
                let entries = t.bucket_offsets(
                    SieveTableLarge::BUCKETS_PER_BLOCK * self.blk_no
                        + r as usize / SieveTableLarge::LBUCKET_WIDTH,
                );
                let overflows = &t.overflows;
                for bucket in [entries, overflows] {
                    for entry in bucket {
                        let (boff, pidx16) =
                            unsafe { std::mem::transmute::<u32, (u16, u16)>(*entry) };
                        if boff == r {
                            let mut pidx = pidx16 as usize;
                            while pidx < self.fbase.len() {
                                if is_factor(r as usize, pidx) {
                                    facs[j].push(pidx)
                                }
                                pidx += 1 << 16;
                            }
                        }
                    }
                }
            }
        }
        (res, facs)
    }
}

const BUCKET_WIDTH: usize = 256;
const BUCKET_SIZE: usize = 32;
// Overflows happen in less than 1/10000th of buckets
// but MPQS params use huge intervals.
const MAX_OVERFLOWS: usize = 64;

/// A SieveTable holds sieve offsets for a size class of primes
/// from the factor base.
/// A size class is an interval [2^k, 2^(k+1)] for k in 15..19
#[derive(Clone)]
pub struct SieveTable {
    // An entry is a transmuted tuple (u8,u8):
    // - a 8-bit offset, relative to a bucket of size BUCKET_WIDTH
    // - 8 lowest bits of a prime index in the factor base
    entries: Vec<u16>,
    // Contiguous array of bucket lengths.
    // This is meant to stay in cache while the table is filled.
    blens: Vec<u8>,
    overflows: [(u16, u8); 32],
    n_overflows: usize,
}

impl SieveTable {
    const N_ENTRIES: usize = BLOCK_SIZE / BUCKET_WIDTH * BUCKET_SIZE;
    const N_BUCKETS: usize = BLOCK_SIZE / BUCKET_WIDTH;

    fn new(nblocks: usize) -> Self {
        SieveTable {
            entries: vec![0; Self::N_ENTRIES * nblocks],
            blens: vec![0; Self::N_BUCKETS * nblocks],
            overflows: [(0u16, 0u8); 32],
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
    unsafe fn unchecked_entry(&self, idx: usize) -> (u8, u8) {
        std::mem::transmute(*self.entries.get_unchecked(idx))
    }

    #[inline]
    fn add(&mut self, offset: usize, pidx: u32) {
        debug_assert!(offset < self.entries.len() * BLOCK_SIZE / Self::N_ENTRIES);
        let (b, bucket_off) = Self::bucket(offset);
        unsafe {
            let blen_p = self.blens.get_unchecked_mut(b);
            let mut blen = *blen_p;
            if blen < BUCKET_SIZE as u8 {
                // Store only the 8 lowest bits of pidx.
                let entry = std::mem::transmute((bucket_off as u8, (pidx & 0xff) as u8));
                *self
                    .entries
                    .get_unchecked_mut(b * BUCKET_SIZE + blen as usize) = entry;
                blen += 1;
                *blen_p = blen;
            } else {
                self.add_overflow(offset, pidx);
            }
        }
    }

    #[cold]
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

/// A SieveTableLarge holds sieve offsets for a size class of primes
/// from the factor base.
///
/// They are stored in larger buckets (for intervals of size 16384 instead of 256).
#[derive(Clone)]
pub struct SieveTableLarge {
    // Sieve hits will be organized by blocks to bring mild locality.
    // Each element is a transmuted (u16,u16).
    hits: Vec<u32>,
    // Current length of each bucket.
    lengths: Vec<u32>,
    // Overflows: common to all buckets.
    overflows: Vec<u32>,
}

impl SieveTableLarge {
    // Size of interval slice covered by a bucket.
    const LBUCKET_WIDTH: usize = 16384;
    const BUCKETS_PER_BLOCK: usize = BLOCK_SIZE / Self::LBUCKET_WIDTH;
    // Size of storage for a given bucket:
    // since 1/16 is less than the expected hit density (0.0541 for size class 19)
    // this is at least 3.2 st deviations above the average and should
    // avoid getting too many overflows.
    const LBUCKET_SIZE: usize = Self::LBUCKET_WIDTH / 16;

    fn new(nblocks: usize) -> Self {
        let nbuckets = nblocks * BLOCK_SIZE / Self::LBUCKET_WIDTH;
        SieveTableLarge {
            hits: vec![0; (nbuckets + 1) * Self::LBUCKET_WIDTH],
            lengths: vec![0; nbuckets + 1],
            overflows: vec![],
        }
    }

    fn reset(&mut self) {
        // It is not necessary to write to self.hits,
        // it is a very large array so it is too costly.
        self.lengths.fill(0);
        self.overflows.clear();
    }

    /// Stores an offset and a truncated prime index.
    #[inline]
    fn add(&mut self, offset: usize, pidx: usize) {
        debug_assert!(pidx < 1 << 30);
        let (blk, blkoff) = (offset / Self::LBUCKET_WIDTH, offset % BLOCK_SIZE);
        unsafe {
            let entry = std::mem::transmute((blkoff as u16, pidx as u16));
            let l = *self.lengths.get_unchecked(blk) as usize;
            if l < Self::LBUCKET_SIZE {
                let idx = blk * Self::LBUCKET_SIZE + l;
                *self.hits.get_unchecked_mut(idx) = entry;
                *self.lengths.get_unchecked_mut(blk) = l as u32 + 1;
            } else {
                self.add_overflow(entry);
            }
        }
    }

    #[inline(never)]
    fn add_overflow(&mut self, entry: u32) {
        // This should only happen once for a billion interval.
        // It may happen a couple of times over a large sieve.
        self.overflows.push(entry)
    }

    fn bucket_offsets(&self, bidx: usize) -> &[u32] {
        let len = self.lengths[bidx] as usize;
        &self.hits[bidx * Self::LBUCKET_SIZE..bidx * Self::LBUCKET_SIZE + len as usize]
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
    use crate::arith::I256;
    use crate::fbase;
    use crate::qsieve;
    use crate::{Int, Uint};
    use bnum::cast::CastFrom;
    use std::str::FromStr;

    let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
    let fb = fbase::FBase::new(n, 2566);
    let nsqrt = crate::arith::isqrt(n);
    let qs = qsieve::SieveQS::new(n, &fb, 1 << 30, false);
    let (mut s, [r1s, r2s]) = qs.init_sieve_for_test();
    s.sieve_block();
    let expect: &[u16] = &[
        314, 957, 1779, 2587, 5882, 7121, 13468, 16323, 22144, 23176, 32407,
    ];
    let (idxs, facss) = s.smooths(70, None, [&r1s[..], &r2s[..]]);
    eprintln!("sieve > 70 {:?}", idxs);
    let mut res = vec![];
    for (i, facs) in idxs.into_iter().zip(facss) {
        let ii = Uint::from(i as u64);
        let x = Int::from_bits((nsqrt + ii) * (nsqrt + ii) - n);
        assert!(x.abs().bits() < 255);
        let Some(((p, q), _)) = fbase::cofactor(&fb, &I256::cast_from(x), &facs[..], 1_000_000, false)
            else { continue };
        if p == 1 && q == 1 {
            res.push(i);
        }
    }
    eprintln!("smooth {:?}", res);
    assert_eq!(res, expect);
}
