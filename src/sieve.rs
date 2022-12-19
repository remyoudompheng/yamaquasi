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
//! We assume that primes fit in 24 bits: for a bucket of size B,
//! the expected number of sieve hits is sum(B/p for large p)
//! and the variance is almost equal when p is large,
//! as a sum of Bernoulli variables with parameter B/p.
//!
//! sum(1/p) = 0.064 for p in primes(32768..1<<16) (3030 primes)
//!
//! An array of size 64 can hold hits over a width 512 interval
//! at avg + 5σ (0.064 * 512 + 5 * sqrt(0.064 * 512) ~= 62)
//!
//! sum(1/p) = 0.144 for p in primes(32768..32768*5) (11487 primes)
//!
//! An array of size 64 can hold hits over a width 256 interval
//! at avg + 4.5σ (0.144 * 256 + 4.5 * sqrt(0.144 * 256) ~= 64)
//!
//! sum(1/p) = 0.287 for p in primes(32768..1<<20) (78k primes)
//!
//! An array of size 64 can hold hits over a width 128 interval
//! at avg + 4.5σ (0.287 * 128 + 4.5 * sqrt(0.287 * 128) ~= 64)
//!
//! sum(1/p) = 0.47 for p in primes(32768..1<<24) (1M primes)
//!
//! An array of size 64 can hold hits over a width 64 interval
//! at avg + 6σ (0.47 * 64 + 6 * sqrt(0.47 * 64) ~= 63)
//!
//! It is fine to ignore all bucket overflows even if a few relations disappear.
//! The size of buckets is critical to keep good instructions/cycle.
//!
//! TODO: Bibliography

use std::cmp::max;
use wide;

use crate::arith::Num;
use crate::fbase::FBase;
use crate::Int;

pub const BLOCK_SIZE: usize = 32 * 1024;
const BUCKETSIZE: usize = 64;
const OFFSET_NONE: u16 = 0xffff;

// The sieve makes an array of offsets progress through the
// sieve interval. All offsets are assumed to fit in a u16.
#[derive(Clone)]
pub struct Sieve<'a> {
    pub offset: i64,
    pub blk_no: usize,
    // First offset of prime actually used in sieve
    pub idxskip: usize,
    // idx_by_log[i] is the index of first primes[idx]
    // such that bit_length(p) == i
    pub idx_by_log: [u32; 16],
    // Unique primes
    pub fbase: &'a FBase,
    // Small primes
    pub nsmalls: usize,    // number of small primes
    pub lo: Vec<u16>,      // cursor offsets
    pub lo_prev: Vec<u16>, // clone of lo before sieving block.
    // Result of sieve.
    pub blk: [u8; BLOCK_SIZE],
    // Cache for large prime hits
    // In each bucket, store the idx of an element of primes
    // A zero value means an empty slot.
    // Elements are bitfields defined by large_entry()
    // containing (block offset, log(p), pidx)
    pub largehits: Vec<Vec<u32>>,
    pub largeoffs: Vec<Vec<u8>>,
    pub blogsize: usize,
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
    pub fn new<F>(offset: i64, nblocks: usize, fbase: &'a FBase, f: F) -> Self
    where
        F: Fn(usize) -> SievePrime,
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
        let n_small = fbase
            .primes
            .iter()
            .position(|&p| p as usize > BLOCK_SIZE)
            .unwrap_or(fbase.len());
        let mut offs = Vec::with_capacity(2 * n_small);
        let mut log = 0;
        let mut idx_by_log = [0; 16];
        let maxprime = fbase.bound();
        let blogsize = match maxprime {
            0..=0x10000 => 9,        // 512
            0x10001..=0x28000 => 8,  // 256
            0x28001..=0x100000 => 7, // 128
            _ => 6,
        };
        let nbuckets = BLOCK_SIZE >> blogsize;
        let (mut largehits, mut largeoffs) = if maxprime as usize > BLOCK_SIZE {
            // No need to allocate large maps.
            (
                vec![vec![0u32; BUCKETSIZE * nbuckets]; nblocks],
                vec![vec![0u8; nbuckets]; nblocks],
            )
        } else {
            (vec![], vec![])
        };
        let mut overflows = 0;
        let mut nsmalls = fbase.len();
        for idx in 0..fbase.len() {
            let SievePrime {
                p,
                offsets: [o1, o2],
            } = f(idx);
            let l = 32 - u32::leading_zeros(p as u32) as usize;
            if l >= log {
                // Register new prime size
                while log <= l && log < 16 {
                    idx_by_log[log] = offs.len() as u32;
                    log += 1;
                }
            }
            if l <= 15 {
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
            } else {
                // Register number of small primes
                if nsmalls == fbase.len() {
                    nsmalls = idx
                }
                // Large prime: register them in hashmap
                for o in [o1, o2] {
                    let Some(o) = o else { continue };
                    let mut off = o as usize;
                    loop {
                        let blk_no = off / BLOCK_SIZE;
                        if blk_no >= nblocks {
                            break;
                        }
                        let blk_off = off % BLOCK_SIZE;
                        let bucket_off = blk_off % (1 << blogsize);
                        // Insert in large table
                        let b = blk_off >> blogsize;
                        unsafe {
                            let blen_p = largeoffs.get_unchecked_mut(blk_no).get_unchecked_mut(b);
                            let blen = *blen_p;
                            if blen < BUCKETSIZE as u8 {
                                // std::mem::replace compiles to a single 64-bit mov
                                *largehits
                                    .get_unchecked_mut(blk_no)
                                    .get_unchecked_mut(b * BUCKETSIZE + blen as usize) =
                                    Self::to_large_entry(bucket_off as u8, l as u8, idx as u32);
                                *blen_p += 1;
                                if *blen_p as usize + 1 == BUCKETSIZE {
                                    overflows += 1;
                                }
                            }
                        }
                        off += p as usize;
                    }
                }
            }
        }
        assert!(overflows < 10, "large bucket overflow!");
        // Fill remainder of idx_by_log
        for l in log..16 {
            idx_by_log[l] = offs.len() as u32;
        }
        let idxskip = 2 * fbase
            .primes
            .iter()
            .position(|&p| p > pskip)
            .unwrap_or(fbase.len());
        let len = offs.len();
        Sieve {
            offset,
            blk_no: 0,
            idxskip,
            idx_by_log,
            fbase,
            nsmalls,
            lo: offs,
            lo_prev: vec![0u16; len],
            blk: [0u8; BLOCK_SIZE],
            largehits,
            largeoffs,
            blogsize,
        }
    }

    #[inline]
    fn to_large_entry(bucket_off: u8, logp: u8, pidx: u32) -> u32 {
        // Encode as:
        // logp-16 => 3 bits
        // bucket_off => 9 bits
        // pidx => 20 bits
        ((logp as u32 - 16) << 29) | (bucket_off as u32) << 20 | pidx
    }

    #[inline]
    fn from_large_entry(x: u32) -> (u32, u8, u32) {
        (
            (x >> 20) as u32 & 511, // bucket_off
            16 + (x >> 29) as u8,   // logp
            x & ((1 << 20) - 1),
        )
    }

    // Recompute largehits/largeoffs for next blocks.
    // This is onyl for classic quadratic sieve where a single
    // polynomial is sieved over a huge interval.
    pub fn rehash<F>(&mut self, f: F)
    where
        F: Fn(usize) -> SievePrime,
    {
        self.blk_no = 0;
        let nblocks = self.largeoffs.len();
        if nblocks == 0 {
            return;
        }
        for i in 0..nblocks {
            self.largeoffs[i].fill(0);
            self.largehits[i].fill(0u32);
        }
        let mut overflows = 0;
        for (idx, &p) in self.fbase.primes.iter().enumerate() {
            if (p as usize) < BLOCK_SIZE {
                continue; // Small prime
            }
            let l = 32 - u32::leading_zeros(p) as usize;
            for o in f(idx).offsets {
                let Some(o) = o else { continue };
                let mut off = o as usize;
                loop {
                    let blk_no = off / BLOCK_SIZE;
                    if blk_no >= nblocks {
                        break;
                    }
                    let blk_off = off % BLOCK_SIZE;
                    let bucket_off = blk_off % (1 << self.blogsize);
                    // Insert in large table
                    let b = self.bucket(blk_off as u16);
                    let blen = self.largeoffs[blk_no][b];
                    if blen < BUCKETSIZE as u8 {
                        self.largehits[blk_no][b * BUCKETSIZE + blen as usize] =
                            Self::to_large_entry(bucket_off as u8, l as u8, idx as u32);
                        self.largeoffs[blk_no][b] = blen + 1;
                        if blen as usize + 1 == BUCKETSIZE {
                            overflows += 1;
                        }
                    }
                    off += p as usize;
                }
            }
        }
        assert!(overflows < 10, "large bucket overflow!");
    }

    pub fn sieve_block(&mut self) {
        assert_eq!(self.nsmalls * 2, self.lo.len());
        assert_eq!(self.nsmalls * 2, self.lo_prev.len());

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
            // They are guaranteed to have offsets inside the block.
            for log in 2..=15 {
                // Interval of primes such that bit length == log.
                let i_start = max(self.idxskip, self.idx_by_log[log] as usize);
                let i_end = if log < 15 {
                    self.idx_by_log[log + 1] as usize
                } else {
                    lo.len()
                };
                for i in i_start..i_end {
                    let i = i as usize;
                    let pidx = i / 2;
                    let p = *primes.get_unchecked(pidx);
                    let mut off: usize = *lo_prev.get_unchecked(i) as usize;
                    if off == OFFSET_NONE as usize {
                        continue;
                    }
                    let log = log as u8;
                    if p < 1024 {
                        let ll = len - 4 * p as usize;
                        while off < ll {
                            *blk.get_unchecked_mut(off) += log;
                            off += p as usize;
                            *blk.get_unchecked_mut(off) += log;
                            off += p as usize;
                            *blk.get_unchecked_mut(off) += log;
                            off += p as usize;
                            *blk.get_unchecked_mut(off) += log;
                            off += p as usize;
                        }
                    }
                    while off < len {
                        *blk.get_unchecked_mut(off) += log;
                        off += p as usize;
                    }
                    // Update state.
                    *lo.get_unchecked_mut(i) = (off % BLOCK_SIZE) as u16;
                }
            }
        }
        // Handle large primes
        if self.largehits.len() == 0 {
            return;
        }
        let largeidx = &self.largehits[self.blk_no];
        let nbuckets = self.n_buckets();
        let blk = &mut self.blk;
        for i in 0..nbuckets {
            unsafe {
                let bucket = largeidx.get_unchecked(i * BUCKETSIZE..(i + 1) * BUCKETSIZE);
                for j in 0..BUCKETSIZE {
                    let x: u32 = *bucket.get_unchecked(j);
                    if x == 0 {
                        break;
                    }
                    let (bucket_off, logp, _) = Self::from_large_entry(x);
                    let off = (i << self.blogsize) + bucket_off as usize;
                    *blk.get_unchecked_mut(off) += logp as u8;
                }
            }
        }
    }

    #[inline]
    fn bucket(&self, i: u16) -> usize {
        i as usize >> self.blogsize
    }

    #[inline]
    fn n_buckets(&self) -> usize {
        BLOCK_SIZE >> self.blogsize
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
            let log = 32 - u32::leading_zeros(p as u32) as usize;
            skipped += log as usize;
        }
        skipped
    }

    // Returns a list of block offsets and factors (as indices into the factor base).
    pub fn smooths(&self, threshold: u8) -> (Vec<u16>, Vec<Vec<usize>>) {
        assert_eq!(self.lo_prev.len(), 2 * self.nsmalls);
        // As smallest primes have been skipped, values in self.blk
        // are smaller than they should be: subtract an upper bound for
        // the missing part from the threshold.
        let skipbits = self.skipbits();
        let threshold2 = threshold - std::cmp::min(skipbits, threshold as usize / 2) as u8;

        let mut res: Vec<u16> = vec![];
        let thr16x = wide::u8x16::splat(threshold2 - 1);
        let mut i = 0;
        while i < BLOCK_SIZE {
            unsafe {
                // Cast as [u8;16] to avoid assuming alignment.
                let blk16 = (&self.blk[i] as *const u8) as *const [u8; 16];
                let blk16w = wide::u8x16::new(*blk16);
                if thr16x != blk16w.max(thr16x) {
                    // Some element is > threshold-1
                    for j in 0..16 {
                        let mut t = (*blk16)[j];
                        let ij = (i + j) as u16;
                        // Now add missing log(p) for smallest primes.
                        for i in 0..self.idxskip {
                            let off = self.lo_prev[i];
                            if off == OFFSET_NONE {
                                continue;
                            }
                            let pp = self.fbase.p(i / 2);
                            let pdiv = self.fbase.div(i / 2);
                            if pdiv.modu16(ij) == off {
                                let log = 32 - u32::leading_zeros(pp) as u8;
                                t += log;
                            }
                        }
                        if t >= threshold {
                            res.push(ij as u16)
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
        for i in 0..self.idx_by_log[15] {
            // Prime less than BLOCK_SIZE/2
            let i = i as usize;
            let pidx = i / 2;
            let off = self.lo_prev[i];
            if off == OFFSET_NONE {
                continue;
            }
            let pdiv = &self.fbase.divs[pidx];
            for idx in 0..rlen {
                let r = res[idx];
                if pdiv.modu16(r) == off {
                    facs[idx].push(pidx)
                }
            }
        }
        // Prime more than BLOCK_SIZE/2
        for i in self.idx_by_log[15]..self.lo_prev.len() as u32 {
            let i = i as usize;
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
        if self.largehits.len() == 0 {
            return (res, facs);
        }
        let largeidx = &self.largehits[self.blk_no];
        for j in 0..rlen {
            let r = res[j];
            let b = self.bucket(r);
            let boff = r & ((1 << self.blogsize) - 1);
            for &x in &largeidx[b * BUCKETSIZE..(b + 1) * BUCKETSIZE] {
                if x == 0 {
                    break;
                }
                let (off, _, pidx) = Self::from_large_entry(x);
                if boff as u32 == off {
                    facs[j].push(pidx as usize);
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
    let mut s = qs.init(None).0;
    s.sieve_block();
    let expect: &[u16] = &[
        314, 957, 1779, 2587, 5882, 7121, 13468, 16323, 22144, 23176, 32407,
    ];
    let (idxs, facss) = s.smooths(70);
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
