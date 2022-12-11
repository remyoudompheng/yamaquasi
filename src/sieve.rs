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
//! In constrast, large primes sieve hits are very sparse: they will
//! typically divide less than half of interval elements. We can use
//! loosely sorted buckets to quiclky find which primes divide a given
//! element. This is similar to a hashmap-based strategy in msieve.
//!
//! TODO: Bibliography

use crate::arith::Num;
use crate::fbase::Prime;
use crate::Int;
use wide;

pub const BLOCK_SIZE: usize = 32 * 1024;
const BUCKETS: usize = 512;
const BUCKETSIZE: usize = BLOCK_SIZE / BUCKETS;

// The sieve makes an array of offsets progress through the
// sieve interval.
// The offsets are relative to the block offset.
// They are stored as hi<<B + lo
// so that an offset if in the current block iff hi==0.
#[derive(Clone)]
pub struct Sieve<'a> {
    pub offset: i64,
    // First offset of prime actually used in sieve
    pub idxskip: usize,
    pub idx14: usize, // Offset of prime > 16384
    pub idx15: usize, // Offset of prime > 32768
    pub primes: Vec<&'a Prime>,
    pub logs: Vec<u8>,
    // The MSB of the offset for each cursor.
    pub hi: Vec<u8>,
    // The LSB of the offset for each cursor.
    pub lo: Vec<u16>,
    // Clone of lo before sieving block.
    pub starts: Vec<u16>,
    pub histarts: Vec<u8>,
    // Result of sieve.
    pub blk: [u8; BLOCK_SIZE],
    // Cache for large prime hits
    pub largehits: [u32; BLOCK_SIZE],
    pub largeoffs: [u8; BUCKETS],
}

impl<'a> Sieve<'a> {
    pub fn new(offset: i64, primes: Vec<&'a Prime>, hi: Vec<u8>, lo: Vec<u16>) -> Self {
        // Skip enough primes to skip ~half of operations
        let pskip = match primes.len() {
            0..=1999 => 3,
            2000..=4999 => 5,
            5000..=9999 => 7,
            10000..=19999 => 11,
            20000..=49999 => 13,
            _ => 17,
        };
        let idxskip = primes
            .iter()
            .position(|&p| p.p > pskip)
            .unwrap_or(primes.len());
        let idx14 = primes
            .iter()
            .position(|&p| p.p > BLOCK_SIZE as u64 / 2)
            .unwrap_or(primes.len());
        let idx15 = primes
            .iter()
            .position(|&p| p.p > BLOCK_SIZE as u64)
            .unwrap_or(primes.len());
        let logs = primes
            .iter()
            .map(|p| (32 - u32::leading_zeros(p.p as u32)) as u8)
            .collect();
        let len = primes.len();
        Sieve {
            offset,
            idxskip,
            idx14,
            idx15,
            primes,
            logs,
            hi,
            lo,
            starts: vec![0u16; len],
            histarts: vec![0u8; len],
            blk: [0u8; BLOCK_SIZE],
            largehits: [0u32; BLOCK_SIZE],
            largeoffs: [0u8; BUCKETS],
        }
    }

    #[inline]
    fn insert_large(&mut self, i: usize, idx: usize) {
        let b = i / BUCKETSIZE;
        let off = self.largeoffs[b];
        if off < BUCKETSIZE as u8 {
            self.largehits[b * BUCKETSIZE + off as usize] = idx as u32;
            self.largeoffs[b] = off + 1;
        }
    }

    pub fn sieve_block(&mut self) {
        let len: usize = BLOCK_SIZE;
        self.blk.fill(0u8);
        self.largeoffs.fill(0u8);
        let blk = &mut self.blk;
        self.starts.copy_from_slice(&self.lo);
        self.histarts.copy_from_slice(&self.hi);
        unsafe {
            // Smallest primes: we don't update self.blk at all,
            // and simply update offsets for next block.
            for i in 0..self.idxskip {
                let p = self.primes.get_unchecked(i);
                let pp = p.p as usize;
                // off -> off + kp - BLOCK_SIZE
                let mut off: usize = *self.lo.get_unchecked(i) as usize;
                off = off + pp - p.div.modu16(BLOCK_SIZE as u16) as usize;
                if off >= pp {
                    off -= pp;
                }
                self.lo[i] = off as u16;
            }
            // Small primes: perform ordinary sieving.
            // They are guaranteed to have offsets inside the block.
            for i in self.idxskip..self.idx15 {
                let i = i as usize;
                let p = self.primes.get_unchecked(i).p;
                debug_assert!(self.hi[i] == 0);
                let mut off: usize = *self.lo.get_unchecked(i) as usize;
                let size = *self.logs.get_unchecked(i);
                if p < 1024 {
                    let ll = len - 4 * p as usize;
                    while off < ll {
                        *blk.get_unchecked_mut(off) += size;
                        off += p as usize;
                        *blk.get_unchecked_mut(off) += size;
                        off += p as usize;
                        *blk.get_unchecked_mut(off) += size;
                        off += p as usize;
                        *blk.get_unchecked_mut(off) += size;
                        off += p as usize;
                    }
                }
                while off < len {
                    *blk.get_unchecked_mut(off) += size;
                    off += p as usize;
                }
                // Update state. No need to set hi=1.
                self.lo[i] = (off % BLOCK_SIZE) as u16;
            }
        }
        // Large primes can have at most 1 hit.
        for i in self.idx15..self.primes.len() {
            if self.hi[i] != 0 {
                continue;
            }
            let i = i as usize;
            let p = self.primes[i].p;
            let lo = self.lo[i] as usize;
            self.blk[lo] += self.logs[i];
            self.insert_large(lo, i);
            let off = lo + p as usize;
            debug_assert!(off > BLOCK_SIZE);
            self.hi[i] = (off / BLOCK_SIZE) as u8;
            self.lo[i] = (off % BLOCK_SIZE) as u16;
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
        // Decrement MSB by 1.
        let mut idx: usize = 0;
        while idx + 16 < self.hi.len() {
            unsafe {
                let p = (&mut self.hi[idx]) as *mut u8 as *mut wide::u8x16;
                *p = (*p).min(*p - 1);
            }
            idx += 16;
        }
        while idx < self.hi.len() {
            if self.hi[idx] > 0 {
                self.hi[idx] -= 1;
            }
            idx += 1
        }
    }

    fn skipbits(&self) -> usize {
        // Extra amount to subtract from threshold if we skipped sieving
        // small primes.
        // Beware that primes typically appear twice.
        let mut skipped = 0usize;
        let mut prev = 0;
        for i in 0..self.idxskip {
            let p = self.primes[i].p;
            if p == prev {
                continue;
            }
            prev = p;
            skipped += self.logs[i] as usize;
        }
        skipped
    }

    pub fn smooths(&self, threshold: u8) -> (Vec<u16>, Vec<Vec<usize>>) {
        assert_eq!(self.starts.len(), self.primes.len());
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
                        for pidx in 0..self.idxskip {
                            let off = self.starts[pidx];
                            let p = self.primes[pidx];
                            if p.div.modu16(ij) == off {
                                t += self.logs[pidx];
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
        for i in 0..self.idx14 {
            // Prime less than BLOCK_SIZE/2
            let off = self.starts[i];
            let pdiv = &self.primes[i].div;
            for idx in 0..rlen {
                let r = res[idx];
                if pdiv.modu16(r) == off {
                    facs[idx].push(i)
                }
            }
        }
        // Prime more than BLOCK_SIZE/2
        for i in self.idx14..self.idx15 {
            let off = self.starts[i];
            let p = self.primes[i].p as u16;
            for idx in 0..rlen {
                let r = res[idx];
                if r == off || r == off + p {
                    facs[idx].push(i)
                }
            }
        }
        // Prime more than BLOCK_SIZE
        // Use reverse lookup table.
        for j in 0..rlen {
            let r = res[j];
            let b = r as usize / BUCKETSIZE;
            let blen = self.largeoffs[b] as usize;
            if blen < BUCKETSIZE {
                for idx in 0..blen {
                    let i = self.largehits[b * BUCKETSIZE + idx] as usize;
                    if self.histarts[i] == 0 && self.starts[i] == r {
                        facs[j].push(i);
                    }
                }
            } else {
                // too full, full scan everything
                for i in self.idx15..self.primes.len() {
                    if self.histarts[i] == 0 && self.starts[i] == r {
                        facs[j].push(i);
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
        for &fidx in facs {
            let item = self.primes[fidx];
            let mut exp = 0;
            loop {
                let (q, r) = item.div.divmod_uint(&cofactor);
                if r == 0 {
                    cofactor = q;
                    exp += 1;
                } else {
                    break;
                }
            }
            factors.push((item.p as i64, exp));
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
    let primes = fbase::primes(5133);
    let fb = fbase::prepare_factor_base(&n, &primes[..]);
    let nsqrt = crate::arith::isqrt(n);
    let mut s = qsieve::init_sieves(&fb, nsqrt).0;
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
