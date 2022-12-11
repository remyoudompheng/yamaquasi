//! Shared common implementation of sieve.

use crate::arith::Num;
use crate::fbase::Prime;
use crate::{Int, Uint};
use wide;

pub const BLOCK_SIZE: usize = 32 * 1024;

// The sieve makes an array of offsets progress through the
// sieve interval.
// The offsets are relative to the block offset.
// They are stored as hi<<B + lo
// so that an offset if in the current block iff hi==0.
#[derive(Clone)]
pub struct Sieve<'a> {
    pub offset: i64,
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
    // Result of sieve
    pub blk: [u8; BLOCK_SIZE],
}

impl<'a> Sieve<'a> {
    pub fn new(offset: i64, primes: Vec<&'a Prime>, hi: Vec<u8>, lo: Vec<u16>) -> Self {
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
            idx15,
            primes,
            logs,
            hi,
            lo,
            starts: vec![0u16; len],
            histarts: vec![0u8; len],
            blk: [0u8; BLOCK_SIZE],
        }
    }
    pub fn sieve_block(&mut self) {
        let len: usize = BLOCK_SIZE;
        self.blk.fill(0u8);
        let blk = &mut self.blk;
        self.starts.copy_from_slice(&self.lo);
        self.histarts.copy_from_slice(&self.hi);
        unsafe {
            for i in 0..self.idx15 {
                let i = i as usize;
                let p = self.primes.get_unchecked(i).p;
                // Small primes always have a hit.
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
        for i in self.idx15..self.primes.len() {
            // Large primes have at most 1 hit.
            if self.hi[i] != 0 {
                continue;
            }
            let i = i as usize;
            let p = self.primes[i].p;
            blk[self.lo[i] as usize] += self.logs[i];
            let off = self.lo[i] as usize + p as usize;
            debug_assert!(off > BLOCK_SIZE);
            self.hi[i] = (off / BLOCK_SIZE) as u8;
            self.lo[i] = (off % BLOCK_SIZE) as u16;
        }
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

    pub fn smooths(&self, threshold: u8) -> (Vec<usize>, Vec<Vec<usize>>) {
        assert_eq!(self.starts.len(), self.primes.len());
        let mut res = vec![];
        for (i, &v) in self.blk.iter().enumerate() {
            if v >= threshold {
                res.push(i)
            }
        }
        if res.len() == 0 {
            return (vec![], vec![]);
        }
        // Now find factors.
        let mut facs = vec![vec![]; res.len()];
        for i in 0..self.idx15 {
            let off = self.starts[i] as u64;
            let p = self.primes[i];
            for (idx, &r) in res.iter().enumerate() {
                if p.p < BLOCK_SIZE as u64 / 2 {
                    if p.div.modi64(r as i64) == off {
                        facs[idx].push(i)
                    }
                } else {
                    if r as u64 == off || r as u64 == off + p.p {
                        facs[idx].push(i)
                    }
                }
            }
        }
        for i in self.idx15..self.primes.len() {
            if self.histarts[i] != 0 {
                continue;
            }
            let idx = self.starts[i] as usize;
            if let Some(j) = res.iter().position(|&j| j == idx) {
                facs[j].push(i);
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
    let expect: &[usize] = &[
        314, 957, 1779, 2587, 5882, 7121, 13468, 16323, 22144, 23176, 32407,
    ];
    let (idxs, facss) = s.smooths(70);
    eprintln!("sieve > 70 {:?}", idxs);
    let mut res = vec![];
    for (i, facs) in idxs.into_iter().zip(facss) {
        let ii = Uint::from(i as u64);
        let x = Int::from_bits((nsqrt + ii) * (nsqrt + ii) - n);
        let (cof, _) = s.cofactor(i, &facs[..], &x);
        if cof == 1 {
            res.push(i);
        }
    }
    eprintln!("smooth {:?}", res);
    assert_eq!(res, expect);
}
