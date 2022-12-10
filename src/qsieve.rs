// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! The classical quadratic sieve (using polynomial (x + Nsqrt)^2 - N).
//!
//! Bibliography:
//! J. Gerver, Factoring Large Numbers with a Quadratic Sieve
//! https://www.jstor.org/stable/2007781

use std::collections::HashMap;

use crate::arith::{isqrt, Num};
use crate::fbase::Prime;
use crate::params::{large_prime_factor, BLOCK_SIZE};
use crate::relations::{combine_large_relation, relation_gap, Relation};
use crate::{Int, Uint};

pub fn qsieve(n: Uint, primes: &[Prime]) -> Vec<Relation> {
    // Prepare sieve
    let nsqrt = isqrt(n);
    let maxlarge: u64 = primes.last().unwrap().p * large_prime_factor(&n);
    eprintln!("Max cofactor {}", maxlarge);

    // Naïve quadratic sieve with polynomial x²-n (x=-M..M)
    // Max value is X = sqrt(n) * M
    // Smooth bound Y = exp(1/2 sqrt(log X log log X))
    // If input number is 512 bits, generated values are less then 300 bits

    // There are at most 10 prime factors => 8-bit precision is enough
    // Blocks optimized to fit 512kB cache memory per core
    // Work unit is 16 blocks

    // We run 2 sieves:
    // Forward: polynomial (R+x)^2 - N where r = isqrt(N)
    // => roots are sqrt(N) - R
    // Backward: polynomial (R-1-x)^2 - N where r = isqrt(N)
    // => roots are sqrt(N) + R-1

    let mut relations = vec![];
    let mut target = primes.len() * 8 / 10;
    let mut larges = HashMap::<u64, Relation>::new();
    let mut extras = 0;
    let sieve = SieveQS { n, nsqrt };
    // Construct 2 initial states, forward and backwards.
    let mut st_primes = vec![];
    let mut st_logs = vec![];
    let mut st_hi1 = vec![];
    let mut st_lo1 = vec![];
    let mut st_hi2 = vec![];
    let mut st_lo2 = vec![];
    for p in primes {
        assert_eq!(p.p >> 24, 0);
        let rp = p.div.mod_uint(&nsqrt);
        let logp = (32 - u32::leading_zeros(p.p as u32)) as u8;
        st_primes.push(p);
        st_logs.push(logp);
        // Forward sieve
        let s1 = p.div.divmod64(p.r + p.p - rp).1;
        let s2 = p.div.divmod64(p.p - p.r + p.p - rp).1;
        st_hi1.push((s1 / BLOCK_SIZE as u64) as u8);
        st_lo1.push((s1 % BLOCK_SIZE as u64) as u16);
        if p.r != 0 {
            // 2 roots
            st_primes.push(p);
            st_logs.push(logp);
            st_hi1.push((s2 / BLOCK_SIZE as u64) as u8);
            st_lo1.push((s2 % BLOCK_SIZE as u64) as u16);
        }
        let x1 = nsqrt + Uint::from(s1);
        let x2 = nsqrt + Uint::from(s2);
        assert_eq!(p.div.mod_uint(&(x1 * x1)), p.div.mod_uint(&n));
        assert_eq!(p.div.mod_uint(&(x2 * x2)), p.div.mod_uint(&n));
        // Backward sieve
        let rp = if rp == 0 { p.p - 1 } else { rp - 1 };
        let s1 = p.div.divmod64(p.r + rp).1;
        let s2 = p.div.divmod64(p.p - p.r + rp).1;
        st_hi2.push((s1 / BLOCK_SIZE as u64) as u8);
        st_lo2.push((s1 % BLOCK_SIZE as u64) as u16);
        if p.r != 0 {
            // 2 roots
            st_hi2.push((s2 / BLOCK_SIZE as u64) as u8);
            st_lo2.push((s2 % BLOCK_SIZE as u64) as u16);
        }
        let x1 = nsqrt - Uint::from(s1 + 1);
        let x2 = nsqrt - Uint::from(s2 + 1);
        assert_eq!(p.div.mod_uint(&(x1 * x1)), p.div.mod_uint(&n));
        assert_eq!(p.div.mod_uint(&(x2 * x2)), p.div.mod_uint(&n));
    }
    let idx15 = st_primes
        .iter()
        .position(|&p| p.p > BLOCK_SIZE as u64)
        .unwrap_or(st_primes.len());
    let mut st_fwd = StateQS {
        offset: 0,
        idx15,
        primes: &st_primes[..],
        logs: &st_logs,
        hi: st_hi1,
        lo: st_lo1,
    };
    let mut st_bck = StateQS {
        offset: 0,
        idx15,
        primes: &st_primes[..],
        logs: &st_logs,
        hi: st_hi2,
        lo: st_lo2,
    };
    loop {
        let (mut found, foundlarge) = sieve_block(&sieve, &mut st_fwd, false);
        if found.len() > primes.len() + 16 {
            // Too many relations! May happen for very small inputs.
            relations.extend_from_slice(&mut found[..primes.len() + 16]);
            let gap = relation_gap(&relations);
            if gap == 0 {
                println!("Found enough relations");
                break;
            } else {
                println!("Need {} additional relations", gap);
                target = relations.len() + gap + 10;
            }
        }
        st_fwd.next_block();
        relations.append(&mut found);
        for r in foundlarge {
            if let Some(rr) = combine_large_relation(&mut larges, &r, &n) {
                relations.push(rr);
                extras += 1;
            }
        }
        let (mut found, foundlarge) = sieve_block(&sieve, &mut st_bck, true);
        st_bck.next_block();
        relations.append(&mut found);
        for r in foundlarge {
            if let Some(rr) = combine_large_relation(&mut larges, &r, &n) {
                relations.push(rr);
                extras += 1;
            }
        }
        let sieved = st_fwd.offset + st_bck.offset;
        if sieved % (10 << 20) == 0 {
            println!(
                "Sieved {}M found {} smooths (cofactors: {} combined, {} pending)",
                sieved >> 20,
                relations.len(),
                extras,
                larges.len(),
            );
        }
        // For small n the sieve must stop quickly:
        // test whether we already have enough relations.
        if n.bits() < 64 || relations.len() >= target {
            let gap = relation_gap(&relations);
            if gap == 0 {
                println!("Found enough relations");
                break;
            } else {
                println!("Need {} additional relations", gap);
                target = relations.len() + gap + 10;
            }
        }
    }
    let sieved = st_fwd.offset + st_bck.offset;
    println!(
        "Sieved {:.1}M found {} smooths (cofactors: {} combined, {} pending)",
        (sieved as f64) / ((1 << 20) as f64),
        relations.len(),
        extras,
        larges.len(),
    );
    relations
}

struct SieveQS {
    n: Uint,
    nsqrt: Uint,
}

struct StateQS<'a> {
    offset: u64,
    idx15: usize, // Offset of prime > 32768
    primes: &'a [&'a Prime],
    logs: &'a [u8],
    // The MSB of the offset for each cursor.
    hi: Vec<u8>,
    // The LSB of the offset for each cursor.
    lo: Vec<u16>,
}

impl<'a> StateQS<'a> {
    fn next_block(&mut self) {
        self.offset += BLOCK_SIZE as u64;
        for i in 0..self.hi.len() {
            let m = self.hi[i];
            if m > 0 {
                self.hi[i] = m - 1;
            }
        }
    }
}

fn sieve_block(s: &SieveQS, st: &mut StateQS, backward: bool) -> (Vec<Relation>, Vec<Relation>) {
    let len: usize = BLOCK_SIZE;
    let mut blk = vec![0u8; len];
    let starts = st.lo.clone();
    unsafe {
        for i in 0..st.idx15 {
            let i = i as usize;
            let p = st.primes.get_unchecked(i).p;
            // Small primes always have a hit.
            debug_assert!(st.hi[i] == 0);
            let mut off: usize = *st.lo.get_unchecked(i) as usize;
            let size = *st.logs.get_unchecked(i);
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
            st.lo[i] = (off % BLOCK_SIZE) as u16;
        }
    }
    for i in st.idx15..st.primes.len() {
        // Large primes have at most 1 hit.
        if st.hi[i] != 0 {
            continue;
        }
        let i = i as usize;
        let p = st.primes[i].p;
        blk[st.lo[i] as usize] += st.logs[i];
        let off = st.lo[i] as usize + p as usize;
        debug_assert!(off > BLOCK_SIZE);
        st.hi[i] = (off / BLOCK_SIZE) as u8;
        st.lo[i] = (off % BLOCK_SIZE) as u16;
    }
    sieve_result(s, st, &blk, &starts[..], backward)
}

fn sieve_result(
    s: &SieveQS,
    st: &StateQS,
    blk: &[u8],
    starts: &[u16],
    backward: bool,
) -> (Vec<Relation>, Vec<Relation>) {
    assert_eq!(starts.len(), st.primes.len());
    let len: usize = BLOCK_SIZE;
    let offset = st.offset;
    let maxprime = st.primes.last().unwrap().p;
    let maxlarge = maxprime * large_prime_factor(&s.n);
    let mut result = vec![];
    let mut extras = vec![];
    let magnitude = u64::BITS - u64::leading_zeros(std::cmp::max(offset, len as u64));
    let target = s.n.bits() / 2 + magnitude - maxlarge.bits();
    let n = &s.n;
    for i in 0..len as u64 {
        if blk[i as usize] as u32 >= target {
            let mut factors: Vec<(i64, u64)> = Vec::with_capacity(20);
            let x = if !backward {
                Int::from_bits(s.nsqrt) + Int::from((offset + i) as i64)
            } else {
                Int::from_bits(s.nsqrt) - Int::from((offset + i + 1) as i64)
            };
            let candidate: Int = x * x - Int::from_bits(*n);
            let cabs = candidate.abs().to_bits() % n;
            if candidate.is_negative() {
                factors.push((-1, 1));
            }
            let mut cofactor: Uint = cabs;
            let mut tmp_p = 0;
            let mut tmp_r = 0;
            for (idx, item) in st.primes.iter().enumerate() {
                if item.p != tmp_p {
                    tmp_p = item.p;
                    tmp_r = item.div.divmod64(i as u64).1;
                }
                if tmp_r == starts[idx] as u64 {
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
                    if exp > 0 {
                        factors.push((item.p as i64, exp));
                    }
                }
            }
            let Some(cofactor) = cofactor.to_u64() else { continue };
            let sabs = (x.abs().to_bits()) % n;
            let x = if candidate.is_negative() {
                n - sabs
            } else {
                sabs
            };
            if cofactor == 1 {
                //println!("i={} smooth {}", i, cabs);
                result.push(Relation {
                    x,
                    cofactor: 1,
                    factors,
                });
            } else if cofactor < maxlarge {
                //println!("i={} smooth {} cofactor {}", i, cabs, cofactor);
                extras.push(Relation {
                    x,
                    cofactor,
                    factors,
                });
            }
        }
    }
    (result, extras)
}
