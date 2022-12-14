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
use crate::fbase::{self, Prime};
use crate::params::{self, large_prime_factor, BLOCK_SIZE};
use crate::relations::{combine_large_relation, relation_gap, Relation};
use crate::sieve::Sieve;
use crate::{Int, Uint};

pub fn qsieve(n: Uint, fb: Option<u32>, tpool: Option<&rayon::ThreadPool>) -> Vec<Relation> {
    // Choose factor base. Sieve twice the number of primes
    // (n will be a quadratic residue for only half of them)
    let fb = fb.unwrap_or(params::factor_base_size(&n));
    let primes = fbase::primes(2 * fb);
    eprintln!("Smoothness bound {}", primes.last().unwrap());
    let primes: Vec<Prime> = fbase::prepare_factor_base(&n, &primes);
    let primes = &primes[..];
    eprintln!("All primes {}", primes.len());
    // Prepare factor base
    let smallprimes: Vec<u64> = primes.iter().map(|f| f.p).take(10).collect();
    eprintln!("Factor base size {} ({:?})", primes.len(), smallprimes);

    // Prepare sieve
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

    let qs = SieveQS::new(n, primes);
    // Construct 2 initial states, forward and backwards.
    let (mut s_fwd, mut s_bck) = qs.init(None);
    loop {
        let (r1, r2) = if let Some(pool) = tpool {
            pool.install(|| {
                rayon::join(
                    || sieve_block(&qs, &mut s_fwd, false),
                    || sieve_block(&qs, &mut s_bck, true),
                )
            })
        } else {
            (
                sieve_block(&qs, &mut s_fwd, false),
                sieve_block(&qs, &mut s_bck, true),
            )
        };
        let (mut found, foundlarge) = r1;
        if found.len() > primes.len() + 16 {
            // Too many relations! May happen for very small inputs.
            relations.extend_from_slice(&mut found[..primes.len() + 16]);
            let gap = relation_gap(&relations);
            if gap == 0 {
                eprintln!("Found enough relations");
                break;
            } else {
                eprintln!("Need {} additional relations", gap);
                target = relations.len() + gap + 16;
            }
        }
        relations.append(&mut found);
        for r in foundlarge {
            if let Some(rr) = combine_large_relation(&mut larges, &r, &n) {
                relations.push(rr);
                extras += 1;
            }
        }
        let (mut found, foundlarge) = r2;
        relations.append(&mut found);
        for r in foundlarge {
            if let Some(rr) = combine_large_relation(&mut larges, &r, &n) {
                relations.push(rr);
                extras += 1;
            }
        }

        // Next block
        s_fwd.next_block();
        s_bck.next_block();
        if s_fwd.blk_no == qs.nblocks() {
            assert_eq!(s_bck.blk_no, qs.nblocks());
            (s_fwd, s_bck) = qs.init(Some((s_fwd, s_bck)));
        }

        let sieved = s_fwd.offset + s_bck.offset;
        let do_print = if sieved > (200 << 20) {
            sieved % (50 << 20) == 0
        } else {
            sieved % (10 << 20) == 0
        };
        if do_print {
            eprintln!(
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
                eprintln!("Found enough relations");
                break;
            } else {
                eprintln!("Need {} additional relations", gap);
                target = relations.len() + gap + 10;
            }
        }
    }
    let sieved = s_fwd.offset + s_bck.offset;
    eprintln!(
        "Sieved {:.1}M found {} smooths (cofactors: {} combined, {} pending)",
        (sieved as f64) / ((1 << 20) as f64),
        relations.len(),
        extras,
        larges.len(),
    );
    relations
}

pub struct SieveQS<'a> {
    n: Uint,
    nsqrt: Uint,
    // Precomputed nsqrt_mods modulo the factor base.
    nsqrt_mods: Vec<u32>,
    primes: &'a [Prime],
}

impl<'a> SieveQS<'a> {
    pub fn new(n: Uint, primes: &'a [Prime]) -> Self {
        let nsqrt = isqrt(n);
        let nsqrt_mods: Vec<u32> = primes
            .iter()
            .map(|p| p.div.mod_uint(&nsqrt) as u32)
            .collect();
        SieveQS {
            n,
            nsqrt,
            primes,
            nsqrt_mods,
        }
    }

    // Rehash large primes every xx blocks
    fn nblocks(&self) -> usize {
        // mpqs_interval_size / BLOCK_SIZE
        let sz = self.n.bits();
        match sz {
            0..=119 => 8,                // doesn't matter
            120..=330 => 8 << (sz / 80), // 8..128
            _ => 128,
        }
    }

    pub fn init(&self, sieves: Option<(Sieve<'a>, Sieve<'a>)>) -> (Sieve<'a>, Sieve<'a>) {
        let f1 = |pidx, p, offset| {
            // Return r such that nsqrt + offset + r is a root of n.
            let p: &'a Prime = p;
            let base = self.nsqrt_mods[pidx] as u64;
            let off: u64 = p.div.modi64(offset);
            let mut s1 = p.r + 2 * p.p - off - base;
            let mut s2 = 3 * p.p - p.r - off - base;
            while s1 >= p.p {
                s1 -= p.p
            }
            while s2 >= p.p {
                s2 -= p.p
            }
            [Some(s1 as u32), Some(s2 as u32)]
        };
        let f2 = |pidx, p, offset| {
            // Return r such that nsqrt - (offset + r + 1) is a root of n.
            let p: &'a Prime = p;
            let base = self.nsqrt_mods[pidx] as u64;
            let off: u64 = p.div.modi64(offset);
            let mut s1 = 2 * p.p + base - p.r - off - 1;
            let mut s2 = p.p + base + p.r - off - 1;
            while s1 >= p.p {
                s1 -= p.p
            }
            while s2 >= p.p {
                s2 -= p.p
            }
            [Some(s1 as u32), Some(s2 as u32)]
        };

        if let Some((mut s1, mut s2)) = sieves {
            s1.rehash(f1);
            s2.rehash(f2);
            (s1, s2)
        } else {
            let s1 = Sieve::new(0, self.nblocks(), self.primes, f1);
            let s2 = Sieve::new(0, self.nblocks(), self.primes, f2);
            (s1, s2)
        }
    }
}

fn sieve_block(s: &SieveQS, st: &mut Sieve, backward: bool) -> (Vec<Relation>, Vec<Relation>) {
    st.sieve_block();

    let len: usize = BLOCK_SIZE;
    let offset = st.offset;
    let maxprime = st.primes.last().unwrap().p;
    let maxlarge = maxprime * large_prime_factor(&s.n);
    let mut result = vec![];
    let mut extras = vec![];
    let magnitude =
        u64::BITS - u64::leading_zeros(std::cmp::max(st.offset.abs() as u64, len as u64));
    let target = s.n.bits() / 2 + magnitude - maxlarge.bits();
    assert!(target < 256);
    let n = &s.n;
    let (idxs, facss) = st.smooths(target as u8);
    for (i, facs) in idxs.into_iter().zip(facss) {
        let x = if !backward {
            Int::from_bits(s.nsqrt) + Int::from(offset as i64 + i as i64)
        } else {
            Int::from_bits(s.nsqrt) - Int::from(offset as i64 + i as i64 + 1)
        };
        let candidate: Int = x * x - Int::from_bits(*n);
        let mut factors: Vec<(i64, u64)> = vec![];
        if candidate.is_negative() {
            factors.push((-1, 1));
        }
        let cabs = candidate.abs().to_bits();
        let mut cofactor: Uint = cabs;
        for p in facs {
            let mut exp = 0;
            loop {
                let (q, r) = p.div.divmod_uint(&cofactor);
                if r == 0 {
                    cofactor = q;
                    exp += 1;
                } else {
                    break;
                }
            }
            factors.push((p.p as i64, exp));
        }
        let Some(cofactor) = cofactor.to_u64() else { continue };
        //println!("i={} smooth {} cofactor {}", i, cabs, cofactor);
        let rel = Relation {
            x: x.abs().to_bits(),
            cofactor,
            factors,
        };
        debug_assert!(
            rel.verify(&s.n),
            "INTERNAL ERROR: failed relation check {:?}",
            rel
        );
        if cofactor == 1 {
            result.push(rel)
        } else {
            extras.push(rel)
        }
    }
    (result, extras)
}
